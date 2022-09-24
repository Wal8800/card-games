import time
from typing import List, TypeVar

import numpy as np
import ray
import tensorflow as tf
from opentelemetry import trace
from opentelemetry.propagators.textmap import CarrierT
from ray import shutdown, tune
from ray.rllib import SampleBatch
from ray.rllib.agents import MultiCallbacks
from ray.rllib.agents.ppo import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import (
    DEFAULT_CONFIG,
    UpdateKL,
    get_policy_class,
    validate_config,
    warn_about_bad_reward_scales,
)
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution import (
    ConcatBatches,
    ParallelRollouts,
    SelectExperiences,
    StandardizeFields,
    StandardMetricsReporting,
)
from ray.rllib.execution.common import (
    AGENT_STEPS_TRAINED_COUNTER,
    LEARN_ON_BATCH_TIMER,
    STEPS_TRAINED_COUNTER,
    WORKER_UPDATE_TIMER,
    _check_sample_batch_type,
    _get_global_vars,
    _get_shared_metrics,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LearnerInfoBuilder
from ray.rllib.utils.sgd import minibatches, standardized
from ray.rllib.utils.typing import (
    PolicyID,
    SampleBatchType,
    TensorType,
    TrainerConfigDict,
)
from ray.tune.registry import register_env
from ray.util.iter import LocalIterator

from ray_runner.bigtwo_multi_agent import BigTwoMultiAgentEnv
from ray_runner.ray_custom_util import (
    CustomMetricCallback,
    TraceLocalIterator,
    TraceOps,
    trace_propagator,
)


class ParametricActionsModel(TFModelV2):
    def value_function(self) -> TensorType:
        return self.action_model.value_function()

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        self.action_model = FullyConnectedNetwork(
            obs_space["game_obs"], action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_logits, _ = self.action_model({"obs": input_dict["obs"]["game_obs"]})
        # Mask out invalid actions (use tf.float32.min for stability)
        # inf_mask_alt = tf.maximum(tf.math.log(tf.cast(action_mask, action_logits.dtype)), action_logits.dtype.min)
        inf_mask = (1.0 - tf.cast(action_mask, action_logits.dtype)) * -1e9
        # print(inf_mask.shape, inf_mask.dtype, inf_mask_alt.shape, inf_mask_alt.dtype)
        """
        (1 - 1) * -1e9 = 0 for valid actions
        (1 - 0) * -1e9 = -1e9 for invalid actions
        """
        return action_logits + inf_mask, state


def custom_do_minibatch_sgd(
    samples,
    policies,
    local_worker,
    num_sgd_iter,
    sgd_minibatch_size,
    standardize_fields,
):
    """Execute minibatch SGD.

    Args:
        samples (SampleBatch): Batch of samples to optimize.
        policies (dict): Dictionary of policies to optimize.
        local_worker (RolloutWorker): Master rollout worker instance.
        num_sgd_iter (int): Number of epochs of optimization to take.
        sgd_minibatch_size (int): Size of minibatches to use for optimization.
        standardize_fields (list): List of sample field names that should be
            normalized prior to optimization.

    Returns:
        averaged info fetches over the last SGD epoch taken.
    """
    if isinstance(samples, SampleBatch):
        samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)

    # Use LearnerInfoBuilder as a unified way to build the final
    # results dict from `learn_on_loaded_batch` call(s).
    # This makes sure results dicts always have the same structure
    # no matter the setup (multi-GPU, multi-agent, minibatch SGD,
    # tf vs torch).
    learner_info_builder = LearnerInfoBuilder(num_devices=1)
    tracer = trace.get_tracer(__name__)
    for policy_id in policies.keys():
        if policy_id not in samples.policy_batches:
            continue

        batch = samples.policy_batches[policy_id]
        for field in standardize_fields:
            batch[field] = standardized(batch[field])

        attr = {
            "num_sgd_iter": num_sgd_iter,
            "sgd_minibatch_size": sgd_minibatch_size,
            "total_batch_size": len(batch),
        }

        kl_target = 0.01

        with tracer.start_as_current_span(
            f"custom_do_minibatch_sgd_{policy_id}", attributes=attr
        ) as sgd_span:
            for i in range(num_sgd_iter):
                mb_kl = []
                for minibatch in minibatches(batch, sgd_minibatch_size):
                    results = (
                        local_worker.learn_on_batch(
                            MultiAgentBatch({policy_id: minibatch}, minibatch.count)
                        )
                    )[policy_id]
                    learner_info_builder.add_learn_on_batch_results(results, policy_id)

                    mb_kl.append(results["learner_stats"]["kl"])

                avg_kl = np.mean(mb_kl)
                if avg_kl > 1.5 * kl_target:
                    sgd_span.set_attribute("early_stopping_step", i)
                    print(
                        f"Early stopping at step {i} due to reaching max kl: {avg_kl}"
                    )
                    break

    learner_info = learner_info_builder.finalize()
    return learner_info


class CustomTrainOneStep:
    """Callable that improves the policy and updates workers.

    This should be used with the .for_each() operator. A tuple of the input
    and learner stats will be returned.

    Examples:
        >>> rollouts = ParallelRollouts(...)
        >>> train_op = rollouts.for_each(CustomTrainOneStep(workers))
        >>> print(next(train_op))  # This trains the policy on one batch.
        SampleBatch(...), {"learner_stats": ...}

    Updates the STEPS_TRAINED_COUNTER counter and LEARNER_INFO field in the
    local iterator context.
    """

    def __init__(
        self,
        workers: WorkerSet,
        policies: List[PolicyID] = frozenset([]),
        num_sgd_iter: int = 1,
        sgd_minibatch_size: int = 0,
    ):
        self.workers = workers
        self.local_worker = workers.local_worker()
        self.policies = policies
        self.num_sgd_iter = num_sgd_iter
        self.sgd_minibatch_size = sgd_minibatch_size

    def __call__(self, batch: SampleBatchType) -> (SampleBatchType, List[dict]):
        _check_sample_batch_type(batch)
        metrics = _get_shared_metrics()
        learn_timer = metrics.timers[LEARN_ON_BATCH_TIMER]
        with learn_timer:
            # Subsample minibatches (size=`sgd_minibatch_size`) from the
            # train batch and loop through train batch `num_sgd_iter` times.
            if self.num_sgd_iter > 1 or self.sgd_minibatch_size > 0:
                lw = self.workers.local_worker()
                learner_info = custom_do_minibatch_sgd(
                    batch,
                    {
                        pid: lw.get_policy(pid)
                        for pid in self.policies or self.local_worker.policies_to_train
                    },
                    lw,
                    self.num_sgd_iter,
                    self.sgd_minibatch_size,
                    [],
                )
            # Single update step using train batch.
            else:
                learner_info = self.workers.local_worker().learn_on_batch(batch)

            metrics.info[LEARNER_INFO] = learner_info
            learn_timer.push_units_processed(batch.count)
        metrics.counters[STEPS_TRAINED_COUNTER] += batch.count
        if isinstance(batch, MultiAgentBatch):
            metrics.counters[AGENT_STEPS_TRAINED_COUNTER] += batch.agent_steps()
        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with metrics.timers[WORKER_UPDATE_TIMER]:
                weights = ray.put(
                    self.workers.local_worker().get_weights(
                        self.policies or self.local_worker.policies_to_train
                    )
                )
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights, _get_global_vars())
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())
        return batch, learner_info


def create_execution_plan(carrier: CarrierT):
    def result_plan(
        workers: WorkerSet, config: TrainerConfigDict
    ) -> LocalIterator[dict]:
        rollouts = TraceLocalIterator(
            "rollout", carrier, ParallelRollouts(workers, mode="bulk_sync")
        )

        # Collect batches for the trainable policies.
        rollouts = rollouts.for_each(
            TraceOps(
                "select_experiences",
                carrier,
                SelectExperiences(workers.trainable_policies()),
            )
        )

        # Concatenate the SampleBatches into one.
        rollouts = rollouts.combine(
            TraceOps(
                "concat_batches",
                carrier,
                ConcatBatches(
                    min_batch_size=config["train_batch_size"],
                    count_steps_by=config["multiagent"]["count_steps_by"],
                ),
            )
        )

        # Standardize advantages.
        rollouts = rollouts.for_each(
            TraceOps("standardize_fields", carrier, StandardizeFields(["advantages"])),
        )

        # Perform one training step on the combined + standardized batch.
        if not config["simple_optimizer"]:
            raise NotImplementedError(
                "haven't implemented training when simple optimizer is false"
            )

        train_op = rollouts.for_each(
            TraceOps(
                "train_one_step",
                carrier,
                CustomTrainOneStep(
                    workers,
                    num_sgd_iter=config["num_sgd_iter"],
                    sgd_minibatch_size=config["sgd_minibatch_size"],
                ),
            )
        )

        # Update KL after each round of training.
        train_op = train_op.for_each(lambda t: t[1]).for_each(
            TraceOps("update_kl", carrier, UpdateKL(workers))
        )

        # Warn about bad reward scales and return training metrics.
        reporting = StandardMetricsReporting(train_op, workers, config).for_each(
            lambda result: warn_about_bad_reward_scales(config, result)
        )

        return reporting

    return result_plan


def config_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def train_multi_agent():
    config_gpu()

    shutdown()
    env_name = "bigtwo_v1"

    def create_env(config):
        return BigTwoMultiAgentEnv()

    register_env(env_name, create_env)
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    test_env = BigTwoMultiAgentEnv()
    gen_obs_space = test_env.observation_space
    gen_act_space = test_env.action_space

    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "pa_model",
            },
            "gamma": 0.99,
        }
        return None, gen_obs_space, gen_act_space, config

    bigtwo_policies = {"policy_0": gen_policy(0)}

    policy_ids = list(bigtwo_policies.keys())
    num_workers = 10

    run_config = {
        # Environment specific
        "env": env_name,
        # General
        "callbacks": MultiCallbacks([CustomMetricCallback]),
        "_disable_preprocessor_api": True,
        "log_level": "ERROR",
        "framework": "tf2",
        "eager_tracing": True,
        "num_gpus": 1,
        "num_workers": 4,
        # "num_gpus_per_worker": num_gpus_per_worker,
        "num_envs_per_worker": 1,
        "compress_observations": False,
        "batch_mode": "truncate_episodes",
        # 'use_critic': True,
        "use_gae": True,
        "lambda": 0.9,
        "gamma": 0.99,
        "kl_coeff": 0,
        "kl_target": 0.01,
        "clip_param": 0.3,
        "grad_clip": None,
        # "entropy_coeff": 0.1,
        # "vf_loss_coeff": 0.25,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 80,  # epoch
        "rollout_fragment_length": 512,
        "train_batch_size": 4096,
        "lr": 1e-04,
        "clip_actions": True,
        # Method specific
        "multiagent": {
            "policies": bigtwo_policies,
            "policy_mapping_fn": (lambda agent_id: policy_ids[0]),
        },
    }

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("ppo training"):
        carrier = {}
        trace_propagator.inject(carrier=carrier)

        CustomPPOTrainer = build_trainer(
            name="CustomPPO",
            default_config=DEFAULT_CONFIG,
            validate_config=validate_config,
            default_policy=PPOTFPolicy,
            get_policy_class=get_policy_class,
            execution_plan=create_execution_plan(carrier=carrier),
        )

        start_time = time.time()
        # ray.init(_tracing_startup_hook="ray_custom_util:setup_tracing")
        _ = tune.run(
            CustomPPOTrainer,
            name="CustomPPO",
            stop={"timesteps_total": 4000},
            checkpoint_freq=100,
            local_dir="./results/" + env_name,
            config=run_config,
            checkpoint_at_end=True,
        )

    print(start_time)
    print(time.time() - start_time)


if __name__ == "__main__":
    train_multi_agent()

"""
    learn_throughput: 500.807
    learn_time_ms: 16293.693
    sample_throughput: 557.321
    sample_time_ms: 14641.477
    update_time_ms: 5.441
    
    2022-07-30 13:48:42,685	INFO tune.py:630 -- Total run time: 118.68 seconds (118.37 seconds for the tuning loop).
    1659145602.8361354
    119.86279034614563
"""
