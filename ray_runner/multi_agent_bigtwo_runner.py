import logging
from abc import ABC
from typing import List, Dict, Union

import numpy as np
import ray
import tensorflow as tf
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from ray import shutdown, tune
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo import PPO
from ray.rllib.execution import (
    synchronous_parallel_sample,
    train_one_step,
)
from ray.rllib.execution.common import (
    LEARN_ON_BATCH_TIMER,
)
from ray.rllib.execution.rollout_ops import standardize_fields as standardize_fields_fn
from ray.rllib.execution.train_ops import multi_gpu_train_one_step
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    NUM_ENV_STEPS_TRAINED,
    NUM_AGENT_STEPS_TRAINED,
)
from ray.rllib.utils.metrics.learner_info import (
    LearnerInfoBuilder,
    LEARNER_STATS_KEY,
)
from ray.rllib.utils.sgd import minibatches, standardized
from ray.rllib.utils.typing import (
    SampleBatchType,
    TensorType,
    ResultDict,
)
from ray.tune.registry import register_env
from ray.util import log_once

from ray_runner.bigtwo_multi_agent import BigTwoMultiAgentEnv
from ray_runner.ray_custom_util import (
    CustomMetricCallback,
)

TRACE_PROPAGATOR = TraceContextTextMapPropagator()
LOGGER = logging.getLogger(__name__)
TRACE_CARRIER_KEY = "trace_carrier"


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
        samples: Batch of samples to optimize.
        policies: Dictionary of policies to optimize.
        local_worker: Master rollout worker instance.
        num_sgd_iter: Number of epochs of optimization to take.
        sgd_minibatch_size: Size of minibatches to use for optimization.
        standardize_fields: List of sample field names that should be
            normalized prior to optimization.

    Returns:
        averaged info fetches over the last SGD epoch taken.
    """

    # Handle everything as if multi-agent.
    samples = samples.as_multi_agent()

    # Use LearnerInfoBuilder as a unified way to build the final
    # results dict from `learn_on_loaded_batch` call(s).
    # This makes sure results dicts always have the same structure
    # no matter the setup (multi-GPU, multi-agent, minibatch SGD,
    # tf vs torch).
    learner_info_builder = LearnerInfoBuilder(num_devices=1)
    tracer = trace.get_tracer(__name__)
    for policy_id, policy in policies.items():
        if policy_id not in samples.policy_batches:
            continue

        batch = samples.policy_batches[policy_id]
        for field in standardize_fields:
            batch[field] = standardized(batch[field])

        # Check to make sure that the sgd_minibatch_size is not smaller
        # than max_seq_len otherwise this will cause indexing errors while
        # performing sgd when using a RNN or Attention model
        if (
            policy.is_recurrent()
            and policy.config["model"]["max_seq_len"] > sgd_minibatch_size
        ):
            raise ValueError(
                "`sgd_minibatch_size` ({}) cannot be smaller than"
                "`max_seq_len` ({}).".format(
                    sgd_minibatch_size, policy.config["model"]["max_seq_len"]
                )
            )

        kl_target = 0.01

        attr = {
            "num_sgd_iter": num_sgd_iter,
            "sgd_minibatch_size": sgd_minibatch_size,
            "total_batch_size": len(batch),
        }

        with tracer.start_as_current_span(
            f"custom_do_minibatch_sgd_{policy_id}", attributes=attr
        ) as sgd_span:
            for i in range(num_sgd_iter):
                with tracer.start_as_current_span(f"iter_{i}"):
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
                        LOGGER.info(
                            f"Early stopping at step {i} due to reaching max kl: {avg_kl}"
                        )
                        break

    learner_info = learner_info_builder.finalize()
    return learner_info


def custom_train_one_step(algorithm, train_batch, policies_to_train=None) -> Dict:
    config = algorithm.config
    workers = algorithm.workers
    local_worker = workers.local_worker()
    num_sgd_iter = config.get("num_sgd_iter", 1)
    sgd_minibatch_size = config.get("sgd_minibatch_size", 0)

    learn_timer = algorithm._timers[LEARN_ON_BATCH_TIMER]
    with learn_timer:
        # Subsample minibatches (size=`sgd_minibatch_size`) from the
        # train batch and loop through train batch `num_sgd_iter` times.
        if num_sgd_iter > 1 or sgd_minibatch_size > 0:
            info = custom_do_minibatch_sgd(
                train_batch,
                {
                    pid: local_worker.get_policy(pid)
                    for pid in policies_to_train
                    or local_worker.get_policies_to_train(train_batch)
                },
                local_worker,
                num_sgd_iter,
                sgd_minibatch_size,
                [],
            )
        # Single update step using train batch.
        else:
            info = local_worker.learn_on_batch(train_batch)

    learn_timer.push_units_processed(train_batch.count)
    algorithm._counters[NUM_ENV_STEPS_TRAINED] += train_batch.count
    algorithm._counters[NUM_AGENT_STEPS_TRAINED] += train_batch.agent_steps()

    if algorithm.reward_estimators:
        info[DEFAULT_POLICY_ID]["off_policy_estimation"] = {}
        for name, estimator in algorithm.reward_estimators.items():
            info[DEFAULT_POLICY_ID]["off_policy_estimation"][name] = estimator.train(
                train_batch
            )
    return info


class TracedPPO(PPO, ABC):
    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("synchronous_parallel_sample"):
            if self._by_agent_steps:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config["train_batch_size"],
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_env_steps=self.config["train_batch_size"],
                )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        with tracer.start_as_current_span("standardize_fields"):
            train_batch = standardize_fields_fn(train_batch, ["advantages"])

        with tracer.start_as_current_span("train_one_step"):
            # Train
            if self.config["simple_optimizer"]:
                train_results = custom_train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with tracer.start_as_current_span("sync_weights"):
            if self.workers.remote_workers():
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)

        # For each policy: update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            self._warn_if_excessively_high_value_function_loss(policy_id, policy_info)
            self._warn_if_bad_clipping_config(policy_id, train_batch)

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results

    def _warn_if_excessively_high_value_function_loss(
        self, policy_id: str, policy_info
    ) -> None:
        scaled_vf_loss = (
            self.config["vf_loss_coeff"] * policy_info[LEARNER_STATS_KEY]["vf_loss"]
        )
        policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
        if (
            log_once("ppo_warned_lr_ratio")
            and self.config.get("model", {}).get("vf_share_layers")
            and scaled_vf_loss > 100
        ):
            LOGGER.warning(
                "The magnitude of your value function loss for policy: {} is "
                "extremely large ({}) compared to the policy loss ({}). This "
                "can prevent the policy from learning. Consider scaling down "
                "the VF loss by reducing vf_loss_coeff, or disabling "
                "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
            )

    def _warn_if_bad_clipping_config(
        self, policy_id: str, train_batch: Union[List[SampleBatchType], SampleBatchType]
    ) -> None:
        train_batch.policy_batches[policy_id].set_get_interceptor(None)
        mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
        if (
            log_once("ppo_warned_vf_clip")
            and mean_reward > self.config["vf_clip_param"]
        ):
            self.warned_vf_clip = True
            LOGGER.warning(
                f"The mean reward returned from the environment is {mean_reward}"
                f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                f" Consider increasing it for policy: {policy_id} to improve"
                " value function convergence."
            )


def train_multi_agent():
    shutdown()
    env_name = "bigtwo_v1"

    register_env(env_name, lambda _: BigTwoMultiAgentEnv())
    ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

    test_env = BigTwoMultiAgentEnv()

    default_policy_id = "policy_0"
    bigtwo_policies = {
        default_policy_id: (
            None,
            test_env.observation_space,
            test_env.action_space,
            {
                "model": {
                    "custom_model": "pa_model",
                },
                "gamma": 0.99,
            },
        )
    }

    ppo_config = (
        PPOConfig()
        .environment(env=env_name)
        .framework(framework="tf2", eager_tracing=True)
        .resources(num_gpus=1)
        .experimental(_disable_preprocessor_api=True)
        .multi_agent(
            policies=bigtwo_policies,
            policy_mapping_fn=lambda agent_id: default_policy_id,
        )
        .rollouts(
            num_rollout_workers=4,
            num_envs_per_worker=1,
            batch_mode="truncate_episodes",
            compress_observations=True,
            rollout_fragment_length=512,
        )
        .callbacks(MultiCallbacks([CustomMetricCallback]))
        .training(
            lr=0.0001,
            use_gae=True,
            gamma=0.99,
            lambda_=0.9,
            kl_coeff=0.2,
            kl_target=0.01,
            sgd_minibatch_size=512,
            num_sgd_iter=30,
            train_batch_size=4096,
            clip_param=0.3,
            grad_clip=None,
        )
        .debugging(log_level="INFO")
    )

    ray.init(_tracing_startup_hook="ray_custom_util:setup_tracing")
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("ppo_training"):
        _ = tune.run(
            TracedPPO,
            name="TracedPPO",
            stop={"timesteps_total": 41000},
            checkpoint_freq=10000,
            local_dir="./temp_results/" + env_name,
            config=ppo_config.to_dict(),
            checkpoint_at_end=True,
        )


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
