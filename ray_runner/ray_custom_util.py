from typing import Dict, Optional, TypeVar

import numpy as np
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from ray.rllib import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from ray_runner.bigtwo_multi_agent import BigTwoMultiAgentEnv

T = TypeVar("T")
U = TypeVar("U")


def setup_tracing() -> None:
    resource = Resource(attributes={SERVICE_NAME: "rayrunner"})

    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
        udp_split_oversized_batches=True,
    )

    processor = BatchSpanProcessor(jaeger_exporter)
    provider = TracerProvider(resource=resource)

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


class CustomMetricCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        hands_played, actions_attempted = [], []
        card_length_played = {}
        for unwrapped_env in base_env.get_unwrapped():
            if not isinstance(unwrapped_env, BigTwoMultiAgentEnv):
                continue

            hands_played.append(unwrapped_env.hands_played())
            actions_attempted.append(unwrapped_env.actions_attempted())

            for player_idx, action in unwrapped_env.base_env.past_actions:
                card_length_played[len(action)] = (
                    card_length_played.get(len(action), 0) + 1
                )

        episode.custom_metrics["hands_played"] = np.mean(hands_played)
        episode.custom_metrics["actions_attempted"] = np.mean(actions_attempted)

        for hand_length, counter in card_length_played.items():
            episode.custom_metrics[f"hand_length_{hand_length}"] = counter


class TracingCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.episode_start_span: Optional[Span] = None

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        tracer = trace.get_tracer(__name__)
        self.episode_start_span = tracer.start_span("on_episode_start")

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        if self.episode_start_span:
            self.episode_start_span.end()

    def on_sample_end(
        self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs
    ) -> None:
        if self.episode_start_span and self.episode_start_span.end_time is None:
            self.episode_start_span.end()

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        # print("learn_on_batch:", train_batch.count, datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        pass
