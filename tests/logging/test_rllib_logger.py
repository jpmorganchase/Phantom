from dataclasses import dataclass, field

from phantom.metrics import Metric
from phantom.utils.rllib.train import RLlibMetricLogger


@dataclass
class _MockEpisode:
    user_data: dict = field(default_factory=dict)
    custom_metrics: dict = field(default_factory=dict)
    media: dict = field(default_factory=dict)


class _MockBaseEnv:
    class _MockEnv:
        class _MockNetwork:
            class _MockGraph:
                @property
                def edges(self):
                    return [("A", "B"), ("B", "A"), ("ENV_AGENT", "A")]

            @property
            def graph(self):
                return self._MockGraph()

        @property
        def network(self):
            return self._MockNetwork()

    def __init__(self, env_class=_MockEnv):
        self.envs = [env_class()]


class DummyMetric(Metric):
    def __init__(self, value: int):
        self.value = value

    def extract(self, _env) -> int:
        return self.value


def test_RLlibMetricLogger():
    episode = _MockEpisode()
    base_env = _MockBaseEnv()

    callback = RLlibMetricLogger({"dummy_metric": DummyMetric(1)})()

    callback.on_episode_start(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )

    callback.on_episode_step(
        worker=None, base_env=base_env, episode=episode, env_index=0
    )
    assert episode.user_data["dummy_metric"] == [1]

    callback.on_episode_end(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )
    assert episode.custom_metrics["dummy_metric"] == 1
