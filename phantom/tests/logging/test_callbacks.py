from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import phantom as ph
from phantom.logging.callbacks import (
    MetricsLoggerCallbacks,
    NetworkPlotCallbacks,
    _fig_to_ndarray,
)
from phantom.logging.metrics import Metric


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
                    return [("A", "B"), ("B", "A"), (ph.env.EnvironmentActor.ID, "A")]

            @property
            def graph(self):
                return self._MockGraph()

        @property
        def network(self):
            return self._MockNetwork()

    def __init__(self, env_class=_MockEnv):
        self.envs = [env_class()]


@dataclass
class _MockWorker:
    worker_index: int = 1


class DummyMetric(Metric):
    def __init__(self, value: int):
        self.value = value

    def extract(self, _env) -> int:
        return self.value


def test_MetricsLoggerCallbacks():
    episode = _MockEpisode()
    base_env = _MockBaseEnv()

    metrics = {"dummy_metric": DummyMetric(1)}

    logger_id = "TEST"
    callback = MetricsLoggerCallbacks(logger_id=logger_id, metrics=metrics)()

    callback.on_episode_start(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )
    assert logger_id in episode.user_data

    callback.on_episode_step(
        worker=None, base_env=base_env, episode=episode, env_index=0
    )
    assert episode.user_data[logger_id].logs["dummy_metric"] == [1]

    callback.on_episode_end(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )
    assert episode.custom_metrics["dummy_metric"] == 1


def test_fig_to_ndarray():
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(
        xlabel="time (s)",
        ylabel="voltage (mV)",
        title="About as simple as it gets, folks",
    )
    ax.grid()

    arr = _fig_to_ndarray(fig)

    assert isinstance(arr, np.ndarray)
    assert len(arr.shape) == 4
    assert arr.shape[0] == 1
    assert arr.shape[1] == 4  # RGBA


def test_NetworkPlotCallbacks():
    episode = _MockEpisode()
    base_env = _MockBaseEnv()
    worker = _MockWorker()

    callback = NetworkPlotCallbacks()

    callback.on_episode_start(
        worker=worker, base_env=base_env, policies=None, episode=episode, env_index=0
    )
    assert "network/worker1" in episode.media
