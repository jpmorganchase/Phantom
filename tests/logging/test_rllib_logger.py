from phantom import PhantomEnv, Network
from phantom.utils.rllib.train import RLlibMetricLogger

from . import MockBaseEnv, MockEpisode, MockMetric


def test_RLlibMetricLogger():
    episode = MockEpisode()
    base_env = MockBaseEnv(PhantomEnv(Network()))

    callback = RLlibMetricLogger({"test_metric": MockMetric(1)})()

    callback.on_episode_start(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )

    callback.on_episode_step(
        worker=None, base_env=base_env, episode=episode, env_index=0
    )
    assert episode.user_data["test_metric"] == [1]

    callback.on_episode_end(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )
    assert episode.custom_metrics["test_metric"] == 1
