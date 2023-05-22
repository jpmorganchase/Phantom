from phantom import FiniteStateMachineEnv, FSMStage, Network
from phantom.metrics import NotRecorded
from phantom.utils.rllib.train import RLlibMetricLogger

from . import MockBaseEnv, MockEpisode, MockMetric


def test_fsm_logging():
    env = FiniteStateMachineEnv(
        num_steps=2,
        network=Network(),
        initial_stage=0,
        stages=[FSMStage(0, [], None, [1]), FSMStage(1, [], None, [0])],
    )

    episode = MockEpisode()
    base_env = MockBaseEnv(env)

    callback = RLlibMetricLogger(
        {
            "stage_0_metric": MockMetric(0, "sum", fsm_stages=[0]),
            "stage_1_metric": MockMetric(1, "sum", fsm_stages=[1]),
        }
    )()

    callback.on_episode_start(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )

    callback.on_episode_step(
        worker=None, base_env=base_env, episode=episode, env_index=0
    )
    assert episode.user_data == {
        "stage_0_metric": [0],
        "stage_1_metric": [NotRecorded()],
    }

    base_env.step()

    callback.on_episode_step(
        worker=None, base_env=base_env, episode=episode, env_index=0
    )
    assert episode.user_data == {
        "stage_0_metric": [0, NotRecorded()],
        "stage_1_metric": [NotRecorded(), 1],
    }

    callback.on_episode_end(
        worker=None, base_env=base_env, policies=None, episode=episode, env_index=0
    )
    assert episode.custom_metrics == {
        "stage_0_metric": 0,
        "stage_1_metric": 1,
    }
