import phantom as ph

from . import MockEnv


def test_telemetry():
    ph.telemetry.logger.configure(
        log_actions=True,
        log_observations=True,
        log_rewards=True,
        log_dones=True,
        log_infos=True,
        log_messages=True,
    )

    env = MockEnv()

    env.reset()
    env.step({})
