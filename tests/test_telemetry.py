import os
import json
import phantom as ph

from . import MockEnv


def test_telemetry(tmpdir):
    ph.telemetry.logger.configure_print_logging(
        print_actions=True,
        print_observations=True,
        print_rewards=True,
        print_dones=True,
        print_infos=True,
        print_messages=True,
    )

    env = MockEnv()

    env.reset()

    for _ in range(5):
        env.step({})

    assert not os.path.isfile(tmpdir.join("log.json"))

    ph.telemetry.logger.configure_print_logging(enable=False)

    ph.telemetry.logger.configure_file_logging(
        file_path=tmpdir.join("log.json"),
    )

    env = MockEnv()

    env.reset()

    for _ in range(5):
        env.step({})

    assert os.path.isfile(tmpdir.join("log.json"))

    json.load(open(tmpdir.join("log.json"), "r"))

    ph.telemetry.logger.configure_file_logging(file_path=None)
