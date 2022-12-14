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
        metrics={"step": ph.metrics.SimpleEnvMetric("current_step")},
    )

    env = MockEnv()

    env.reset()

    for _ in range(5):
        env.step({})

    assert ph.telemetry.logger._current_episode is None
    assert not os.path.isfile(tmpdir.join("log.json"))

    ph.telemetry.logger.configure_print_logging(enable=False)

    ph.telemetry.logger.configure_file_logging(
        file_path=tmpdir.join("log.json"),
        metrics={"step": ph.metrics.SimpleEnvMetric("current_step")},
    )

    env = MockEnv()

    env.reset()

    for _ in range(5):
        env.step({})

    assert os.path.isfile(tmpdir.join("log.json"))

    data = json.load(open(tmpdir.join("log.json"), "r"))

    assert set(data.keys()) == {"start", "steps"}
    assert len(data["steps"]) == 6
    assert set(data["steps"][0]) == {"messages", "metrics", "observations"}
    assert set(data["steps"][1]) == {
        "actions",
        "dones",
        "infos",
        "messages",
        "metrics",
        "observations",
        "rewards",
    }

    ph.telemetry.logger.configure_file_logging(file_path=None)
