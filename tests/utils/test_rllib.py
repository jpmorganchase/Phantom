import phantom as ph
import pytest

from .. import MockStrategicAgent, MockEnv, MockPolicy


def test_rllib_train_rollout(tmpdir):
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        policies={
            "mock_policy": MockStrategicAgent,
        },
        rllib_config={
            "disable_env_checking": True,
            "num_rollout_workers": 1,
        },
        iterations=2,
        checkpoint_freq=1,
        results_dir=tmpdir,
    )

    # Without workers, without env class:
    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_config={},
        num_repeats=3,
        num_workers=0,
    )
    assert len(list(results)) == 3

    # With workers, with env class:
    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_class=MockEnv,
        env_config={},
        num_repeats=3,
        num_workers=1,
    )
    results = list(results)
    assert len(results) == 3
    assert results[0].actions_for_agent("a1") == [0, 0, 0, 0, 0]

    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_class=MockEnv,
        env_config={},
        custom_policy_mapping={"a1": MockPolicy},
        num_repeats=1,
        num_workers=1,
    )
    assert list(results)[0].actions_for_agent("a1") == [1, 1, 1, 1, 1]

    # Evaluate policy:
    results = ph.utils.rllib.evaluate_policy(
        directory=f"{tmpdir}/LATEST",
        obs=ph.utils.ranges.LinspaceRange(0, 1, 2, name="r"),
        policy_id="mock_policy",
    )
    assert list(results) == [({"r": 0.0}, 0.0, 0), ({"r": 1.0}, 1.0, 0)]


def test_rllib_rollout_bad(tmpdir):
    # num_repeats < 1
    with pytest.raises(AssertionError):
        list(
            ph.utils.rllib.rollout(
                directory=tmpdir,
                env_class=MockEnv,
                env_config={},
                num_repeats=0,
            )
        )

    # num_repeats < 0
    with pytest.raises(AssertionError):
        list(
            ph.utils.rllib.rollout(
                directory=tmpdir,
                env_class=MockEnv,
                env_config={},
                num_workers=-1,
            )
        )
