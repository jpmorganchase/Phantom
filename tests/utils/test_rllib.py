import phantom as ph
import pytest

from .. import MockAgent, MockEnv, MockPolicy


def test_rllib_train_rollout(tmpdir):
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        policies={
            "mock_policy": MockAgent,
        },
        rllib_config={
            "disable_env_checking": True,
        },
        tune_config={
            "checkpoint_freq": 1,
            "stop": {
                "training_iteration": 2,
            },
            "local_dir": tmpdir,
        },
        num_workers=1,
    )

    # Without workers:
    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/PPO/LATEST",
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        num_repeats=3,
        num_workers=0,
    )
    assert len(list(results)) == 3

    # With workers:
    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/PPO/LATEST",
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        num_repeats=3,
        num_workers=1,
    )
    results = list(results)
    assert len(results) == 3
    assert results[0].actions_for_agent("a1") == [0, 0, 0, 0, 0]

    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/PPO/LATEST",
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        custom_policy_mapping={"a1": MockPolicy},
        num_repeats=1,
        num_workers=1,
    )
    assert list(results)[0].actions_for_agent("a1") == [1, 1, 1, 1, 1]

    # Evaluate policy:
    results = ph.utils.rllib.evaluate_policy(
        directory=f"{tmpdir}/PPO/LATEST",
        algorithm="PPO",
        env_class=MockEnv,
        obs=0,
        policy_id="mock_policy",
    )
    assert len(list(results)) == 1


def test_rllib_rollout_bad(tmpdir):
    # num_repeats < 1
    with pytest.raises(AssertionError):
        list(
            ph.utils.rllib.rollout(
                directory=tmpdir,
                algorithm="PPO",
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
                algorithm="PPO",
                env_class=MockEnv,
                env_config={},
                num_workers=-1,
            )
        )
