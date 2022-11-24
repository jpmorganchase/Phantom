import phantom as ph
import pytest

from .. import MockRLAgent, MockEnv


def test_rllib_train_rollout(tmpdir):
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        policies={
            "mock_policy": MockRLAgent,
        },
        policies_to_train=["mock_policy"],
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
        num_repeats=1,
        num_workers=0,
    )
    assert len(list(results)) == 1

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
        num_repeats=1,
        num_workers=1,
    )
    assert len(list(results)) == 1

    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/PPO/LATEST",
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        num_repeats=3,
        num_workers=1,
    )
    assert len(list(results)) == 3

    results = ph.utils.rllib.evaluate_policy(
        directory=f"{tmpdir}/PPO/LATEST",
        algorithm="PPO",
        env_class=MockEnv,
        obs=0,
        policy_id="mock_policy",
    )
    assert len(list(results)) == 1


def test_rllib_train_bad(tmpdir):
    # policy to train not defined
    with pytest.raises(ValueError):
        ph.utils.rllib.train(
            algorithm="PPO",
            env_class=MockEnv,
            env_config={},
            policies={
                "mock_policy": MockRLAgent,
            },
            policies_to_train=["undefined_policy"],
            rllib_config={
                "disable_env_checking": True,
            },
            tune_config={
                "local_dir": tmpdir,
            },
        )


def test_rllib_rollout_bad():
    # num_repeats < 1
    with pytest.raises(AssertionError):
        list(
            ph.utils.rllib.rollout(
                directory=f"PPO/LATEST",
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
                directory=f"PPO/LATEST",
                algorithm="PPO",
                env_class=MockEnv,
                env_config={},
                num_workers=-1,
            )
        )
