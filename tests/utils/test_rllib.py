from tempfile import TemporaryDirectory

import phantom as ph
import pytest

from .. import MockAgent, MockEnv


def test_rllib_train_rollout():
    with TemporaryDirectory() as tmp_dir:
        ph.utils.rllib.train(
            algorithm="PPO",
            env_class=MockEnv,
            env_config={},
            policies={
                "mock_policy": MockAgent,
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
                "local_dir": tmp_dir,
            },
            num_workers=1,
        )

        import os

        print(list(os.listdir(tmp_dir)))

        # Without workers:
        results = ph.utils.rllib.rollout(
            directory=f"{tmp_dir}/PPO/LATEST",
            algorithm="PPO",
            env_class=MockEnv,
            env_config={},
            num_repeats=1,
            num_workers=0,
        )

        assert len(list(results)) == 1

        results = ph.utils.rllib.rollout(
            directory=f"{tmp_dir}/PPO/LATEST",
            algorithm="PPO",
            env_class=MockEnv,
            env_config={},
            num_repeats=3,
            num_workers=0,
        )

        assert len(list(results)) == 3

        # With workers:
        results = ph.utils.rllib.rollout(
            directory=f"{tmp_dir}/PPO/LATEST",
            algorithm="PPO",
            env_class=MockEnv,
            env_config={},
            num_repeats=1,
            num_workers=1,
        )

        assert len(list(results)) == 1

        results = ph.utils.rllib.rollout(
            directory=f"{tmp_dir}/PPO/LATEST",
            algorithm="PPO",
            env_class=MockEnv,
            env_config={},
            num_repeats=3,
            num_workers=1,
        )

        assert len(list(results)) == 3


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
