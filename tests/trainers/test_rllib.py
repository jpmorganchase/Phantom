import tempfile

import phantom as ph

from .. import MockAgent, MockEnv


def test_rllib_trainer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ph.rllib.train(
            algorithm="PPO",
            env_class=MockEnv,
            policies={
                "ppo_policy": MockAgent,
            },
            policies_to_train=["ppo_policy"],
            tune_config={
                "local_dir": tmp_dir,
                "stop": {
                    "training_iteration": 10,
                },
            },
        )
