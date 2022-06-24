import tempfile

import phantom as ph

from .. import MockAgent, MockEnv


def test_ppo_trainer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ph.trainers.PPOTrainer(tensorboard_log_dir=tmp_dir)

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "ppo_policy": MockAgent,
            },
            policies_to_train=["ppo_policy"],
        )
