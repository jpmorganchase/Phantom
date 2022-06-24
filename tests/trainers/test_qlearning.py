import tempfile

import phantom as ph
import pytest

from .. import MockEnv, MockPolicy


def test_qlearning_trainer_single_policy():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ph.trainers.QLearningTrainer(tensorboard_log_dir=tmp_dir)

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "q_policy": ["a1"],
                "fixed_policy": (MockPolicy, ["a2", "a3"]),
            },
            policies_to_train=["q_policy"],
        )


def test_qlearning_trainer_shared_single_policy():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ph.trainers.QLearningTrainer(tensorboard_log_dir=tmp_dir)

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "q_policy": ["a1", "a2"],
                "fixed_policy": (MockPolicy, ["a3"]),
            },
            policies_to_train=["q_policy"],
        )


def test_qlearning_trainer_multiple_policies():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ph.trainers.QLearningTrainer(tensorboard_log_dir=tmp_dir)

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "q_policy_1": ["a1"],
                "q_policy_2": ["a2"],
                "fixed_policy": (MockPolicy, ["a3"]),
            },
            policies_to_train=["q_policy_1", "q_policy_2"],
        )


def test_qlearning_trainer_missing_policy():
    with pytest.raises(ValueError) as e:
        trainer = ph.trainers.QLearningTrainer()

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "q_policy": ["a1"],
                "fixed_policy": (MockPolicy, ["a2"]),
            },
            policies_to_train=["q_policy"],
        )

    assert str(e.value) == "Agent 'a3' takes actions but is not assigned a policy."
