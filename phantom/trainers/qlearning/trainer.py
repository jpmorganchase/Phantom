from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from ...env import PhantomEnv
from ...types import AgentID, PolicyID
from ..trainer import Trainer
from .policy import QLearningPolicy


class QLearningTrainer(Trainer):
    """
    Simple QLearning algorithm implementation.

    Arguments:
        tensorboard_log_dir: If provided, will save metrics to the given directory
            in a format that can be viewed with tensorboard.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate.
    """

    policy_class = QLearningPolicy

    def __init__(
        self,
        # Trainer general args:
        tensorboard_log_dir: Optional[str] = None,
        # PPOTrainer specific args:
        # Learning rate:
        alpha: float = 0.1,
        # Discount factor:
        gamma: float = 0.6,
        # Exploration rate:
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(tensorboard_log_dir)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def training_step(
        self,
        env: PhantomEnv,
        policy_mapping: Mapping[AgentID, PolicyID],
        policies: Mapping[PolicyID, QLearningPolicy],
        policies_to_train: Iterable[PolicyID],
    ) -> None:
        assert len(policies_to_train) == 1
        policy_to_train = policies_to_train[0]
        batch_size = 10

        for _ in range(batch_size):
            observations = env.reset()

            while not env.is_done():
                actions: Dict[AgentID, Any] = {}

                for agent_id, obs in observations.items():
                    policy_name = policy_mapping[agent_id]
                    policy = policies[policy_name]

                    if policy_name == policy_to_train:
                        assert isinstance(policy, QLearningPolicy)  # mypy satisfier

                        if np.random.uniform(0, 1) < self.epsilon:
                            # Explore action space
                            action = policy.action_space.sample()
                        else:
                            # Exploit learned values
                            action = np.argmax(policy.q_table[obs])
                    else:
                        action = policy.compute_action(obs)

                    actions[agent_id] = action

                next_observations, rewards, _, _ = env.step(actions)

                for agent_id, obs in observations.items():
                    policy_name = policy_mapping[agent_id]
                    policy = policies[policy_name]

                    if policy_name == policy_to_train:
                        assert isinstance(policy, QLearningPolicy)  # mypy satisfier

                        reward = rewards[agent_id]
                        next_obs = next_observations[agent_id]

                        old_value = policy.q_table[obs, actions[agent_id]]
                        next_max = np.max(policy.q_table[next_obs])

                        new_value = (1 - self.alpha) * old_value + self.alpha * (
                            reward + self.gamma * next_max
                        )
                        policy.q_table[obs, actions[agent_id]] = new_value

                observations = next_observations

                self.log_rewards(rewards)
                self.log_metrics(env)