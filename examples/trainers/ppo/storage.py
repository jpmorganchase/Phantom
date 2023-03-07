from typing import Iterator, Optional, Tuple

import gymnasium as gym
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T: int, N: int, tensor: torch.Tensor):
    return tensor.view(T * N, *tensor.size()[2:])


class RolloutStorage:
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_space: gym.Space,
        action_space: gym.Space,
        recurrent_hidden_state_size: int,
    ) -> None:
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_space.shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size
        )
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(
        self,
        obs: torch.Tensor,
        recurrent_hidden_states: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        bad_masks: torch.Tensor,
    ) -> None:
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self) -> None:
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
        self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True
    ) -> None:
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae_tensor = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae_tensor + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                    ) * self.bad_masks[step + 1] + (
                        1 - self.bad_masks[step + 1]
                    ) * self.value_preds[
                        step
                    ]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator(
        self,
        advantages,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Iterator[Tuple]:
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None and num_mini_batch is None:
            raise ValueError

        if mini_batch_size is None:
            assert num_mini_batch is not None
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({num_processes}) * number of "
                f"steps ({num_steps}) = {num_processes * num_steps} to be greater than "
                f"or equal to the number of PPO mini batches ({num_mini_batch})."
            )
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1)
            )[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(
        self, advantages, num_mini_batch: int
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            f"PPO requires the number of processes ({num_processes}) to be greater than"
            f" or equal to the number of PPO mini batches ({num_mini_batch})."
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch_stack = []
            recurrent_hidden_states_batch_stack = []
            actions_batch_stack = []
            value_preds_batch_stack = []
            return_batch_stack = []
            masks_batch_stack = []
            old_action_log_probs_batch_stack = []
            adv_targ_stack = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch_stack.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch_stack.append(
                    self.recurrent_hidden_states[0:1, ind]
                )
                actions_batch_stack.append(self.actions[:, ind])
                value_preds_batch_stack.append(self.value_preds[:-1, ind])
                return_batch_stack.append(self.returns[:-1, ind])
                masks_batch_stack.append(self.masks[:-1, ind])
                old_action_log_probs_batch_stack.append(self.action_log_probs[:, ind])
                adv_targ_stack.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch_stack, 1)
            actions_batch = torch.stack(actions_batch_stack, 1)
            value_preds_batch = torch.stack(value_preds_batch_stack, 1)
            return_batch = torch.stack(return_batch_stack, 1)
            masks_batch = torch.stack(masks_batch_stack, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch_stack, 1
            )
            adv_targ = torch.stack(adv_targ_stack, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch_stack, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
