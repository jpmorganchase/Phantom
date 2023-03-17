from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import rich.progress
import torch

from ...env import PhantomEnv
from ...logging import Metric
from ...types import AgentID, PolicyID
from ...utils import check_env_config
from ..trainer import PolicyMapping, Trainer, TrainingResults

from .policy import PPOPolicy
from .storage import RolloutStorage
from .utils import update_linear_schedule


class PPOTrainer(Trainer):
    """
    Proximal Policy Optimisation (PPO) algorithm implementation derived from
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.

    For performance and stability reasons, it is recommended that the RLlib
    implementation is used using the :func:`utils.rllib.train` function.

    Arguments:
        tensorboard_log_dir: If provided, will save metrics to the given directory
            in a format that can be viewed with tensorboard.
        ppo_epoch:
        num_mini_batch:
        clip_param:
        use_clipped_value_loss:
        use_linear_lr_decay:
        lr:
        eps:
        value_loss_coef:
        entropy_coef:
        max_grad_norm:
        use_gae:
        gamma:
        gae_lambda:
        use_proper_time_limits:
    """

    policy_class = PPOPolicy

    def __init__(
        self,
        # Trainer general args:
        tensorboard_log_dir: Optional[str] = None,
        # PPOTrainer specific args:
        ppo_epoch: int = 4,
        num_mini_batch: int = 32,
        clip_param: float = 0.2,
        use_clipped_value_loss: bool = True,
        use_linear_lr_decay: bool = False,
        lr: float = 7e-4,
        eps: float = 1e-5,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_gae: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_proper_time_limits: bool = False,
    ) -> None:
        super().__init__(tensorboard_log_dir)

        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_linear_lr_decay = use_linear_lr_decay
        self.lr = lr
        self.eps = eps
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

    def train(
        self,
        env_class: Type[PhantomEnv],
        num_iterations: int,
        policies: PolicyMapping,
        policies_to_train: Sequence[PolicyID],
        env_config: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
    ) -> TrainingResults:
        env_config = env_config or {}
        self.metrics = metrics or {}

        check_env_config(env_config)

        num_envs = 10

        envs = []
        observations = []

        for _ in range(num_envs):
            env = env_class(**env_config)
            observations.append(env.reset())
            envs.append(env)

        policy_mapping, policy_instances = self.setup_policy_specs_and_mapping(
            env, policies
        )

        assert len(policies_to_train) == 1
        policy_to_train = policies_to_train[0]

        training_policy = policy_instances[policy_to_train]
        training_agent = next(
            a for a, p in policy_mapping.items() if p == policy_to_train
        )

        assert isinstance(training_policy, self.policy_class)

        device = torch.device("cpu")

        self.actor_critic = PPOPolicy(
            training_policy.observation_space,
            training_policy.action_space,
            base_kwargs={"recurrent": False},
        )
        self.actor_critic.to(device)

        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=self.lr, eps=self.eps
        )

        rollouts = RolloutStorage(
            envs[0].num_steps,
            num_envs,
            training_policy.observation_space,
            training_policy.action_space,
            self.actor_critic.recurrent_hidden_state_size,
        )

        agent_obs = np.array([obs[training_agent] for obs in observations])

        rollouts.obs[0].copy_(torch.FloatTensor(agent_obs))
        rollouts.to(device)

        # episode_rewards = deque(maxlen=10)

        for i in rich.progress.track(range(num_iterations), description="Training..."):
            if self.use_linear_lr_decay:
                # decrease learning rate linearly
                update_linear_schedule(self.optimizer, i, num_iterations, self.lr)

            episode_rewards = defaultdict(list)

            for step in range(env.num_steps):
                # Sample actions
                with torch.no_grad():
                    (
                        value,
                        trained_policy_actions,
                        action_log_prob,
                        recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        rollouts.obs[step].reshape((-1, 1)),
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                    )

                new_observations: List[Dict[AgentID, Any]] = []
                rewards: List[Dict[AgentID, float]] = []
                terminations: List[Dict[AgentID, bool]] = []
                truncations: List[Dict[AgentID, bool]] = []
                infos: List[Dict[AgentID, Any]] = []

                for env, obs, tpa in zip(envs, observations, trained_policy_actions):
                    actions: Dict[AgentID, Any] = {}

                    for agent_id, agent_obs in obs.items():
                        policy_name = policy_mapping[agent_id]
                        policy = policy_instances[policy_name]

                        if policy_name == policy_to_train:
                            if len(tpa) == 1:
                                actions[agent_id] = tpa[0]
                            else:
                                actions[agent_id] = np.array(tpa)

                        else:
                            actions[agent_id] = policy.compute_action(agent_obs)

                    o, r, te, tr, i_ = env.step(actions)

                    new_observations.append(o)
                    rewards.append(r)
                    terminations.append(te)
                    truncations.append(tr)
                    infos.append(i_)

                observations = new_observations

                for agent_id in rewards[0].keys():
                    episode_rewards[agent_id].append(
                        np.mean([r[agent_id] for r in rewards])
                    )

                # for info in infos:
                #     if 'episode' in info.keys():
                #         episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [
                        [0.0] if te[training_agent] or tr[training_agent] else [1.0]
                        for te, tr in zip(terminations, truncations)
                    ]
                )
                bad_masks = torch.FloatTensor(
                    [
                        [0.0]
                        if "bad_transition" in info[training_agent].keys()
                        else [1.0]
                        for info in infos
                    ]
                )

                training_observations = torch.FloatTensor(
                    [obs[training_agent] for obs in observations]
                )

                training_rewards = torch.FloatTensor(
                    [[rwd[training_agent]] for rwd in rewards]
                )

                rollouts.insert(
                    training_observations,
                    recurrent_hidden_states,
                    trained_policy_actions,
                    action_log_prob,
                    value,
                    training_rewards,
                    masks,
                    bad_masks,
                )

                self.log_vec_rewards(rewards)
                self.log_vec_metrics(envs)

            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    rollouts.obs[-1].reshape((-1, 1)),
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(
                next_value,
                self.use_gae,
                self.gamma,
                self.gae_lambda,
                self.use_proper_time_limits,
            )

            # value_loss, action_loss, dist_entropy = self.update(rollouts)
            self.update(rollouts)

            rollouts.after_update()

            self.tbx_write_values(i)

            # save for every interval-th episode or for the last epoch
            # if (j % args.save_interval == 0
            #         or j == num_updates - 1) and args.save_dir != "":
            #     save_path = os.path.join(args.save_dir, args.algo)
            #     try:
            #         os.makedirs(save_path)
            #     except OSError:
            #         pass

            #     torch.save([
            #         actor_critic,
            #         getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            #     ], os.path.join(save_path, args.env_name + ".pt"))

            # if j % args.log_interval == 0 and len(episode_rewards) > 1:
            #     total_num_steps = (j + 1) * args.num_processes * args.num_steps
            #     end = time.time()
            #     print(
            #         "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            #         .format(j, total_num_steps,
            #                 int(total_num_steps / (end - start)),
            #                 len(episode_rewards), np.mean(episode_rewards),
            #                 np.median(episode_rewards), np.min(episode_rewards),
            #                 np.max(episode_rewards), dist_entropy, value_loss,
            #                 action_loss))

            # if (args.eval_interval is not None and len(episode_rewards) > 1
            #         and j % args.eval_interval == 0):
            #     obs_rms = utils.get_vec_normalize(envs).obs_rms
            #     evaluate(actor_critic, obs_rms, args.env_name, args.seed,
            #             args.num_processes, eval_log_dir, device)

        return TrainingResults(policy_instances)

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        for _ in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch
                )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch.reshape((-1, 1)),
                    recurrent_hidden_states_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_param,
                        1.0 + self.clip_param,
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                ).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
