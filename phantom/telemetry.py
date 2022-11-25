from typing import Any, Mapping, Optional, Sequence, TypeVar, Union

import numpy as np
from termcolor import colored

from .message import Message
from .types import AgentID, StageID


Action = TypeVar("Action")
Observation = TypeVar("Observation")

TAB_SIZE = 4


class TelemetryLogger:
    def __init__(self) -> None:
        self.configure()

    def configure(
        self,
        enable: bool = False,
        log_actions: Union[bool, Sequence[AgentID]] = False,
        log_observations: Union[bool, Sequence[AgentID]] = False,
        log_rewards: Union[bool, Sequence[AgentID]] = False,
        log_dones: Union[bool, Sequence[AgentID]] = False,
        log_infos: Union[bool, Sequence[AgentID]] = False,
        log_messages: Union[bool, Sequence[AgentID]] = False,
    ) -> None:
        self._enable = (
            enable
            or log_actions
            or log_observations
            or log_rewards
            or log_dones
            or log_infos
            or log_messages
        )
        self._log_actions = log_actions
        self._log_observations = log_observations
        self._log_rewards = log_rewards
        self._log_dones = log_dones
        self._log_infos = log_infos
        self._log_messages = log_messages

    def log_reset(self) -> None:
        if self._enable:
            print(colored("=" * 80, attrs=["dark"]))
            print(colored(f"ENV RESET", attrs=["bold"]))

    def log_step(self, current_step: int, num_steps: int) -> None:
        if self._enable:
            print(colored("-" * 80, attrs=["dark"]))
            print(colored(f"STEP {current_step}/{num_steps}:", attrs=["bold"]))

    def log_start_decoding_actions(self) -> None:
        if self._enable:
            print(_t(1) + colored(f"DECODING ACTIONS:", color="cyan"))

    def log_actions(self, actions: Mapping[AgentID, Action]) -> None:
        if self._enable and self._log_actions:
            print(_t(1) + colored("ACTIONS:", color="cyan"))

            if not isinstance(self._log_actions, bool):
                actions = {
                    a: act for a, act in actions.items() if a in self._log_actions
                }

            if len(actions) > 0:
                for agent, action in actions.items():
                    print(_t(2) + f"{agent}: {_pretty_format_space(action)}")
            else:
                print(_t(2) + "None")

    def log_observations(self, observations: Mapping[AgentID, Observation]) -> None:
        if self._enable and self._log_observations:
            print(_t(1) + colored("OBSERVATIONS:", color="cyan"))

            if not isinstance(self._log_observations, bool):
                observations = {
                    a: obs
                    for a, obs in observations.items()
                    if a in self._log_observations
                }

            if len(observations) > 0:
                for agent, observation in observations.items():
                    print(_t(2) + f"{agent}: {_pretty_format_space(observation)}")
            else:
                print(_t(2) + "None")

    def log_rewards(self, rewards: Mapping[AgentID, float]) -> None:
        if self._enable and self._log_rewards:
            print(_t(1) + colored("REWARDS:", color="cyan"))

            if not isinstance(self._log_rewards, bool):
                rewards = {
                    a: rew for a, rew in rewards.items() if a in self._log_rewards
                }

            if len(rewards) > 0:
                for agent, reward in rewards.items():
                    print(_t(2) + f"{agent}: {reward:.2f}")
            else:
                print(_t(2) + "None")

    def log_dones(self, dones: Mapping[AgentID, bool]) -> None:
        if self._enable and self._log_dones:
            print(_t(1) + colored("DONES:", color="cyan"))

            dones = [a for a, done in dones.items() if done]

            if not isinstance(self._log_dones, bool):
                dones = [a for a in dones if a in self._log_dones]

            if len(dones) > 0:
                print(_t(2) + ", ".join(dones))
            else:
                print(_t(2) + "None")

    def log_infos(self, infos: Mapping[AgentID, Any]) -> None:
        if self._enable and self._log_infos:
            print(_t(1) + colored("INFOS:", color="cyan"))

            if not isinstance(self._log_infos, bool):
                infos = {a: info for a, info in infos.items() if a in self._log_infos}

            infos = {a: info for a, info in infos.items() if info is not None and info != {}}

            if len(infos) > 0:
                for agent, info in infos.items():
                    print(_t(2) + f"{agent}: {info}")
            else:
                print(_t(2) + "None")

    def log_step_values(
        self,
        observations: Mapping[AgentID, Observation],
        rewards: Mapping[AgentID, float],
        dones: Mapping[AgentID, bool],
        infos: Mapping[AgentID, Any],
    ) -> None:
        self.log_observations(observations)
        self.log_rewards(rewards)
        self.log_dones(dones)
        self.log_infos(infos)

    def log_fsm_transition(self, current_stage: StageID, next_stage: StageID) -> None:
        if self._enable:
            print(
                _t(1)
                + colored(
                    f"FSM TRANSITION: {current_stage} --> {next_stage}",
                    "magenta",
                )
            )

    def log_start_resolving_msgs(self) -> None:
        if self._enable:
            print(_t(1) + colored("RESOLVING MESSAGES:", color="cyan"))

    def log_resolver_round(self, round: int, max: Optional[int]) -> None:
        if self._enable and self._log_messages:
            print(
                _t(1)
                + colored(
                    f"Batch Resolver round {round+1}/{max or 'Inf'}:",
                    color="grey",
                )
            )

    def log_msg_send(self, message: Message) -> None:
        if self._should_log_msg(message):
            route_str = f"{message.sender_id: >10} --> {message.receiver_id: <10}"
            msg_name = f"({message.payload.__class__.__name__})"

            print(_t(2) + f"MSG SEND: {route_str} {msg_name: <20}")

    def log_msg_recv(self, message: Message) -> None:
        if self._should_log_msg(message):
            route_str = f"{message.sender_id: >10} --> {message.receiver_id: <10}"
            msg_name = f"({message.payload.__class__.__name__})"
            fields = ", ".join(f"{k}: {v}" for k, v in message.payload.__dict__.items())

            print(
                _t(2)
                + f"MSG RECV: {route_str} {msg_name: <20}"
                + colored(fields, attrs=["dark"])
            )

    def log_episode_done(self) -> None:
        if self._enable:
            print(_t(1) + colored("EPISODE DONE", color="green", attrs=["bold"]))

    def _should_log_msg(self, message: Message) -> bool:
        return (
            self._enable
            and self._log_messages
            and (
                isinstance(self._log_messages, bool)
                or message.sender_id in self._log_messages
                or message.receiver_id in self._log_messages
            )
        )


def _t(n: int) -> str:
    return " " * n * TAB_SIZE


def _pretty_format_space(space) -> str:
    if isinstance(space, tuple):
        return "(" + ", ".join(_pretty_format_space(x) for x in space) + ")"
    elif isinstance(space, dict):
        return (
            "{"
            + ", ".join(k + ": " + _pretty_format_space(v) for k, v in space.items())
            + "}"
        )
    elif isinstance(space, np.ndarray):
        return str(space[0]) if space.shape == (1,) else str(space)
    elif isinstance(space, (int, float)):
        return str(space)

    raise NotImplementedError(type(space))


logger = TelemetryLogger()
