import io
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
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
        self._enable_print = False
        self._print_actions: Union[bool, Sequence[AgentID]] = False
        self._print_observations: Union[bool, Sequence[AgentID]] = False
        self._print_rewards: Union[bool, Sequence[AgentID]] = False
        self._print_dones: Union[bool, Sequence[AgentID]] = False
        self._print_infos: Union[bool, Sequence[AgentID]] = False
        self._print_messages: Union[bool, Sequence[AgentID]] = False

        self._output_file: Optional[io.TextIOBase] = None
        self._human_readable: bool = False

        self._current_episode = None

    def configure_print_logging(
        self,
        enable: Union[bool, None] = None,
        print_actions: Union[bool, Sequence[AgentID], None] = None,
        print_observations: Union[bool, Sequence[AgentID], None] = None,
        print_rewards: Union[bool, Sequence[AgentID], None] = None,
        print_dones: Union[bool, Sequence[AgentID], None] = None,
        print_infos: Union[bool, Sequence[AgentID], None] = None,
        print_messages: Union[bool, Sequence[AgentID], None] = None,
    ) -> None:
        if enable is not None:
            self._enable_print = enable

        if print_actions is not None:
            self._print_actions = print_actions

        if print_observations is not None:
            self._print_observations = print_observations

        if print_rewards is not None:
            self._print_rewards = print_rewards

        if print_dones is not None:
            self._print_dones = print_dones

        if print_infos is not None:
            self._print_infos = print_infos

        if print_messages is not None:
            self._print_messages = print_messages

    def configure_file_logging(
        self,
        file_path: Union[str, Path, None],
        append: bool = True,
        human_readable: Optional[bool] = None,
    ) -> None:
        if human_readable is not None:
            self._human_readable = human_readable

        if file_path is None:
            self._output_file = None
        else:
            self._output_file = open(file_path, "a" if append else "w")

    def log_reset(self) -> None:
        self._current_episode = {
            "start": str(datetime.now()),
            "steps": [{"messages": []}],
        }

        if self._enable_print:
            print(colored("=" * 80, attrs=["dark"]))
            print(colored(f"ENV RESET", attrs=["bold"]))

    def log_step(self, current_step: int, num_steps: int) -> None:
        if self._current_episode is not None:
        self._current_episode["steps"].append({"messages": []})

        if self._enable_print:
            print(colored("-" * 80, attrs=["dark"]))
            print(colored(f"STEP {current_step}/{num_steps}:", attrs=["bold"]))

    def log_start_decoding_actions(self) -> None:
        if self._enable_print:
            print(_t(1) + colored(f"DECODING ACTIONS:", color="cyan"))

    def log_actions(self, actions: Mapping[AgentID, Action]) -> None:
        if self._current_episode is not None:
        self._current_episode["steps"][-1]["actions"] = actions

        if self._enable_print and self._print_actions:
            print(_t(1) + colored("ACTIONS:", color="cyan"))

            if not isinstance(self._print_actions, bool):
                actions = {
                    a: act for a, act in actions.items() if a in self._print_actions
                }

            if len(actions) > 0:
                for agent, action in actions.items():
                    print(_t(2) + f"{agent}: {_pretty_format_space(action)}")
            else:
                print(_t(2) + "None")

    def log_observations(self, observations: Mapping[AgentID, Observation]) -> None:
        if self._current_episode is not None:
        self._current_episode["steps"][-1]["observations"] = observations

        if self._enable_print and self._print_observations:
            print(_t(1) + colored("OBSERVATIONS:", color="cyan"))

            if not isinstance(self._print_observations, bool):
                observations = {
                    a: obs
                    for a, obs in observations.items()
                    if a in self._print_observations
                }

            if len(observations) > 0:
                for agent, observation in observations.items():
                    print(_t(2) + f"{agent}: {_pretty_format_space(observation)}")
            else:
                print(_t(2) + "None")

    def log_rewards(self, rewards: Mapping[AgentID, float]) -> None:
        if self._current_episode is not None:
        self._current_episode["steps"][-1]["rewards"] = rewards

        if self._enable_print and self._print_rewards:
            print(_t(1) + colored("REWARDS:", color="cyan"))

            if not isinstance(self._print_rewards, bool):
                rewards = {
                    a: rew for a, rew in rewards.items() if a in self._print_rewards
                }

            if len(rewards) > 0:
                for agent, reward in rewards.items():
                    print(_t(2) + f"{agent}: {reward:.2f}")
            else:
                print(_t(2) + "None")

    def log_dones(self, dones: Mapping[AgentID, bool]) -> None:
        dones = [a for a, done in dones.items() if done]

        if self._current_episode is not None:
        self._current_episode["steps"][-1]["dones"] = dones

        if self._enable_print and self._print_dones:
            print(_t(1) + colored("DONES:", color="cyan"))

            if not isinstance(self._print_dones, bool):
                dones = [a for a in dones if a in self._print_dones]

            if len(dones) > 0:
                print(_t(2) + ", ".join(dones))
            else:
                print(_t(2) + "None")

    def log_infos(self, infos: Mapping[AgentID, Any]) -> None:
        if self._current_episode is not None:
        self._current_episode["steps"][-1]["infos"] = infos

        if self._enable_print and self._print_infos:
            print(_t(1) + colored("INFOS:", color="cyan"))

            if not isinstance(self._print_infos, bool):
                infos = {a: info for a, info in infos.items() if a in self._print_infos}

            infos = {
                a: info for a, info in infos.items() if info is not None and info != {}
            }

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
        if self._current_episode is not None:
        self._current_episode["steps"][-1]["fsm_current_stage"] = current_stage
        self._current_episode["steps"][-1]["fsm_next_stage"] = next_stage

        if self._enable_print:
            print(
                _t(1)
                + colored(
                    f"FSM TRANSITION: {current_stage} --> {next_stage}",
                    "magenta",
                )
            )

    def log_start_resolving_msgs(self) -> None:
        if self._enable_print:
            print(_t(1) + colored("RESOLVING MESSAGES:", color="cyan"))

    def log_resolver_round(self, round: int, max: Optional[int]) -> None:
        if self._enable_print and self._print_messages:
            print(
                _t(1)
                + colored(
                    f"Batch Resolver round {round+1}/{max or 'Inf'}:",
                    color="grey",
                )
            )

    def log_msg_send(self, message: Message) -> None:
        if self._should_print_msg(message):
            route_str = f"{message.sender_id: >10} --> {message.receiver_id: <10}"
            msg_name = f"({message.payload.__class__.__name__})"

            print(_t(2) + f"MSG SEND: {route_str} {msg_name: <20}")

    def log_msg_recv(self, message: Message) -> None:
        if self._current_episode is not None:
        self._current_episode["steps"][-1]["messages"].append(asdict(message))

        if self._should_print_msg(message):
            route_str = f"{message.sender_id: >10} --> {message.receiver_id: <10}"
            msg_name = f"({message.payload.__class__.__name__})"
            fields = ", ".join(f"{k}: {v}" for k, v in message.payload.__dict__.items())

            print(
                _t(2)
                + f"MSG RECV: {route_str} {msg_name: <20}"
                + colored(fields, attrs=["dark"])
            )

    def log_episode_done(self) -> None:
        self._write_episode_to_file()

        if self._enable_print:
            print(_t(1) + colored("EPISODE DONE", color="green", attrs=["bold"]))

    def _write_episode_to_file(self) -> None:
        if self._output_file is not None and self._current_episode is not None:
            json.dump(
                self._current_episode,
                self._output_file,
                indent=2 if self._human_readable else None,
                cls=NumpyArrayEncoder,
            )
            self._output_file.write("\n")
            self._output_file.flush()
            self._current_episode = None

    def _should_print_msg(self, message: Message) -> bool:
        return (
            self._enable_print
            and self._print_messages
            and (
                isinstance(self._print_messages, bool)
                or message.sender_id in self._print_messages
                or message.receiver_id in self._print_messages
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


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


logger = TelemetryLogger()
