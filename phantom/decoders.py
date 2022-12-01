from abc import abstractmethod, ABC
from itertools import chain
from typing import Any, Dict, Generic, Iterable, List, Mapping, Tuple, TypeVar

import gym
import numpy as np

from .context import Context
from .message import MsgPayload
from .types import AgentID
from .utils import flatten


Action = TypeVar("Action")


class Decoder(Generic[Action], ABC):
    """A trait for types that decode raw actions into packets."""

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        """The action/input space of the decoder type."""

    @abstractmethod
    def decode(self, ctx: Context, action: Action) -> List[Tuple[AgentID, MsgPayload]]:
        """Convert an action into a packet given a network context.

        Arguments:
            ctx: The local network context.
            action: An action instance which is an element of the decoder's
                action space.
        """

    def chain(self, others: Iterable["Decoder"]) -> "ChainedDecoder":
        """Chains this decoder together with adjoint set of decoders.

        This method returns a :class:`ChainedDecoder` instance where the action
        space reduces to a tuple with each element given by the action space
        specified in each of the decoders provided.
        """
        return ChainedDecoder(flatten([self, others]))

    def reset(self):
        """Resets the decoder."""

    def __repr__(self) -> str:
        return repr(self.action_space)

    def __str__(self) -> str:
        return str(self.action_space)


class EmptyDecoder(Decoder[Any]):
    """Converts any actions into empty packets."""

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, (0,))

    def decode(self, _: Context, action: Action) -> List[Tuple[AgentID, MsgPayload]]:
        return []


class ChainedDecoder(Decoder[Tuple]):
    """Combines n decoders into a single decoder with a tuple action space.

    Attributes:
        decoders: An iterable collection of decoders which is flattened into a
            list.
    """

    def __init__(self, decoders: Iterable[Decoder]):
        self.decoders: List[Decoder] = flatten(decoders)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Tuple(tuple(d.action_space for d in self.decoders))

    def decode(self, ctx: Context, action: Tuple) -> List[Tuple[AgentID, MsgPayload]]:
        return list(
            chain.from_iterable(
                decoder.decode(ctx, sub_action)
                for decoder, sub_action in zip(self.decoders, action)
            )
        )

    def chain(self, others: Iterable["Decoder"]) -> "ChainedDecoder":
        return ChainedDecoder(self.decoders + list(others))

    def reset(self):
        for decoder in self.decoders:
            decoder.reset()


class DictDecoder(Decoder[Dict[str, Any]]):
    """Combines n decoders into a single decoder with a dict action space.

    Attributes:
        decoders: A mapping of decoder names to decoders.
    """

    def __init__(self, decoders: Mapping[str, Decoder]):
        self.decoders: Dict[str, Decoder] = dict(decoders)

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {name: decoder.action_space for name, decoder in self.decoders.items()}
        )

    def decode(
        self, ctx: Context, action: Dict[str, Any]
    ) -> List[Tuple[AgentID, MsgPayload]]:
        return list(
            chain.from_iterable(
                decoder.decode(ctx, action[name])
                for name, decoder in self.decoders.items()
            )
        )

    def reset(self):
        for decoder in self.decoders.values():
            decoder.reset()
