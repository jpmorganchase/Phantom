from abc import abstractmethod, abstractproperty, ABC
from itertools import chain
from typing import Any, Dict, Generic, Iterable, List, Mapping, Tuple, TypeVar

import numpy as np
from gym.spaces import Box, Space, Dict as GymDict, Tuple as GymTuple
from mercury import Network

from .packet import Packet


Action = TypeVar("Action")


def flatten(xs: Iterable[Any]) -> List[Any]:
    """Recursively flatten an iterable object into a list.

    Arguments:
        xs: The iterable object.
    """
    return sum(([x] if not isinstance(x, Iterable) else flatten(x) for x in xs), [])


class Decoder(Generic[Action], ABC):
    """A trait for types that decode raw actions into packets."""

    @abstractproperty
    def action_space(self) -> Space:
        """The action/input space of the decoder type."""
        pass

    @abstractmethod
    def decode(self, ctx: Network.Context, action: Action) -> Packet:
        """Convert an action into a packet given a network context.

        Arguments:
            ctx: The local network context.
            action: An action instance which is an element of the decoder's
                action space.
        """
        pass

    def chain(self, others: Iterable["Decoder"]) -> "ChainedDecoder":
        """Chains this decoder together with adjoint set of decoders.

        This method returns a :class:`ChainedDecoder` instance where the action
        space reduces to a tuple with each element given by the action space
        specified in each of the decoders provided.
        """
        return ChainedDecoder(flatten([self, others]))

    def reset(self):
        """Resets the decoder."""
        pass

    def __repr__(self) -> str:
        return repr(self.action_space)

    def __str__(self) -> str:
        return str(self.action_space)


class EmptyDecoder(Decoder[Any]):
    """Converts empty actions into empty packets."""

    @property
    def action_space(self) -> Space:
        return Box(-np.inf, np.inf, (0,))

    def decode(self, ctx: Network.Context, action: Action) -> Packet:
        return Packet()


class ChainedDecoder(Decoder[Tuple]):
    """Combines n decoders into a single decoder with a tuple action space.

    Attributes:
        decoders: An iterable collection of decoders which is flattened into a
            list.
    """

    def __init__(self, decoders: Iterable[Decoder]):
        self.decoders: List[Decoder] = flatten(decoders)

    @property
    def action_space(self) -> Space:
        return GymTuple(tuple(d.action_space for d in self.decoders))

    def decode(self, ctx: Network.Context, action: Tuple) -> Packet:
        packet = Packet()

        for i, decoder in enumerate(self.decoders):
            new_packet = decoder.decode(ctx, action[i])

            packet.mutations = chain(packet.mutations, new_packet.mutations)

            for aid, ms in new_packet.messages.items():
                if aid in packet.messages:
                    packet.messages[aid] = chain(packet.messages[aid], ms)

                else:
                    packet.messages[aid] = ms

        return packet

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
    def action_space(self) -> Space:
        return GymDict(
            {name: decoder.action_space for name, decoder in self.decoders.items()}
        )

    def decode(self, ctx: Network.Context, action: Dict[str, Any]) -> Packet:
        packet = Packet()

        for name, decoder in self.decoders.items():
            new_packet = decoder.decode(ctx, action[name])

            packet.mutations = chain(packet.mutations, new_packet.mutations)

            for aid, ms in new_packet.messages.items():
                if aid in packet.messages:
                    packet.messages[aid] = chain(packet.messages[aid], ms)

                else:
                    packet.messages[aid] = ms

        return packet

    def reset(self):
        for decoder in self.decoders.values():
            decoder.reset()
