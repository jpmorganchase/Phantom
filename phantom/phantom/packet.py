from dataclasses import dataclass, field
from typing import Iterable, Mapping

from mercury import Payload, ID


class Mutation:
    """Immutable mutation structure.

    Unlike messages in the Mercury framework, mutations do not correspond to
    any meaningful flow between actors. Instead, they convey instructions for
    direct updates to an actor's internal state. These will mostly be useful
    when engineering modifications to an actor's state in a programmatic way.

    The type variable, A, denotes the class of actor that this mutator applies
    to. More often than not this can just be set to :class:`Actor`, but it may
    sometimes be helpful to constrain this for the benefit of the end user.
    """


@dataclass
class Packet:
    """A collection of mutations and messages to send across the network.

    Packets are used to encapsulate the information returned by the ``Decoder.decode``
    and the ``Agent.decode_action`` methods.

    Attributes:
        mutations: An iterable collection of mutations to apply to the ego-actor.
        messages: An iterable collections of messages to send along the network.
    """

    mutations: Iterable[Mutation] = field(default_factory=list)
    messages: Mapping[ID, Payload] = field(default_factory=dict)
