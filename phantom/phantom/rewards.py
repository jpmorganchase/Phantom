from abc import abstractmethod, ABC

import mercury as me


class RewardFunction(ABC):
    """A trait for types that can compute rewards from a local context.

    Note - this trait only support scalar rewards for the time being.
    """

    @abstractmethod
    def reward(self, ctx: me.Network.Context) -> float:
        """Compute the reward from context.

        Arguments:
            ctx: The local network context.
        """
        raise NotImplementedError

    def reset(self):
        """Resets the reward function."""
        pass


class Constant(RewardFunction):
    """A reward function that always returns a given constant.

    Attributes:
        value: The reward to be returned in any state.
    """

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def reward(self, _: me.Network.Context) -> float:
        return self.value
