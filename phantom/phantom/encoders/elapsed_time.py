import numpy as np
from gym.spaces import Box
from mercury import Network

from .encoder import Encoder


class ElapsedTime(Encoder[np.ndarray]):
    """Encoder providing the fraction of time elapsed since the beginning of the episode.

    The `clock` comes from the `ActorMixin`, `MarketShareFrequencyTracker` for Market Maker
    or `TradeSideFrequencyTracker` for Investor.
    """

    @property
    def output_space(self) -> Box:
        return Box(low=-1e-12, high=1.5, shape=(1,))

    @staticmethod
    def _elapsed_time_fraction(clock):
        return (
            1.0
            if clock.is_terminal
            else 2 * np.ceil(clock.elapsed / 2) / clock.terminal_time
        )

    def encode(self, ctx: Network.Context) -> np.ndarray:
        elapsed_time = self._elapsed_time_fraction(ctx.actor.clock)
        return np.array([elapsed_time])
