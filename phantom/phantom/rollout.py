from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import mercury as me


@dataclass
class RolloutReplay:
    """
    Class describing all the actions, observations, rewards, infos and dones of
    a single episode.

    Generated when using the phantom-rollout command.
    """

    observations: List[Dict[me.ID, Any]]
    rewards: List[Dict[me.ID, float]]
    dones: List[Dict[me.ID, bool]]
    infos: List[Dict[me.ID, Dict[str, Any]]]
    actions: List[Dict[me.ID, Any]]
