import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import phantom as ph


def import_file(path: str):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_rollouts(file_path: str, rollouts: List[ph.utils.rollout.Rollout]) -> None:
    with open(file_path, "r") as file:
        previous_rollouts_str = file.read().split("\n")

    rollouts_str = json.dumps(rollouts, cls=RolloutJSONEncoder, indent=4).split("\n")

    assert len(rollouts_str) == len(previous_rollouts_str)

    for i, (old, new) in enumerate(zip(previous_rollouts_str, rollouts_str)):
        assert old == new, f"Line {i+1} doesn't match -- Existing: '{old.strip()}' -- New: '{new.strip()}'"


class RolloutJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.number):
            return int(obj)
        if isinstance(obj, ph.utils.rollout.Rollout):
            return obj.__dict__
        if isinstance(obj, ph.utils.rollout.Step):
            return asdict(obj)

        return json.JSONEncoder.default(self, obj)
