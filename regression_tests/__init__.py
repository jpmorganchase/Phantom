import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import phantom as ph


def import_file(path: str):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RolloutJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, ph.utils.rollout.Rollout):
            return obj.__dict__
        if isinstance(obj, ph.utils.rollout.Step):
            return asdict(obj)

        return json.JSONEncoder.default(self, obj)
