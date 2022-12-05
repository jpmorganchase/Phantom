import json

import numpy as np
import pandas as pd
import phantom as ph
import pytest

from . import import_file, RolloutJSONEncoder


@pytest.fixture
def supply_chain():
    return import_file("examples/environments/supply_chain/supply_chain.py")


def test_supply_chain(tmpdir, supply_chain):
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=supply_chain.SupplyChainEnv,
        policies={
            "shop_policy": supply_chain.ShopAgent,
        },
        rllib_config={
            "seed": 1,
            "disable_env_checking": True,
        },
        tune_config={
            "checkpoint_freq": 10,
            "local_dir": tmpdir,
            "stop": {
                "training_iteration": 10,
            },
        },
    )

    rollouts = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/PPO/LATEST",
        num_repeats=5,
        metrics=supply_chain.metrics,
    )

    rollouts = list(sorted(rollouts, key=lambda r: r.rollout_id))

    file_path = "regression_tests/data/supply-chain-2022-12-05.json"

    # # TO GENERATE FILE:
    # with open(file_path, "w") as file:
    #     json.dump(rollouts, file, cls=RolloutJSONEncoder, indent=4)

    with open(file_path, "r") as file:
        previous_rollouts_str = file.read().split("\n")

    rollouts_str = json.dumps(rollouts, cls=RolloutJSONEncoder, indent=4).split("\n")

    assert len(rollouts_str) == len(previous_rollouts_str)

    for i, (old, new) in enumerate(zip(previous_rollouts_str, rollouts_str)):
        assert old == new, (i + 1, old, new)
