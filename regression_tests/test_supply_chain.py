import phantom as ph
import pytest

from . import compare_rollouts, import_file


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
        num_workers=0,
        iterations=10,
        checkpoint_freq=10,
        results_dir=tmpdir,
    )

    rollouts = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        num_repeats=5,
        num_workers=0,
        metrics=supply_chain.metrics,
    )

    rollouts = list(sorted(rollouts, key=lambda r: r.rollout_id))

    file_path = f"regression_tests/data/supply-chain-2023-02-09.json"

    # TO GENERATE FILE:
    # import json
    # from . import RolloutJSONEncoder
    # with open(file_path, "w") as file:
    #     json.dump(rollouts, file, cls=RolloutJSONEncoder, indent=4)

    compare_rollouts(file_path, rollouts)
