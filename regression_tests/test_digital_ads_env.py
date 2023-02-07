import phantom as ph
import pytest

from . import compare_rollouts, import_file


@pytest.fixture
def digital_ads_market():
    return import_file("examples/environments/digital_ads_market/digital_ads_market.py")


def test_digital_ads_market(tmpdir, digital_ads_market):
    dam = digital_ads_market

    policies = {
        "adv_policy_travel": [
            f"ADV_{i}" for i in range(1, dam.NUM_TRAVEL_ADVERTISERS + 1)
        ],
        "adv_policy_tech": [
            f"ADV_{i}"
            for i in range(
                dam.NUM_TRAVEL_ADVERTISERS + 1,
                dam.NUM_TRAVEL_ADVERTISERS + dam.NUM_TECH_ADVERTISERS + 1,
            )
        ],
        "adv_policy_sport": [
            f"ADV_{i}"
            for i in range(
                dam.NUM_TRAVEL_ADVERTISERS + dam.NUM_TECH_ADVERTISERS + 1,
                dam.NUM_TRAVEL_ADVERTISERS
                + dam.NUM_TECH_ADVERTISERS
                + dam.NUM_SPORT_ADVERTISERS
                + 1,
            )
        ],
    }
    policies["publisher"] = (dam.PublisherPolicy, dam.PublisherAgent)

    agent_supertypes = {}

    # travel agency (i.e. agent 1 and 5) have a rather limited budget
    agent_supertypes.update(
        {
            f"ADV_{i}": dict(
                budget=ph.utils.samplers.UniformFloatSampler(
                    low=5.0, high=15.0 + 0.001, clip_low=5.0, clip_high=15.0
                )
                # budget=10.0,
            )
            for i in range(1, dam.NUM_TRAVEL_ADVERTISERS + 1)
        }
    )

    # sport companies have a bigger budget
    agent_supertypes.update(
        {
            f"ADV_{i}": dict(
                budget=ph.utils.samplers.UniformFloatSampler(
                    low=7.0, high=17.0 + 0.001, clip_low=7.0, clip_high=17.0
                )
                # budget=12.0,
            )
            for i in range(
                dam.NUM_TRAVEL_ADVERTISERS + 1,
                dam.NUM_TRAVEL_ADVERTISERS + dam.NUM_TECH_ADVERTISERS + 1,
            )
        }
    )

    # tech companies have the bigger budget
    agent_supertypes.update(
        {
            f"ADV_{i}": dict(
                budget=ph.utils.samplers.UniformFloatSampler(
                    low=10.0,
                    high=20.0 + 0.001,
                    clip_low=10.0,
                    clip_high=20.0,
                )
                # budget=15.0,
            )
            for i in range(
                dam.NUM_TRAVEL_ADVERTISERS + dam.NUM_TECH_ADVERTISERS + 1,
                dam.NUM_TRAVEL_ADVERTISERS
                + dam.NUM_TECH_ADVERTISERS
                + dam.NUM_SPORT_ADVERTISERS
                + 1,
            )
        }
    )

    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=dam.DigitalAdsEnv,
        policies=policies,
        env_config={
            "agent_supertypes": agent_supertypes,
            "num_agents_theme": {
                "travel": dam.NUM_TRAVEL_ADVERTISERS,
                "tech": dam.NUM_TECH_ADVERTISERS,
                "sport": dam.NUM_SPORT_ADVERTISERS,
            },
        },
        rllib_config={
            "batch_mode": "complete_episodes",
            "disable_env_checking": True,
        },
        iterations=1,
        checkpoint_freq=1,
        results_dir=tmpdir,
    )

    agent_supertypes = {}
    # travel agency (i.e. agent 1 and 5) have a rather limited budget
    agent_supertypes.update(
        {
            f"ADV_{i}": dict(budget=10.0)
            for i in range(1, dam.NUM_TRAVEL_ADVERTISERS + 1)
        }
    )

    # sport companies have a bigger budget
    agent_supertypes.update(
        {
            f"ADV_{i}": dict(budget=12.0)
            for i in range(
                dam.NUM_TRAVEL_ADVERTISERS + 1,
                dam.NUM_TRAVEL_ADVERTISERS + dam.NUM_TECH_ADVERTISERS + 1,
            )
        }
    )

    # tech companies have the bigger budget
    agent_supertypes.update(
        {
            f"ADV_{i}": dict(budget=15.0)
            for i in range(
                dam.NUM_TRAVEL_ADVERTISERS + dam.NUM_TECH_ADVERTISERS + 1,
                dam.NUM_TRAVEL_ADVERTISERS
                + dam.NUM_TECH_ADVERTISERS
                + dam.NUM_SPORT_ADVERTISERS
                + 1,
            )
        }
    )

    rollouts = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        num_repeats=5,
        metrics=dam.metrics,
        env_config={
            "agent_supertypes": agent_supertypes,
            "num_agents_theme": {
                "travel": dam.NUM_TRAVEL_ADVERTISERS,
                "tech": dam.NUM_TECH_ADVERTISERS,
                "sport": dam.NUM_SPORT_ADVERTISERS,
            },
        },
    )

    rollouts = list(sorted(rollouts, key=lambda r: r.rollout_id))

    file_path = f"regression_tests/data/digital-ads-market-2023-02-07.json"

    # TO GENERATE FILE:
    # import json
    # from . import RolloutJSONEncoder
    # with open(file_path, "w") as file:
    #     json.dump(rollouts, file, cls=RolloutJSONEncoder, indent=4)

    compare_rollouts(file_path, rollouts)
