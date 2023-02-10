import numpy as np
import phantom as ph
import pytest

from .. import MockStrategicAgent, MockEnv, MockPolicy


def test_rllib_train_rollout(tmpdir):
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=MockEnv,
        env_config={},
        policies={
            "mock_policy": MockStrategicAgent,
        },
        rllib_config={
            "disable_env_checking": True,
            "num_rollout_workers": 1,
        },
        iterations=2,
        checkpoint_freq=1,
        results_dir=tmpdir,
    )

    # Without workers, without env class:
    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_config={},
        num_repeats=3,
        num_workers=0,
    )
    assert len(list(results)) == 3

    # With workers, with env class:
    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_class=MockEnv,
        env_config={},
        num_repeats=3,
        num_workers=1,
    )
    results = list(results)
    assert len(results) == 3
    assert np.array(results[0].actions_for_agent("a1")).flatten().tolist() == [
        0.5021063685417175,
        0.5272096991539001,
        0.5454075932502747,
        0.5570195913314819,
        0.564277172088623,
    ]
    assert np.array(results[1].actions_for_agent("a1")).flatten().tolist() == [
        0.5021063685417175,
        0.5272096991539001,
        0.5454075932502747,
        0.5570195913314819,
        0.564277172088623,
    ]
    assert np.array(results[2].actions_for_agent("a1")).flatten().tolist() == [
        0.5021063685417175,
        0.5272096991539001,
        0.5454075932502747,
        0.5570195913314819,
        0.564277172088623,
    ]

    # Data export:
    ph.utils.rollout.rollouts_to_dataframe(results, avg_over_repeats=False)

    with open(f"{tmpdir}/rollouts.json", "w") as f:
        ph.utils.rollout.rollouts_to_jsonl(results, f)

    # With batched inference:
    results2 = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_class=MockEnv,
        env_config={},
        num_repeats=3,
        num_workers=1,
        policy_inference_batch_size=3,
    )

    assert results == list(results2)

    # With custom policy mapping:
    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_class=MockEnv,
        env_config={},
        custom_policy_mapping={"a1": MockPolicy},
        num_repeats=1,
        num_workers=1,
    )
    assert list(results)[0].actions_for_agent("a1") == [1, 1, 1, 1, 1]

    # Evaluate policy (explore=False):
    results = ph.utils.rllib.evaluate_policy(
        directory=f"{tmpdir}/LATEST",
        obs=[ph.utils.ranges.LinspaceRange(0.0, 1.0, 3, name="r")],
        policy_id="mock_policy",
        explore=False,
    )
    assert list(results) == [
        ({"r": 0.0}, [0.0], np.array([0.50210637], dtype=np.float32)),
        ({"r": 0.5}, [0.5], np.array([0.55189204], dtype=np.float32)),
        ({"r": 1.0}, [1.0], np.array([0.56885237], dtype=np.float32)),
    ]

    # Evaluate policy (explore=True):
    results = ph.utils.rllib.evaluate_policy(
        directory=f"{tmpdir}/LATEST",
        obs=[ph.utils.ranges.LinspaceRange(1.0, 1.0, 5, name="r")],
        policy_id="mock_policy",
        explore=True,
    )
    assert list(results) == [
        ({"r": 1.0}, [1.0], np.array([0.34793535], dtype=np.float32)),
        ({"r": 1.0}, [1.0], np.array([1.0], dtype=np.float32)),
        ({"r": 1.0}, [1.0], np.array([1.0], dtype=np.float32)),
        ({"r": 1.0}, [1.0], np.array([0.43988213], dtype=np.float32)),
        ({"r": 1.0}, [1.0], np.array([1.0], dtype=np.float32)),
    ]


def test_rllib_rollout_bad(tmpdir):
    # num_repeats < 1
    with pytest.raises(AssertionError):
        list(
            ph.utils.rllib.rollout(
                directory=tmpdir,
                env_class=MockEnv,
                env_config={},
                num_repeats=0,
            )
        )

    # num_repeats < 0
    with pytest.raises(AssertionError):
        list(
            ph.utils.rllib.rollout(
                directory=tmpdir,
                env_class=MockEnv,
                env_config={},
                num_workers=-1,
            )
        )
