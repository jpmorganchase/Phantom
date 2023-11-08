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
    assert np.all(
        results[0].actions_for_agent("a1")
        == results[1].actions_for_agent("a1")
        == results[2].actions_for_agent("a1")
    )

    # Data export:
    ph.utils.rollout.rollouts_to_dataframe(results, avg_over_repeats=False)

    with open(f"{tmpdir}/rollouts.json", "w") as f:
        ph.utils.rollout.rollouts_to_jsonl(results, f)

    # TODO: fix (very small floating point rounding difference in some actions)
    # With batched inference:
    # results2 = ph.utils.rllib.rollout(
    #     directory=f"{tmpdir}/LATEST",
    #     env_class=MockEnv,
    #     env_config={},
    #     num_repeats=3,
    #     num_workers=1,
    #     policy_inference_batch_size=3,
    # )

    # assert results == list(results2)

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
    results = list(results)

    assert results[0][0] == {"r": 0.0}
    assert results[1][0] == {"r": 0.5}
    assert results[2][0] == {"r": 1.0}
    assert results[0][1][0] == 0.0
    assert results[1][1][0] == 0.5
    assert results[2][1][0] == 1.0

    # Evaluate policy (explore=True):
    results = ph.utils.rllib.evaluate_policy(
        directory=f"{tmpdir}/LATEST",
        obs=[ph.utils.ranges.LinspaceRange(0.0, 1.0, 3, name="r")],
        policy_id="mock_policy",
        explore=True,
    )
    results = list(results)

    assert results[0][0] == {"r": 0.0}
    assert results[1][0] == {"r": 0.5}
    assert results[2][0] == {"r": 1.0}
    assert results[0][1][0] == 0.0
    assert results[1][1][0] == 0.5
    assert results[2][1][0] == 1.0


def test_rllib_rollout_vectorized_fsm_env(tmpdir):
    # Non-stochastic FSM Env:
    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("A")]
            network = ph.Network(agents)
            super().__init__(num_steps=1, network=network, initial_stage="StageA")

        @ph.FSMStage(stage_id="StageA", acting_agents=["A"], next_stages=["StageA"])
        def handle(self):
            return "StageA"

    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=Env,
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

    results1 = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_config={},
        num_repeats=3,
        num_workers=1,
        policy_inference_batch_size=1,
    )

    results2 = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_config={},
        num_repeats=3,
        num_workers=1,
        policy_inference_batch_size=3,
    )

    assert list(results1) == list(results2)

    # Stochastic FSM Env:
    class Env2(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("A")]
            network = ph.Network(agents)
            super().__init__(num_steps=1, network=network, initial_stage="StageA")

        @ph.FSMStage(
            stage_id="StageA", acting_agents=["A"], next_stages=["StageA", "StageB"]
        )
        def handleA(self):
            return "StageB"

        @ph.FSMStage(
            stage_id="StageB", acting_agents=["A"], next_stages=["StageA", "StageB"]
        )
        def handleB(self):
            return "StageA"

    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=Env2,
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

    with pytest.raises(ValueError):
        list(
            ph.utils.rllib.rollout(
                directory=f"{tmpdir}/LATEST",
                env_config={},
                num_repeats=3,
                num_workers=1,
                policy_inference_batch_size=3,
            )
        )


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
