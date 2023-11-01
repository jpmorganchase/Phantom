import phantom as ph

from .. import MockStackedAgentEnv, MockStackedStrategicAgent


def test_rllib_train_stacked_agent(tmpdir):
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=MockStackedAgentEnv,
        env_config={},
        policies={
            "mock_policy": MockStackedStrategicAgent,
        },
        rllib_config={
            "disable_env_checking": True,
            "num_rollout_workers": 1,
        },
        iterations=2,
        checkpoint_freq=1,
        results_dir=tmpdir,
    )

    results = ph.utils.rllib.rollout(
        directory=f"{tmpdir}/LATEST",
        env_config={},
        num_repeats=3,
        num_workers=0,
    )
    assert len(list(results)) == 3
