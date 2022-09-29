import phantom as ph


def test_rollout_class():
    rollout = ph.utils.rollout.Rollout(
        rollout_id=0,
        repeat_id=0,
        env_config={},
        rollout_params={},
        steps=[
            ph.utils.rollout.Step(
                i=0,
                observations={"agent": {"obs": 1}},
                rewards={"agent": 1.0},
                dones={"agent": {"obs": 1}},
                infos={"agent": {"info": 1}},
                actions={"agent": {"action": 1}},
                messages=None,
                stage=None,
            ),
            ph.utils.rollout.Step(
                i=0,
                observations={},
                rewards={},
                dones={},
                infos={},
                actions={},
                messages=None,
                stage=None,
            ),
        ],
        metrics={},
    )

    obs = rollout.observations_for_agent("agent", drop_nones=False)
    assert obs == [{"obs": 1}, None]
    obs = rollout.observations_for_agent("agent", drop_nones=True)
    assert obs == [{"obs": 1}]

    rewards = rollout.rewards_for_agent("agent", drop_nones=False)
    assert rewards == [1.0, None]
    rewards = rollout.rewards_for_agent("agent", drop_nones=True)
    assert rewards == [1.0]

    dones = rollout.rewards_for_agent("agent", drop_nones=False)
    assert dones == [1.0, None]
    dones = rollout.rewards_for_agent("agent", drop_nones=True)
    assert dones == [1.0]

    infos = rollout.rewards_for_agent("agent", drop_nones=False)
    assert infos == [1.0, None]
    infos = rollout.rewards_for_agent("agent", drop_nones=True)
    assert infos == [1.0]

    actions = rollout.rewards_for_agent("agent", drop_nones=False)
    assert actions == [1.0, None]
    actions = rollout.rewards_for_agent("agent", drop_nones=True)
    assert actions == [1.0]
