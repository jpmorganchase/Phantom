import phantom as ph


def test_is_fsm_deterministic():
    env = ph.FiniteStateMachineEnv(
        num_steps=1,
        network=ph.Network([]),
        initial_stage="A",
        stages=[ph.FSMStage(stage_id="A", acting_agents=[], next_stages=["A"])],
    )

    assert env.is_fsm_deterministic()

    env = ph.FiniteStateMachineEnv(
        num_steps=1,
        network=ph.Network([]),
        initial_stage="A",
        stages=[
            ph.FSMStage(stage_id="A", acting_agents=[], next_stages=["B"]),
            ph.FSMStage(stage_id="B", acting_agents=[], next_stages=["C"]),
            ph.FSMStage(stage_id="C", acting_agents=[], next_stages=["A"]),
        ],
    )

    assert env.is_fsm_deterministic()

    env = ph.FiniteStateMachineEnv(
        num_steps=1,
        network=ph.Network([]),
        initial_stage="A",
        stages=[
            ph.FSMStage(
                stage_id="A",
                acting_agents=[],
                next_stages=["A", "B"],
                handler=lambda x: "B",
            ),
            ph.FSMStage(
                stage_id="B",
                acting_agents=[],
                next_stages=["A", "B"],
                handler=lambda x: "A",
            ),
        ],
    )

    assert not env.is_fsm_deterministic()
