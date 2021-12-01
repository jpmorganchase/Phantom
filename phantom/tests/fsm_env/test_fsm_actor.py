import phantom as ph


class StageHandler(ph.fsm.StageHandler[ph.fsm.FSMActor]):
    pass


def test_fsm_actor():
    handler = StageHandler()
    actor = ph.fsm.FSMActor("actor", stage_handlers={"stage1": handler})
