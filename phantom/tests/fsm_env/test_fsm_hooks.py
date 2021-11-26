from enum import Enum

import mercury as me
import phantom as ph


class Stages(Enum):
    ODD = 1
    EVEN = 2


class MockFSMActor(ph.fsm.FSMActor):
    def __init__(self, actor_id: str) -> None:
        super().__init__(
            actor_id=actor_id,
            stage_handlers={
                Stages.ODD: OddStateActorHandler(),
                Stages.EVEN: EvenStateActorHandler(),
            },
        )

        self.odd_pre_stage_count = 0
        self.odd_post_stage_count = 0
        self.odd_pre_msg_resolution_count = 0
        self.odd_post_msg_resolution_count = 0

        self.even_pre_stage_count = 0
        self.even_post_stage_count = 0
        self.even_pre_msg_resolution_count = 0
        self.even_post_msg_resolution_count = 0

    # @me.actors.handler(str)
    # def handle_message(self, ctx, msg):
    #     yield from ()


class OddStateActorHandler(ph.fsm.StageHandler[MockFSMActor]):
    @staticmethod
    def pre_stage_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.odd_pre_stage_count += 1

    @staticmethod
    def pre_msg_resolution_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.odd_pre_msg_resolution_count += 1

    @staticmethod
    def post_msg_resolution_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.odd_post_msg_resolution_count += 1

    @staticmethod
    def post_stage_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.odd_post_stage_count += 1


class EvenStateActorHandler(ph.fsm.StageHandler[MockFSMActor]):
    @staticmethod
    def pre_stage_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.even_pre_stage_count += 1

    @staticmethod
    def pre_msg_resolution_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.even_pre_msg_resolution_count += 1

    @staticmethod
    def post_msg_resolution_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.even_post_msg_resolution_count += 1

    @staticmethod
    def post_stage_hook(actor: MockFSMActor, ctx: me.Network.Context) -> None:
        actor.even_post_stage_count += 1


class MockEnv(ph.fsm.FiniteStateMachineEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        actors = [MockFSMActor("actor")]

        network = me.Network(me.resolvers.UnorderedResolver(), actors)

        super().__init__(
            network=network,
            n_steps=3,
            initial_stage=Stages.ODD,
            stages=[
                ph.fsm.FSMStage(
                    stage_id=Stages.ODD,
                    next_stages=[Stages.EVEN],
                ),
                ph.fsm.FSMStage(
                    stage_id=Stages.EVEN,
                    next_stages=[Stages.ODD],
                ),
            ],
        )


def test_fsm_hooks():
    env = MockEnv()

    actor = env.network.actors["actor"]

    env.reset()
    assert env.current_stage == Stages.ODD

    assert actor.odd_pre_stage_count == 1
    assert actor.odd_post_stage_count == 0
    assert actor.odd_pre_msg_resolution_count == 0
    assert actor.odd_post_msg_resolution_count == 0

    assert actor.even_pre_stage_count == 0
    assert actor.even_post_stage_count == 0
    assert actor.even_pre_msg_resolution_count == 0
    assert actor.even_post_msg_resolution_count == 0

    # Step through ODD stage and into EVEN stage
    env.step(actions={})
    assert env.current_stage == Stages.EVEN

    assert actor.odd_pre_stage_count == 1
    assert actor.odd_post_stage_count == 1
    assert actor.odd_pre_msg_resolution_count == 1
    assert actor.odd_post_msg_resolution_count == 1

    assert actor.even_pre_stage_count == 1
    assert actor.even_post_stage_count == 0
    assert actor.even_pre_msg_resolution_count == 0
    assert actor.even_post_msg_resolution_count == 0

    # Step through EVEN stage and into ODD stage
    env.step(actions={})
    assert env.current_stage == Stages.ODD

    assert actor.odd_pre_stage_count == 2
    assert actor.odd_post_stage_count == 1
    assert actor.odd_pre_msg_resolution_count == 1
    assert actor.odd_post_msg_resolution_count == 1

    assert actor.even_pre_stage_count == 1
    assert actor.even_post_stage_count == 1
    assert actor.even_pre_msg_resolution_count == 1
    assert actor.even_post_msg_resolution_count == 1

    # Step through ODD stage and into EVEN stage
    env.step(actions={})
    assert env.current_stage == Stages.EVEN

    assert actor.odd_pre_stage_count == 2
    assert actor.odd_post_stage_count == 2
    assert actor.odd_pre_msg_resolution_count == 2
    assert actor.odd_post_msg_resolution_count == 2

    assert actor.even_pre_stage_count == 2
    assert actor.even_post_stage_count == 1
    assert actor.even_pre_msg_resolution_count == 1
    assert actor.even_post_msg_resolution_count == 1
