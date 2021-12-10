import gym
import mercury as me
import numpy as np
import phantom as ph


class MockStageHandler(ph.fsm.StagePolicyHandler["MockFSMAgent"]):
    @staticmethod
    def compute_reward(
        agent: "MockFSMAgent", stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> float:
        return 0

    @staticmethod
    def encode_obs(
        agent: "MockFSMAgent", stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> np.ndarray:
        return np.array([0])

    @staticmethod
    def decode_action(
        agent: "MockFSMAgent", stage: ph.fsm.StageID, ctx: me.Network.Context, action
    ) -> ph.Packet:
        return ph.Packet()

    @staticmethod
    def get_observation_space(agent: "MockFSMAgent") -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    @staticmethod
    def get_action_space(agent: "MockFSMAgent") -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


class MockFSMAgent(ph.fsm.FSMAgent):
    def __init__(self, id: str, stage_handlers) -> None:
        super().__init__(agent_id=id, stage_handlers=stage_handlers)

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()


class MockAgent(ph.Agent):
    def __init__(self, agent_id: me.ID) -> None:
        super().__init__(agent_id)

        self.compute_reward_count = 0
        self.encode_obs_count = 0
        self.decode_action_count = 0

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        self.decode_action_count += 1
        return ph.agent.Packet()

    def encode_obs(self, ctx: me.Network.Context):
        self.encode_obs_count += 1
        return np.zeros((1,))

    def compute_reward(self, ctx: me.Network.Context) -> float:
        self.compute_reward_count += 1
        return 0.0

    def get_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def get_action_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))
