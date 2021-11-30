import gym.spaces
import mercury as me
import numpy as np
import phantom as ph


class StageHandler(ph.fsm.StagePolicyHandler[ph.fsm.FSMAgent]):
    @staticmethod
    def compute_reward(
        agent: ph.fsm.FSMAgent, stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> float:
        agent.compute_reward_count += 1
        return 0

    @staticmethod
    def encode_obs(
        agent: ph.fsm.FSMAgent, stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> np.ndarray:
        agent.encode_obs_count += 1
        return np.array([1])

    @staticmethod
    def decode_action(
        agent: ph.fsm.FSMAgent, stage: ph.fsm.StageID, ctx: me.Network.Context, action
    ) -> ph.Packet:
        agent.decode_action_count += 1
        return ph.Packet(messages={agent.id: ["message"]})

    @staticmethod
    def get_observation_space(agent: ph.fsm.FSMAgent) -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    @staticmethod
    def get_action_space(agent: ph.fsm.FSMAgent) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


def test_fsm_agent():
    handler = StageHandler()
    agent = ph.fsm.FSMAgent("agent", stage_handlers={"stage1": handler})
