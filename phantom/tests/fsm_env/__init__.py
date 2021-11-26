import gym
import mercury as me
import numpy as np
import phantom as ph


class MinimalStageHandler(ph.fsm.StagePolicyHandler["MinimalAgent"]):
    @staticmethod
    def compute_reward(agent: "MinimalAgent", ctx: me.Network.Context) -> float:
        return 0

    @staticmethod
    def encode_obs(agent: "MinimalAgent", ctx: me.Network.Context) -> np.ndarray:
        return np.array([0])

    @staticmethod
    def decode_action(
        agent: "MinimalAgent", ctx: me.Network.Context, action
    ) -> ph.Packet:
        return ph.Packet()

    @staticmethod
    def get_observation_space(agent: "MinimalAgent") -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    @staticmethod
    def get_action_space(agent: "MinimalAgent") -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


class MinimalAgent(ph.fsm.FSMAgent):
    def __init__(self, id: str, stage_policy_handlers) -> None:
        super().__init__(agent_id=id, stage_policy_handlers=stage_policy_handlers)

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()
