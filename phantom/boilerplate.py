"""
THIS FILE CONTAINS BOILERPLATE IMPLEMENTATIONS OF SOME PHANTOM CLASSES AND FEATURES
"""

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple, Union

import gym
import mercury as me
import numpy as np
import phantom as ph


SpaceType = Union[
    int,
    float,
    np.ndarray,
    Dict[str, "SpaceType"],
    List["SpaceType"],
    Tuple["SpaceType", ...],
]


"""
MESSAGE DEFINITION
"""


@dataclass(frozen=True)
class ExampleMessage(me.Payload):
    int_field: int
    str_field: str


"""
FIXED POLICY DEFINITION
"""


class ExampleFixedPolicy(ph.FixedPolicy):
    def __init__(
        self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: Dict
    ) -> None:
        super().__init__(obs_space, action_space, config)

        self.param = config["param"]

    def compute_action(self, obs: SpaceType) -> SpaceType:
        return 0.0


"""
SUPERTYPE DEFINITION
"""


@dataclass
class ExampleSupertype(ph.BaseSupertype):
    field: ph.SupertypeField[float]


"""
AGENT DEFINITION
"""


class ExampleAgent(ph.Agent):
    def __init__(self, agent_id: me.ID):
        super().__init__(
            agent_id,
            # If using a fixed policy:
            # policy_class=ExampleFixedPolicy,
            # policy_config=dict(param=10),
        )

    @me.actors.handler(ExampleMessage)
    def handle_example_message(
        self, ctx: me.Network.Context, msg: me.Message
    ) -> Iterator[Tuple[me.ID, Iterable[me.Payload]]]:
        self.tracked_value = msg.payload.int_field
        yield from ()

    def decode_action(
        self, ctx: me.Network.Context, action: SpaceType
    ) -> ph.packet.Packet:
        return ph.packet.Packet(messages={"example-actor": [ExampleMessage(1, "req")]})

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0.0

    def encode_obs(self, ctx: me.Network.Context) -> SpaceType:
        return (self.type.to_obs_space_compatible_type(), np.random.random((10, 10)))

    def reset(self) -> None:
        super().reset()  # self.type set here

    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Tuple(
            (
                self.type.to_obs_space(),
                gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10, 10)),
            )
        )

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(100)


"""
ACTOR DEFINITION
"""


class ExampleActor(me.actors.SimpleSyncActor):
    def __init__(self, actor_id: me.ID) -> None:
        super().__init__(actor_id)

    @me.actors.handler(ExampleMessage)
    def handle_example_message(
        self, ctx: me.Network.Context, msg: me.Message
    ) -> Iterator[Tuple[me.ID, Iterable[me.Payload]]]:
        yield (msg.sender_id, [ExampleMessage(msg.payload.int_field * 2, "res")])


"""
ENVIRONMENT DEFINITION
"""

AGENT_IDS = [f"example-agent-{i}" for i in range(3)]
ACTOR_IDS = ["example-actor"]


class ExampleEnv(ph.PhantomEnv):

    env_name: str = "example-env"

    def __init__(self):
        agents = [ExampleAgent(aid) for aid in AGENT_IDS]
        actors = [ExampleActor(aid) for aid in ACTOR_IDS]

        network = me.Network(me.resolvers.UnorderedResolver(), agents + actors)

        network.add_connections_between(AGENT_IDS, ACTOR_IDS)

        clock = ph.Clock(start_time=0, terminal_time=100, increment=1)

        super().__init__(network=network, clock=clock)


if __name__ == "__main__":
    metrics = {}

    for id in AGENT_IDS:
        metrics[f"example_metric/{id}"] = ph.logging.SimpleAgentMetric(
            id, "tracked_value", "mean"
        )

    if sys.argv[1].lower() == "train":
        ph.train(
            experiment_name="boilerplate",
            algorithm="PPO",
            num_workers=3,
            num_episodes=100,
            env_class=ExampleEnv,
            agent_supertypes={
                aid: ExampleSupertype(
                    ph.utils.samplers.UniformSampler(low=0.0, high=1.0)
                )
                for aid in AGENT_IDS
            },
            metrics=metrics,
        )

    elif sys.argv[1].lower() == "rollout":
        ph.rollout(
            directory="boilerplate/LATEST",
            algorithm="PPO",
            num_workers=3,
            num_repeats=5,
            env_class=ExampleEnv,
            agent_supertypes={
                id: ExampleSupertype(
                    ph.utils.ranges.UniformRange(
                        start=0.0, end=1.0, step=0.2, name=f"Example Supertype Range"
                    )
                )
                for id in AGENT_IDS
            },
            metrics=metrics,
            save_messages=True,
            save_trajectories=True,
        )
