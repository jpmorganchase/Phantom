.. _api_supertypes:

Types & Supertypes
==================

Types and supertypes can be used to extend Agents in experiments.

.. Basic example:

    .. code-block:: python

        # We want to model an agent that we name TrafficAgent.

        # We define a base type for our agent:
        @dataclass
        class TrafficAgentType(ph.BaseType):
            speed_reward_factor: float

        # Next we define our agent that encodes this type:
        class TrafficAgent(ph.Agent[TrafficAgentType]):
            def __init__(
                self,
                agent_id: mercury.ID,
                supertype: ph.Supertype[MarketMakerType],
            ) -> None:
                super().__init__(agent_id, supertype=supertype)

            def reset(self) -> None:
                super().reset()

                self.reward_function = TrafficAgentReward(speed_reward_factor=self.type.speed_reward_factor)

        # Next we define a supertype that produces this type:
        class TrafficAgentSupertype1(ph.Supertype[TrafficAgentType]):
            def sample(self) -> TrafficAgentType:
                return TrafficAgentType(speed_reward_factor=np.random.normal(0.0, 1.0))

        # Now we can create an instance of our agent:
        t_agent1 = TrafficAgent("TA1", supertype=TrafficAgentSupertype1())

        # We can also define supertypes that take parameters:
        class TrafficAgentSupertype2(ph.Supertype[TrafficAgentType]):
            def __init__(self, mean: float, stddev: float) -> TrafficAgentType:
                self.mean = mean
                self.stddev = stddev

            def sample(self) -> TrafficAgentType:
                return TrafficAgentType(speed_reward_factor=np.random.normal(self.mean, self.stddev))

        t_agent2 = TrafficAgent("TA1", supertype=TrafficAgentSupertype2(2.0, 0.5))


.. autoclass:: phantom.supertype.BaseSupertype
   :inherited-members:


.. autoclass:: phantom.supertype.BaseType
   :inherited-members:
