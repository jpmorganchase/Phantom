.. _supertypes:

Supertypes
==========

Supertypes and types can be used to produce agents and environments that are generalised
over certain defined parameters. For a specific agent class or environment class we
declare the fields we want to be general over. During training, for each episode we can
randomly sample a value from a distribution for each parameter. During rollouts we can
then explore the full sample space.

Below is an example demonstrating what can be acheived: 

.. code-block:: python

    # We want to model an agent called PokerAgent that learns to play poker against
    # other agents. Our agent has a skill parameter and we want to learn a generalised
    # policy that performs well at many skill levels.

    # We define a base supertype dataclass for our agent:
    @dataclass
    class PokerAgentSupertype(ph.BaseSupertype):
        # Each field in the dataclass is a parameter of the Type.
        skill_weight: ph.SupertypeField[float]

    # Next we define our agent that encodes this type:
    class PokerAgent(ph.Agent[PokerAgentSupertype]):
        # We don't need to provide an instance of the PokerAgentSupertype class when we
        # create instances of the agent class.
        def __init__(self, agent_id: mercury.ID):
            super().__init__(agent_id)

        def reset(self):
            # When reset() is called on the ph.Agent class the supertype is sampled
            # and values populated in the 'type' property of the agent.
            return super().reset()

        def compute_reward(self, ctx) -> float:
            # We access the sampled values through the 'type' property on the agent
            if randint() > self.type.skill:
                return REWARD_FOR_WIN
            else:
                return REWARD_FOR_LOSS

    # Next we define a supertype that produces this type:
    traffic_supertype = TrafficAgentSupertype()

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
