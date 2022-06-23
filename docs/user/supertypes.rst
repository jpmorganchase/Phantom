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

    # We want to model an agent called SimpleAgent that completes tasks of varying
    # difficulty each step. Our agent has a skill parameter that affects how likely the
    # agent is to complete the task. We want to train the agent for many different skill
    # levels to learn a generalised policy.

    # We define a base supertype dataclass for our agent:
    @dataclass
    class SimpleAgentSupertype(ph.BaseSupertype):
        # Each field in the dataclass is a parameter of the Type.
        skill_weight: ph.SupertypeField[float]

    # Next we define our agent that encodes this type:
    class SimpleAgent(ph.Agent[SimpleAgentSupertype]):
        # We don't need to provide an instance of the SimpleAgentSupertype class when we
        # create instances of the agent class.
        def __init__(self, agent_id: mercury.ID):
            super().__init__(agent_id)

        def reset(self):
            # When reset() is called on the ph.Agent class the supertype is sampled
            # and values populated in the 'type' property of the agent.
            return super().reset()

        def compute_reward(self, ctx) -> float:
            task_difficulty = random()

            # We access the sampled values through the 'type' property on the agent
            if task_difficulty > self.type.skill:
                return REWARD_FOR_COMPLETE
            else:
                return PENALTY_FOR_NO_COMPLETE

When we call the train method we setup the sampling of the supertype with a mapping of
agent IDs to supertype instances. We initialise each supertype instance with either
fixed values or 'Sampler' classes. Each time the supertype is sampled, a new 'type'
object is created containing the sampled values, and is attached to the respective agent.

.. code-block:: python
    
    ph.train(
        ...
        agent_supertypes={
            "SIMPLE_AGENT": SimpleAgentSupertype(
                # When training is run, for each episode the 'skill_weight' parameter
                # will be uniformly sampled from the range 0.0 to 1.0:
                skill_weight: UniformSampler(0.0, 1.0)
            )
        }
        ...
    )

Afterwards, when we call the rollout method we instead initialise each supertype with
either fixed values or 'Range' classes:

.. code-block:: python
    
    ph.rollout(
        ...
        agent_supertypes={
            "SIMPLE_AGENT": SimpleAgentSupertype(
                # 11 rollouts will be performed, each with a value along the linearly
                # spaced range from 0.0 to 1.0:
                skill_weight: LinspaceRange(0.0, 1.0, n=11)
            )
        }
        ...
    )