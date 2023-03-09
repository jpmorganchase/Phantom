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

    class SimpleAgent(ph.Agent):
        # We define a supertype dataclass for our agent:
        @dataclass
        class Supertype(ph.Supertype):
            # Each field in the dataclass is a parameter of the Type.
            skill_weight: float = 1.0

        # We don't need to provide an instance of the Supertype class when we create
        # instances of the agent class.
        def __init__(self, agent_id: ph.AgentID):
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
fixed values or :class:`Sampler` classes. Each time the supertype is sampled, a new
'type' object is created containing the sampled values, and is attached to the
respective agent.

Initialising supertypes in this scenario can be done in one of two ways. The first is
passing in a Supertype instance:

.. code-block:: python
    
    ph.utils.rllib.train(
        ...
        env_config={
            "agent_supertypes": {
                "SIMPLE_AGENT": SimpleAgent.Supertype(
                    # When training is run, for each episode the 'skill_weight' parameter
                    # will be uniformly sampled from the range 0.0 to 1.0:
                    skill_weight: UniformFloatSampler(0.0, 1.0)
                )
            }
        }
        ...
    )

The second is to pass in a dict that is used to populate the supertype, this relies on
the agent that the supertype is intended for having a Supertype sub class defined in it:

.. code-block:: python
    
    ph.utils.rllib.train(
        ...
        env_config={
            "agent_supertypes": {
                "SIMPLE_AGENT": {
                    # When training is run, for each episode the 'skill_weight' parameter
                    # will be uniformly sampled from the range 0.0 to 1.0:
                    "skill_weight" : UniformFloatSampler(0.0, 1.0)
                }
            }
        }
        ...
    )

Afterwards, when we perform rollouts we instead initialise each supertype with either
fixed values or :class:`Range` classes that will sample over a fixed set of values.

The following shows the use of :class:`Ranges` with the :func:`utils.rllib.rollout`
function:


.. code-block:: python
    
    ph.utils.rllib.rollout(
        ...
        agent_supertypes={
            "SIMPLE_AGENT": SimpleAgent.Supertype(
                # 11 rollouts will be performed, each with a value along the linearly
                # spaced range from 0.0 to 1.0:
                skill_weight: LinspaceRange(0.0, 1.0, n=11)
            )
        }
        ...
    )


Supertypes can also be applied to the environment as a whole, this is useful in
scenarios such as varying the stochastic network connectivity probabilities:

.. code-block:: python

    class SimpleEnv(ph.PhantomEnvironment):
        # We define a base supertype dataclass for the env, just as we do for an agent:
        @dataclass
        class Supertype(ph.Supertype):
            avg_connectivity: float = 0.5

        def __init__(self, env_supertype, **kwargs):
            agents = [
                SimpleAgent("a1"),
                SimpleAgent("a2"),
            ]

            network = StochasticNetwork(agents)

            network.add_connection("a1", "a2", env_supertype.avg_connectivity)

            super().__init__(
                num_steps=10, network=network, env_supertype=env_supertype, **kwargs
            )
