.. _env_execution:

Environment Execution
=====================

An environment episode can be manually executed using it's :meth:`reset()` and
:meth:`step()` methods. See the simple example below that runs through a single complete
episode, using random actions sampled from the agent's action spaces and prints out the
evaluated actions and observations:

.. code-block:: python

    env = SupplyChainEnv()

    observations = env.reset()

    while not env.is_done():
        print(f"\nSTEP: {env.current_step}")
        print(f"OBS:")
        for agent, obs in observations.items():
            print(f"\t{agent}: {obs}\n")
        
        actions = {
            agent.id: agent.action_space.sample()
            for agent in env.strategic_agents
        }

        print(f"ACTIONS:")
        for agent, action in actions.items():
            print(f"\t{agent}: {action}")

        observations = env.step(actions).observations

If this method is used, any supertypes must manually be passed to the agent's
:meth:`__init__()` method or manually set as the :attr:`.supertype` property of the
agent before the :meth:`reset()` method is called:

.. code-block:: python

    env = SupplyChainEnv()

    env["SHOP"].supertype = ShopAgent.Supertype(...)
