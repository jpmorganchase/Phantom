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
        actions = {
            agent.id: agent.action_space.sample()
            for agent in env.strategic_agents
        }

        observations = env.step(actions).observations

If this method is used, any supertypes must manually be passed to the agent's
:meth:`__init__()` method or manually set as the :attr:`.supertype` property of the
agent before the :meth:`reset()` method is called:

.. code-block:: python

    env = SupplyChainEnv()

    env["SHOP"].supertype = ShopAgent.Supertype(...)


Enabling Telemetry
------------------

Phantom includes a powerful tool for aiding the development and debugging of
environments called Telemetry. There are two output destinations for the output: print
logging (to the terminal) and file logging:

.. code-block:: python

    ph.telemetry.logger.configure_print_logging(enable=True)
    ph.telemetry.logger.configure_file_logging(file_path="log.json", append=False)

    env = SupplyChainEnv()

    observations = env.reset()

    while not env.is_done():
        actions = {
            agent.id: agent.action_space.sample()
            for agent in env.strategic_agents
        }

        observations = env.step(actions).observations


There are many options to configure what will be logged for both functions, see the
:ref:`api_telemetry` page for full details.

.. warning::
    This feature is only designed for single-process execution. Using this with multiple
    processes/workers may lead to invalid output.

The following is an example of what the print logging looks like:

.. figure:: /img/telemetry_stdout.png
   :figclass: align-center
   :width: 80%

Phantom also comes with a handy easy to use web viewer for log files:

.. figure:: /img/telemetry_streamlit.png
   :figclass: align-center
   :width: 90%

This can be opened by running:

.. code-block:: bash

    streamlit run scripts/view_telemetry.py <log_file>

Once started, navigate to ``http://localhost:8501`` in your browser.
