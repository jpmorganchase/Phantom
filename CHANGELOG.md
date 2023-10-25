# Changelog

## Version 2.1.0 (2023-10-25)

- Update many dependencies to newer versions, including ray==2.7.1 and pytorch>=2.0.
- Add option to strictly enforce which agents can send and receive specific payload
types.
- Bring digital-ads-market and simple-market example environments up to date.
- Add policy 'explore' parameter to rollout function when using RLlib.
- Loosen requirement on some samplers to allow high == low input parameters.
- Allow ray.init() configuration in training.


## Version 2.0.0 (2023-05-22)

**Major rewrite of Phantom - this version contains many backward incompatible changes.**

- The Mercury library has been subsumed into Phantom to simplify the codebase, the
installation process and development.
- It is now possible to train environments with non-RLlib trainers/algorithms.
- The Clock and EnvironmentActor classes have been removed and their functionality
replaced with the EnvView class.
- A SingleAgentEnvAdapter class has been created to allow use of Phantom environments in
single-agent RL frameworks when only one agent is learning.
- The FiniteStateMachineEnv has been simplified and a complimentary StackelbergEnv class
has been added for common use-cases.
- Additional Metric implementations have been added.
- Additional tools for logging and telemetry of environments and agents have been added.
- More comprehensive documentation and unit-tests have been added.
- Many dependencies have been updated to newer versions including gym (now gymnasium)
and RLlib.
- Performance of rollouts has been increased significantly when using RLlib by using new
APIs and by vectorizing environments.
- Pytorch is now the default backend when using RLlib.
- The minimum Python version has been increased to 3.8. 


## Version 1.1.0 (2022-06-17)

### Phantom

#### New Features

- Boilerplate/bootstrap file for creating new Envs/Agents/etc.
- `AggregatedAgentMetric` class for recording values from groups of agents.
- Option to save messages when running rollouts.
- Option to set a custom RLlib trainer when training.

#### Improvements

- Updated RLlib dependency to `1.11.0`.
- New metrics, rollouts and analysis sections in tutorial.
- Simplified environment initialisation when using types/supertypes.
- Better `Range` and `Sampler` string representation formats.
- More parameters/setup settings are saved when running rollouts.
- Additional unit tests for better coverage.

### Mercury

#### New Features

- Optional message tracking.

#### Improvements

- Better error messages when using message handlers.

### Environments

#### Improvements

- Refinement of 'Supply Chain' parameters and wording.