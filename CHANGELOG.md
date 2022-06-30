# Changelog

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