
# RLlib DEPENDENCY CHANGES:

- Removed a lot of ph.train boilerplate code, uses raw RLlib config instead / user rolls own
- Envs/Agents don't depend on RLlib
- Phantom Trainer/Policy interface for implementing non-RLlib policies/algorithms
- Q Learning trainer / policy
- PPO trainer / policy


# BIG ARCHITECTURE CHANGES:

- Mercury merged into Phantom
- Removed Clock class
- Removed Actor class (use Agent instead)
- Removed Env actor, env has view instead accessible by agents via Context class
- Removed Mutations + Packet + message batches
- Supertype values/samplers passed into env class via env config
- 'handle_message' and 'decode_action' agent methods have same return type for messages


# SMALL ARCHITECTURE CHANGES:

- Removed SyncActor
- Renamed SimpleSyncActor to MessageHandlerAgent
- Simplified Resolvers
- Network uses BatchResolver by default
- handle_message message arg split into sender_id and message (payload) args
- 'pre/post_resolution' -> 'pre/post_message_resolution'
- Replaced 'yielding' of messages with normal 'return'
- Simplified logging code for RLlib
- Policy definition / mapping more explicit


# TO RE-ADD:

- FSM Env
- (RLlib) rollout function


# TO DISCUSS:

- Actor vs Agent
- Mutations, ordering of pre/post resolution update of state
- Supertypes, where to define


# TO UPDATE:

- Example Envs
- Docs
- Unit Tests
