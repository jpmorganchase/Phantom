from dataclasses import dataclass, field

from phantom.metrics import SimpleMetric


@dataclass
class MockEpisode:
    user_data: dict = field(default_factory=dict)
    custom_metrics: dict = field(default_factory=dict)
    media: dict = field(default_factory=dict)


class MockMetric(SimpleMetric):
    def __init__(self, value: int, reduce_action="last", fsm_stages=None):
        super().__init__(reduce_action, fsm_stages)
        self.value = value

    def extract(self, _env) -> int:
        return self.value


class MockBaseEnv:
    def __init__(self, env_class):
        self.envs = [env_class]

    def step(self, actions={}):
        for env in self.envs:
            env.step(actions)
