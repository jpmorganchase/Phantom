import phantom as ph


class MockAgent:
    def __init__(self, inc):
        self.inc = inc
        self.test_property = 0.0

    def step(self):
        self.test_property += self.inc


class MockProperty:
    def __init__(self):
        self.sub_property = 1.0


class MockEnv:
    def __init__(self):
        self.test_property = 0.0

        self.nested_property = MockProperty()

        self.agents = {
            "agent1": MockAgent(1.0),
            "agent2": MockAgent(2.0),
        }

    def step(self):
        self.test_property += 1.0
        for agent in self.agents.values():
            agent.step()


def test_simple_env_metric_1():
    env = MockEnv()

    metric = ph.metrics.SimpleEnvMetric(
        env_property="test_property",
        train_reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 5.0


def test_simple_env_metric_2():
    env = MockEnv()

    metric = ph.metrics.SimpleEnvMetric(
        env_property="test_property",
        train_reduce_action="mean",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 3.0


def test_simple_env_metric_3():
    env = MockEnv()

    metric = ph.metrics.SimpleEnvMetric(
        env_property="test_property",
        train_reduce_action="sum",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 15.0


def test_simple_agent_metric_1():
    env = MockEnv()

    metric = ph.metrics.SimpleAgentMetric(
        agent_id="agent1",
        agent_property="test_property",
        train_reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 5.0


def test_simple_agent_metric_2():
    env = MockEnv()

    metric = ph.metrics.SimpleAgentMetric(
        agent_id="agent1",
        agent_property="test_property",
        train_reduce_action="mean",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 3.0


def test_simple_agent_metric_3():
    env = MockEnv()

    metric = ph.metrics.SimpleAgentMetric(
        agent_id="agent1",
        agent_property="test_property",
        train_reduce_action="sum",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 15.0


def test_aggregated_agent_metric_1():
    env = MockEnv()

    metric = ph.metrics.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="min",
        train_reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 5.0


def test_aggregated_agent_metric_2():
    env = MockEnv()

    metric = ph.metrics.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="max",
        train_reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 10.0


def test_aggregated_agent_metric_3():
    env = MockEnv()

    metric = ph.metrics.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="mean",
        train_reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 7.5


def test_aggregated_agent_metric_4():
    env = MockEnv()

    metric = ph.metrics.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="sum",
        train_reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values, mode="train") == 15.0


def test_nested_metric():
    env = MockEnv()

    metric = ph.metrics.SimpleEnvMetric(
        env_property="nested_property.sub_property",
        train_reduce_action="last",
    )

    env.step()
    assert metric.extract(env) == 1.0
