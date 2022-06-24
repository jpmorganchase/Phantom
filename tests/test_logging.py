import phantom as ph


class MockAgent:
    def __init__(self, inc):
        self.inc = inc
        self.test_property = 0.0

    def step(self):
        self.test_property += self.inc


class MockEnv:
    def __init__(self):
        self.test_property = 0.0

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

    metric = ph.logging.SimpleEnvMetric(
        env_property="test_property",
        reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 5.0


def test_simple_env_metric_2():
    env = MockEnv()

    metric = ph.logging.SimpleEnvMetric(
        env_property="test_property",
        reduce_action="mean",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 3.0


def test_simple_env_metric_3():
    env = MockEnv()

    metric = ph.logging.SimpleEnvMetric(
        env_property="test_property",
        reduce_action="sum",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 15.0


def test_simple_agent_metric_1():
    env = MockEnv()

    metric = ph.logging.SimpleAgentMetric(
        agent_id="agent1",
        agent_property="test_property",
        reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 5.0


def test_simple_agent_metric_2():
    env = MockEnv()

    metric = ph.logging.SimpleAgentMetric(
        agent_id="agent1",
        agent_property="test_property",
        reduce_action="mean",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 3.0


def test_simple_agent_metric_3():
    env = MockEnv()

    metric = ph.logging.SimpleAgentMetric(
        agent_id="agent1",
        agent_property="test_property",
        reduce_action="sum",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 15.0


def test_aggregated_agent_metric_1():
    env = MockEnv()

    metric = ph.logging.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="min",
        reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 5.0


def test_aggregated_agent_metric_2():
    env = MockEnv()

    metric = ph.logging.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="max",
        reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 10.0


def test_aggregated_agent_metric_3():
    env = MockEnv()

    metric = ph.logging.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="mean",
        reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 7.5


def test_aggregated_agent_metric_4():
    env = MockEnv()

    metric = ph.logging.AggregatedAgentMetric(
        agent_ids=["agent1", "agent2"],
        agent_property="test_property",
        group_reduce_action="sum",
        reduce_action="last",
    )

    values = []

    for _ in range(5):
        env.step()
        values.append(metric.extract(env))

    assert metric.reduce(values) == 15.0
