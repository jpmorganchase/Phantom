import pytest

from phantom.utils.samplers import UniformFloatSampler, UniformIntSampler, LambdaSampler


@pytest.fixture
def float_sampler():
    return UniformFloatSampler()


@pytest.fixture
def int_sampler():
    return UniformIntSampler()


def test_comparison_with_float(float_sampler):
    float_sampler.value = float_sampler.sample()

    assert float_sampler <= 1.0
    assert float_sampler >= 0.0
    assert float_sampler == float_sampler.value
    assert float_sampler != (float_sampler.value + 0.1)


def test_comparison_with_int(int_sampler):
    int_sampler.value = int_sampler.sample()

    assert int_sampler == 0 or int_sampler == 1
    assert int_sampler == int_sampler.value
    assert int_sampler != (int_sampler.value + 1)


def test_comparison_with_sampler(float_sampler):
    float_sampler.value = 0.5

    float_sampler2 = UniformFloatSampler()
    float_sampler2.value = 0.5

    assert not (float_sampler == float_sampler2)
    assert float_sampler != float_sampler2


def test_iterable():
    sampler1 = UniformFloatSampler()
    sampler1.value = 0.5

    sampler2 = UniformFloatSampler()
    sampler2.value = 0.5

    sampler3 = UniformFloatSampler()
    sampler3.value = 0.5

    l = [sampler1, sampler2]
    assert not sampler3 in l


def test_lambda_sampler():
    def _my_func(a_, b_=0):
        return a_ + b_

    a = 5
    b = 1
    sampler = LambdaSampler(_my_func, a, b_=b)
    assert sampler.sample() == 6
    assert sampler.sample() == 6

    sampler = LambdaSampler(_my_func, a)
    assert sampler.sample() == 5
    assert sampler.sample() == 5
