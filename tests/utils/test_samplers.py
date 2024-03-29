import pytest

from phantom.utils.samplers import (
    UniformArraySampler,
    UniformFloatSampler,
    UniformIntSampler,
    LambdaSampler,
)


@pytest.fixture
def float_sampler():
    return UniformFloatSampler()


@pytest.fixture
def int_sampler():
    return UniformIntSampler()


def test_comparison_with_float(float_sampler):
    float_sampler._value = float_sampler.sample()

    assert float_sampler <= 1.0
    assert float_sampler >= 0.0
    assert float_sampler == float_sampler._value
    assert float_sampler != (float_sampler._value + 0.1)


def test_comparison_with_int(int_sampler):
    int_sampler._value = int_sampler.sample()

    assert int_sampler == 0 or int_sampler == 1
    assert int_sampler == int_sampler._value
    assert int_sampler != (int_sampler._value + 1)


def test_comparison_with_sampler(float_sampler):
    float_sampler._value = 0.5

    float_sampler2 = UniformFloatSampler()
    float_sampler2._value = 0.5

    assert not (float_sampler == float_sampler2)
    assert float_sampler != float_sampler2


def test_iterable():
    sampler1 = UniformFloatSampler()
    sampler1._value = 0.5

    sampler2 = UniformFloatSampler()
    sampler2._value = 0.5

    sampler3 = UniformFloatSampler()
    sampler3._value = 0.5

    assert sampler3 not in [sampler1, sampler2]


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


def test_asserts():
    with pytest.raises(AssertionError):
        UniformFloatSampler(high=0.0, low=1.0)

    with pytest.raises(AssertionError):
        UniformIntSampler(high=0, low=1)

    with pytest.raises(AssertionError):
        UniformArraySampler(high=0.0, low=1.0)
