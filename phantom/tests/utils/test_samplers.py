from phantom.utils.samplers import UniformSampler, LambdaSampler
import pytest


@pytest.fixture
def sampler():
    return UniformSampler()


def test_comparison_with_float(sampler):
    sampler.value = sampler.sample()

    assert sampler <= 1.0
    assert sampler >= 0.0
    assert sampler == sampler.value
    assert sampler != (sampler.value + 0.1)


def test_comparison_with_sampler(sampler):
    sampler.value = 0.5

    sampler2 = UniformSampler()
    sampler2.value = 0.5

    assert not (sampler == sampler2)
    assert sampler != sampler2


def test_iterable():
    sampler1 = UniformSampler()
    sampler1.value = 0.5

    sampler2 = UniformSampler()
    sampler2.value = 0.5

    sampler3 = UniformSampler()
    sampler3.value = 0.5

    l = [sampler1, sampler2]
    assert not sampler3 in l


def test_lambda_sampler():
    def _my_func(a_, b_=0):
        return a_ + b_

    a = 5
    b = 1
    sampler = LambdaSampler(a, b_=b, func=_my_func)
    assert sampler.sample() == 6
    assert sampler.sample() == 6

    sampler = LambdaSampler(a, func=_my_func)
    assert sampler.sample() == 5
    assert sampler.sample() == 5
