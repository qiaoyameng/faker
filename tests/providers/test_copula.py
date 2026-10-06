import math

from collections import Counter

import pytest

from faker import Faker
from faker.providers.copula import (
    MAX_COPULA_DIMENSION,
    discrete_marginal,
    lognormal_marginal,
    normal_marginal,
)


def _pearson(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    return cov / math.sqrt(var_x * var_y)


def _ranks(values):
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and values[ordered[j + 1]] == values[ordered[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = average
        i = j + 1
    return ranks


def _spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


class TestGaussianCopula:
    num_samples = 4000

    def test_perfectly_correlated_singular_matrix(self, faker):
        faker.seed_instance(31337)
        marginals = [normal_marginal(), lognormal_marginal()]
        for _ in range(100):
            normal_value, lognormal_value = faker.gaussian_copula([[1.0, 1.0], [1.0, 1.0]], marginals)
            assert lognormal_value == pytest.approx(math.exp(normal_value), rel=1e-12)

    def test_perfectly_independent(self, faker, num_samples):
        faker.seed_instance(20261006)
        marginals = [normal_marginal(), normal_marginal()]
        samples = [faker.gaussian_copula([[1.0, 0.0], [0.0, 1.0]], marginals) for _ in range(num_samples)]
        xs, ys = zip(*samples)
        assert abs(_pearson(xs, ys)) < 0.05
        assert abs(_spearman(xs, ys)) < 0.05

    @pytest.mark.parametrize(
        "correlation",
        (
            [[1.0, 2.0], [2.0, 1.0]],  # indefinite: |rho| > 1
            [[1.0, 0.5], [0.4, 1.0]],  # not symmetric
            [[1.0, 0.5], [0.5, 0.9]],  # diagonal not one
            [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5]],  # not square
            [[1.0, 0.5], [0.5, 1.0], [0.5, 0.5]],  # not square
            [],  # empty
        ),
    )
    def test_invalid_matrix_rejected(self, faker, correlation):
        marginals = [normal_marginal()] * max(2, len(correlation))
        with pytest.raises(ValueError):
            faker.gaussian_copula(correlation, marginals)

    def test_indefinite_singular_looking_matrix_rejected(self, faker):
        # Symmetric, unit diagonal, but one eigenvalue is negative.
        correlation = [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]
        with pytest.raises(ValueError, match="positive semidefinite"):
            faker.gaussian_copula(correlation, [normal_marginal()] * 3)

    def test_dimension_cap(self, faker):
        marginals = [normal_marginal()] * (MAX_COPULA_DIMENSION + 1)
        identity = [[float(i == j) for j in range(MAX_COPULA_DIMENSION + 1)] for i in range(MAX_COPULA_DIMENSION + 1)]
        with pytest.raises(ValueError, match="dimension"):
            faker.gaussian_copula(identity, marginals)
        identity_ok = [[float(i == j) for j in range(MAX_COPULA_DIMENSION)] for i in range(MAX_COPULA_DIMENSION)]
        sample = faker.gaussian_copula(identity_ok, [normal_marginal()] * MAX_COPULA_DIMENSION)
        assert len(sample) == MAX_COPULA_DIMENSION

    def test_marginal_count_mismatch_rejected(self, faker):
        with pytest.raises(ValueError, match="one quantile function per column"):
            faker.gaussian_copula([[1.0, 0.0], [0.0, 1.0]], [normal_marginal()])

    def test_non_callable_marginal_rejected(self, faker):
        with pytest.raises(ValueError, match="callable"):
            faker.gaussian_copula([[1.0]], [3.14])

    def test_randomness_belongs_to_instance(self):
        correlation = [[1.0, 0.7], [0.7, 1.0]]
        marginals = [normal_marginal(), lognormal_marginal()]
        first = Faker()
        first.seed_instance(1234)
        second = Faker()
        second.seed_instance(1234)
        for _ in range(10):
            assert first.gaussian_copula(correlation, marginals) == second.gaussian_copula(correlation, marginals)

    def test_fixed_seed_marginals_and_rank_correlation(self, faker, num_samples):
        faker.seed_instance(987654321)
        rho = 0.8
        correlation = [
            [1.0, rho, rho],
            [rho, 1.0, rho],
            [rho, rho, 1.0],
        ]
        marginals = [
            normal_marginal(mu=100.0, sigma=15.0),
            lognormal_marginal(mu=3.0, sigma=0.5),
            discrete_marginal([1, 2, 3], [0.2, 0.3, 0.5]),
        ]
        samples = [faker.gaussian_copula(correlation, marginals) for _ in range(num_samples)]
        amounts, durations, buckets = zip(*samples)

        # Normal marginal: sample moments match the requested parameters.
        mean = sum(amounts) / num_samples
        assert mean == pytest.approx(100.0, abs=1.0)
        variance = sum((x - mean) ** 2 for x in amounts) / num_samples
        assert math.sqrt(variance) == pytest.approx(15.0, rel=0.05)

        # Lognormal marginal: median matches exp(mu).
        median = sorted(durations)[num_samples // 2]
        assert median == pytest.approx(math.exp(3.0), rel=0.05)

        # Discrete marginal: empirical frequencies match the probabilities.
        frequencies = Counter(buckets)
        assert set(frequencies) == {1, 2, 3}
        assert frequencies[1] / num_samples == pytest.approx(0.2, abs=0.03)
        assert frequencies[2] / num_samples == pytest.approx(0.3, abs=0.03)
        assert frequencies[3] / num_samples == pytest.approx(0.5, abs=0.03)

        # Rank association between the continuous columns follows the
        # Gaussian copula relation rho_spearman = 6/pi * arcsin(rho / 2).
        expected_spearman = 6.0 / math.pi * math.asin(rho / 2.0)
        assert _spearman(amounts, durations) == pytest.approx(expected_spearman, abs=0.03)

        # Discretization attenuates dependence: the discrete column keeps the
        # ordering (positive association) but its rank correlation no longer
        # equals the latent coefficient.
        discrete_spearman = _spearman(amounts, buckets)
        assert 0.3 < discrete_spearman < expected_spearman - 0.05

    def test_fixed_seed_reproducibility(self, faker):
        correlation = [[1.0, 0.3], [0.3, 1.0]]
        marginals = [normal_marginal(), discrete_marginal([10, 20, 30])]
        faker.seed_instance(7)
        first_run = [faker.gaussian_copula(correlation, marginals) for _ in range(50)]
        faker.seed_instance(7)
        second_run = [faker.gaussian_copula(correlation, marginals) for _ in range(50)]
        assert first_run == second_run


class TestMarginals:
    def test_normal_marginal_requires_positive_sigma(self):
        with pytest.raises(ValueError):
            normal_marginal(sigma=0.0)

    def test_lognormal_marginal_quantiles(self):
        quantile = lognormal_marginal(mu=2.0, sigma=0.25)
        assert quantile(0.5) == pytest.approx(math.exp(2.0))
        assert quantile(0.25) < quantile(0.5) < quantile(0.75)

    def test_discrete_marginal_uniform_default(self):
        quantile = discrete_marginal([3, 1, 2])
        assert quantile(0.1) == 1
        assert quantile(0.5) == 2
        assert quantile(0.9) == 3

    def test_discrete_marginal_weighted_and_sorted(self):
        quantile = discrete_marginal([30, 10, 20], [0.5, 0.2, 0.3])
        assert quantile(0.1) == 10
        assert quantile(0.3) == 20
        assert quantile(0.6) == 30

    @pytest.mark.parametrize(
        "values,probabilities",
        (
            ([], None),
            ([1, 2], [0.5]),
            ([1, 2], [0.5, 0.4, 0.1]),
            ([1, 2], [0.5, -0.1]),
            ([1, 2], [0.0, 0.0]),
        ),
    )
    def test_discrete_marginal_invalid_input(self, values, probabilities):
        with pytest.raises(ValueError):
            discrete_marginal(values, probabilities)
