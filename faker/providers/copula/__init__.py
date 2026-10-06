import math

from bisect import bisect_left
from statistics import NormalDist
from typing import Callable, List, Optional, Sequence

from .. import BaseProvider

#: Largest dimension accepted by :meth:`Provider.gaussian_copula`. The copula
#: is meant for small stress-test scenarios; keeping a hard cap also keeps the
#: pure-stdlib matrix factorization cheap and predictable.
MAX_COPULA_DIMENSION = 16

#: Quantile functions are only defined on the open interval (0, 1). The
#: probability integral transform can, in rare floating-point cases, return
#: exactly 0.0 or 1.0, so uniforms are clamped to this range before being
#: handed to the caller-supplied quantile functions.
_U_MIN = 1e-15
_U_MAX = 1.0 - 1e-15

_STANDARD_NORMAL = NormalDist()

MarginalQuantile = Callable[[float], float]


def normal_marginal(mu: float = 0.0, sigma: float = 1.0) -> MarginalQuantile:
    """Return the quantile function of a normal distribution.

    :param mu: mean of the distribution
    :param sigma: standard deviation, must be strictly positive
    """
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive")
    return NormalDist(mu=mu, sigma=sigma).inv_cdf


def lognormal_marginal(mu: float = 0.0, sigma: float = 1.0) -> MarginalQuantile:
    """Return the quantile function of a lognormal distribution.

    The returned marginal produces values ``exp(X)`` where ``X`` follows a
    normal distribution with mean ``mu`` and standard deviation ``sigma``.

    :param mu: mean of the underlying normal distribution
    :param sigma: standard deviation of the underlying normal distribution,
        must be strictly positive
    """
    normal_quantile = normal_marginal(mu=mu, sigma=sigma)

    def quantile(u: float) -> float:
        return math.exp(normal_quantile(u))

    return quantile


def discrete_marginal(
    values: Sequence[float],
    probabilities: Optional[Sequence[float]] = None,
) -> MarginalQuantile:
    """Return the quantile function of a finite discrete distribution.

    ``values`` are the possible outcomes; they are sorted so that the
    resulting quantile function is monotone. ``probabilities`` defaults to a
    uniform distribution and is normalized if it does not sum to one.

    Note that discretization attenuates dependence: when this marginal is
    used with :meth:`Provider.gaussian_copula`, neither the Pearson nor the
    Spearman correlation of the generated column equals the corresponding
    entry of the latent correlation matrix. The latent correlation only
    drives the joint uniform ranks before discretization.

    :param values: non-empty sequence of possible outcomes
    :param probabilities: optional non-negative weights, one per value
    """
    outcomes = sorted(values)
    if not outcomes:
        raise ValueError("values must not be empty")
    if probabilities is None:
        weights = [1.0] * len(outcomes)
    else:
        if len(probabilities) != len(outcomes):
            raise ValueError("probabilities must have the same length as values")
        paired = sorted(zip(values, (float(p) for p in probabilities)), key=lambda pair: pair[0])
        weights = [weight for _, weight in paired]
        if any(weight < 0 for weight in weights):
            raise ValueError("probabilities must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("probabilities must have a positive sum")
    cumulative = []
    running = 0.0
    for weight in weights:
        running += weight / total
        cumulative.append(running)
    cumulative[-1] = 1.0

    def quantile(u: float) -> float:
        index = bisect_left(cumulative, u)
        return outcomes[min(index, len(outcomes) - 1)]

    return quantile


def _validate_correlation_matrix(correlation: Sequence[Sequence[float]]) -> List[List[float]]:
    try:
        matrix = [[float(entry) for entry in row] for row in correlation]
    except (TypeError, ValueError) as exc:
        raise ValueError("correlation must be a square matrix of real numbers") from exc
    if not matrix:
        raise ValueError("correlation matrix must not be empty")
    dimension = len(matrix)
    if dimension > MAX_COPULA_DIMENSION:
        raise ValueError(f"correlation matrix dimension must not exceed {MAX_COPULA_DIMENSION}")
    if any(len(row) != dimension for row in matrix):
        raise ValueError("correlation matrix must be square")
    for i in range(dimension):
        if not math.isclose(matrix[i][i], 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("correlation matrix diagonal entries must all be 1")
        for j in range(i + 1, dimension):
            if not math.isclose(matrix[i][j], matrix[j][i], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("correlation matrix must be symmetric")
    return matrix


def _cholesky_psd(matrix: List[List[float]]) -> List[List[float]]:
    """Factor a symmetric PSD matrix as ``L @ L.T`` via pivoted Cholesky.

    Valid singular (rank-deficient) matrices are supported: once the remaining
    residual diagonal is numerically zero, the leftover columns of ``L`` are
    left as zeros. An indefinite matrix raises :class:`ValueError`; no jitter
    or noise is ever added to make an invalid input factorizable.
    """
    n = len(matrix)
    factor = [[0.0] * n for _ in range(n)]
    order = list(range(n))
    tolerance = 1e-12 * max(1.0, max(abs(matrix[i][i]) for i in range(n)))
    for k in range(n):
        pivot, pivot_value = -1, 0.0
        for i in range(k, n):
            row = order[i]
            residual = matrix[row][row] - sum(factor[row][j] ** 2 for j in range(k))
            if pivot == -1 or residual > pivot_value:
                pivot, pivot_value = i, residual
        if pivot_value < -tolerance:
            raise ValueError("correlation matrix is not positive semidefinite")
        if pivot_value <= tolerance:
            break
        order[k], order[pivot] = order[pivot], order[k]
        pivot_row = order[k]
        factor[pivot_row][k] = math.sqrt(pivot_value)
        for i in range(k + 1, n):
            row = order[i]
            cross = sum(factor[row][j] * factor[pivot_row][j] for j in range(k))
            factor[row][k] = (matrix[row][pivot_row] - cross) / factor[pivot_row][k]
    return factor


class Provider(BaseProvider):
    """Generate small vectors of correlated numbers with a Gaussian copula."""

    max_copula_dimension = MAX_COPULA_DIMENSION

    def gaussian_copula(
        self,
        correlation: Sequence[Sequence[float]],
        marginals: Sequence[MarginalQuantile],
    ) -> List[float]:
        """Generate one sample from a Gaussian copula with the given marginals.

        Gaussian copula semantics: a latent vector ``Z ~ N(0, R)`` is drawn
        from a zero-mean multivariate normal whose covariance is the given
        correlation matrix ``R``. Each component is mapped through the
        standard normal CDF to a uniform ``U_i = Phi(Z_i)``, and each uniform
        is mapped through the caller-supplied marginal quantile (inverse CDF)
        function ``X_i = Q_i(U_i)``. Use :func:`normal_marginal`,
        :func:`lognormal_marginal`, and :func:`discrete_marginal` to build
        quantile functions, or pass any callable accepting a probability in
        ``(0, 1)``.

        The correlation matrix must be symmetric, have unit diagonal entries,
        and be positive semidefinite. Valid singular matrices (e.g. perfectly
        correlated columns) are accepted as-is; invalid matrices raise
        :class:`ValueError` instead of being silently perturbed. The matrix
        dimension is capped at ``max_copula_dimension``.

        Because the marginal transforms are monotone, rank-based dependence
        (Spearman) of continuous marginals is a known function of the latent
        correlation. Discrete marginals are an exception: discretization ties
        and attenuates ranks, so the realized correlation of a discrete
        column no longer equals the input coefficient.

        All randomness is drawn from this Faker instance's random generator,
        so seeding the instance (``Faker.seed`` / ``seed_instance``) fully
        determines the generated samples.

        :param correlation: symmetric, unit-diagonal, positive semidefinite
            matrix of latent correlations
        :param marginals: one quantile (inverse CDF) callable per column
        :return: a list with one generated value per column
        """
        matrix = _validate_correlation_matrix(correlation)
        quantiles = list(marginals)
        if len(quantiles) != len(matrix):
            raise ValueError("marginals must provide one quantile function per column")
        if any(not callable(quantile) for quantile in quantiles):
            raise ValueError("each marginal must be a callable quantile function")
        factor = _cholesky_psd(matrix)
        random = self.generator.random
        dimension = len(matrix)
        z = [random.gauss(0.0, 1.0) for _ in range(dimension)]
        sample = []
        for i in range(dimension):
            y = sum(factor[i][j] * z[j] for j in range(dimension))
            u = min(max(_STANDARD_NORMAL.cdf(y), _U_MIN), _U_MAX)
            sample.append(quantiles[i](u))
        return sample
