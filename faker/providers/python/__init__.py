import bisect
import math
import string
import sys
import warnings

from decimal import Decimal
from enum import Enum
from statistics import NormalDist
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

from faker.typing import BasicNumber

from ...exceptions import BaseFakerException
from .. import BaseProvider, ElementsType

TypesNames = List[str]
TypesSpec = Union[List[Type], Tuple[Type, ...]]
TEnum = TypeVar("TEnum", bound=Enum)


class EmptyEnumException(BaseFakerException):
    pass


def normal_quantile(loc: float = 0.0, scale: float = 1.0) -> Callable[[float], float]:
    """
    Returns the quantile (inverse CDF) function of a normal distribution.

    Suitable as a marginal for :meth:`Provider.correlated_sample`.

    :param loc: mean of the distribution.
    :param scale: standard deviation of the distribution, must be positive.
    """
    if scale <= 0:
        raise ValueError("`scale` must be positive")
    dist = NormalDist(mu=loc, sigma=scale)
    return dist.inv_cdf


def lognormal_quantile(mu: float = 0.0, sigma: float = 1.0) -> Callable[[float], float]:
    """
    Returns the quantile (inverse CDF) function of a lognormal distribution
    whose underlying normal has mean `mu` and standard deviation `sigma`.

    Suitable as a marginal for :meth:`Provider.correlated_sample`.

    :param mu: mean of the underlying normal distribution.
    :param sigma: standard deviation of the underlying normal distribution, must be positive.
    """
    if sigma <= 0:
        raise ValueError("`sigma` must be positive")
    dist = NormalDist(mu=mu, sigma=sigma)

    def quantile(u: float) -> float:
        return math.exp(dist.inv_cdf(u))

    return quantile


def discrete_quantile(values: Sequence[Any], probabilities: Optional[Sequence[float]] = None) -> Callable[[float], Any]:
    """
    Returns the quantile (inverse CDF) function of a distribution over a
    finite set of discrete `values`.

    Suitable as a marginal for :meth:`Provider.correlated_sample`. Note that
    discretization attenuates correlation: the (Pearson or rank) correlation
    of the generated discrete values is generally weaker than the correlation
    coefficients handed to the Gaussian copula.

    :param values: the support of the distribution, must be non-empty.
    :param probabilities: optional probability (weight) per value, uniform if omitted.
    """
    if not values:
        raise ValueError("`values` must be a non-empty sequence")
    if probabilities is None:
        probabilities = [1.0] * len(values)
    if len(probabilities) != len(values):
        raise ValueError("`probabilities` must have the same length as `values`")
    if any(p < 0 for p in probabilities):
        raise ValueError("`probabilities` must be non-negative")
    total = float(sum(probabilities))
    if total <= 0:
        raise ValueError("`probabilities` must have a positive sum")
    cumulative: List[float] = []
    running = 0.0
    for p in probabilities:
        running += p / total
        cumulative.append(running)
    cumulative[-1] = 1.0
    support = list(values)

    def quantile(u: float) -> Any:
        index = bisect.bisect_left(cumulative, u)
        return support[min(index, len(support) - 1)]

    return quantile


class Provider(BaseProvider):
    default_value_types: ElementsType[str] = (
        "str",
        "str",
        "str",
        "str",
        "float",
        "int",
        "int",
        "decimal",
        "date_time",
        "uri",
        "email",
    )

    def _check_signature(self, value_types: Optional[TypesSpec], allowed_types: Optional[TypesSpec]) -> TypesSpec:
        if value_types is not None and not isinstance(value_types, (list, tuple)):
            value_types = (value_types,)
            warnings.warn(
                "Passing `value_types` as positional arguments is going to be "
                "deprecated.  Pass them as a list or tuple instead.",
                PendingDeprecationWarning,
            )
        if allowed_types is not None and not isinstance(allowed_types, (list, tuple)):
            allowed_types = (allowed_types,)
            warnings.warn(
                "Passing `allowed_types` as positional arguments is going to be "
                "deprecated.  Pass them as a list or tuple instead.",
                PendingDeprecationWarning,
            )
        if value_types is None:
            value_types = ()
        if allowed_types is None:
            allowed_types = ()
        return tuple(value_types) + tuple(allowed_types)

    def pyobject(
        self,
        object_type: Optional[Type[Union[bool, str, float, int, tuple, set, list, Iterable, dict]]] = None,
    ) -> Optional[Union[bool, str, float, int, tuple, set, list, Iterable, dict]]:
        """
        Generates a random object passing the type desired.

        :object_type: the type of the object to generate.
        :return: the random object generated.
        :raises ValueError: if the object type passed is not supported
        """
        if object_type is None:
            return None
        elif object_type == bool:
            return self.pybool()
        elif object_type == str:
            return self.pystr()
        elif object_type == float:
            return self.pyfloat()
        elif object_type == int:
            return self.pyint()
        elif object_type == tuple:
            return self.pytuple()
        elif object_type == set:
            return self.pyset()
        elif object_type == list:
            return self.pylist()
        elif object_type == Iterable:
            return self.pyiterable()
        elif object_type == dict:
            return self.pydict()
        else:
            raise ValueError(f"Object type `{object_type}` is not supported by `pyobject` function")

    def pybool(self, truth_probability: int = 50) -> bool:
        """
        Generates a random boolean, optionally biased towards `True` or `False`.

        :truth_probability: Probability of generating a `True` value. Must be between `0` and `100` inclusive'.
        :return: Random boolean.
        :raises ValueError: If invalid `truth_probability` is provided.
        """
        if truth_probability < 0 or truth_probability > 100:
            raise ValueError("Invalid `truth_probability` value: must be between `0` and `100` inclusive")

        return self.random_int(1, 100) <= truth_probability

    def pystr(
        self,
        min_chars: Optional[int] = None,
        max_chars: int = 20,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """
        Generates a random string of upper and lowercase letters.

        :param min_chars: minimum length of the random part.
        :param max_chars: maximum length of the random part.
        :param prefix: an optional prefix to prepend to the random string.
        :param suffix: an optional suffix to append to the random string.
        :return: Random of random length between min and max characters.
        """
        if min_chars is None:
            chars = "".join(self.random_letters(length=max_chars))
        else:
            assert max_chars >= min_chars, "Maximum length must be greater than or equal to minimum length"
            chars = "".join(
                self.random_letters(
                    length=self.generator.random.randint(min_chars, max_chars),
                ),
            )

        return prefix + chars + suffix

    def pystr_format(
        self,
        string_format: str = "?#-###{{random_int}}{{random_letter}}",
        letters: str = string.ascii_letters,
    ) -> str:
        return self.bothify(self.generator.parse(string_format), letters=letters)

    @no_type_check
    def pyfloat(
        self,
        left_digits: Optional[int] = None,
        right_digits: Optional[int] = None,
        positive: Optional[bool] = None,
        min_value: Optional[Union[float, int]] = None,
        max_value: Optional[Union[float, int]] = None,
    ) -> float:
        if left_digits is not None and left_digits < 0:
            raise ValueError("A float number cannot have less than 0 digits in its integer part")
        if right_digits is not None and right_digits < 0:
            raise ValueError("A float number cannot have less than 0 digits in its fractional part")
        if left_digits == 0 and right_digits == 0:
            raise ValueError("A float number cannot have less than 0 digits in total")
        if min_value is not None and max_value is not None:
            if min_value > max_value:
                raise ValueError("Min value cannot be greater than max value")
        if None not in (min_value, max_value) and min_value == max_value:
            raise ValueError("Min and max value cannot be the same")
        if positive and min_value is not None and min_value <= 0:
            raise ValueError("Cannot combine positive=True with negative or zero min_value")
        if left_digits is not None and max_value and math.ceil(math.log10(abs(max_value))) > left_digits:
            raise ValueError("Max value must fit within left digits")
        if left_digits is not None and min_value and math.ceil(math.log10(abs(min_value))) > left_digits:
            raise ValueError("Min value must fit within left digits")

        # Make sure at least either left or right is set
        if left_digits is None and right_digits is None:
            needed_left_digits = max(1, math.ceil(math.log10(max(abs(max_value or 1), abs(min_value or 1)))))
            right_digits = self.random_int(1, sys.float_info.dig - needed_left_digits)

        # If only one side is set, choose #digits for other side
        if (left_digits is None) ^ (right_digits is None):
            if left_digits is None:
                left_digits = max(1, sys.float_info.dig - right_digits)
            else:
                right_digits = max(1, sys.float_info.dig - left_digits)

        # Make sure we don't ask for too many digits!
        if left_digits + right_digits > sys.float_info.dig:
            raise ValueError(
                f"Asking for too many digits ({left_digits} + {right_digits} == {left_digits + right_digits} > "
                f"{sys.float_info.dig})",
            )

        sign = ""
        if (min_value is not None) or (max_value is not None):
            # Copy values to ensure we're not modifying the original values and thus going out of bounds
            left_min_value = min_value
            left_max_value = max_value
            # Make sure left_digits still respected
            if left_digits is not None:
                if max_value is None:
                    left_max_value = 10**left_digits  # minus smallest representable, adjusted later
                if min_value is None:
                    left_min_value = -(10**left_digits)  # plus smallest representable, adjusted later

            if max_value is not None and max_value < 0:
                left_max_value += 1  # as the random_int will be generated up to max_value - 1
            if min_value is not None and min_value < 0:
                left_min_value += 1  # as we then append digits after the left_number
            left_number = self._safe_random_int(
                left_min_value,
                left_max_value,
                positive,
            )
        else:
            if positive is None:
                sign = self.random_element(("+", "-"))
            elif positive is True:
                sign = "+"
            else:
                sign = "-"

            left_number = self.random_number(left_digits)

        result = float(f"{sign}{left_number}.{self.random_number(right_digits)}")
        if positive and result == 0:
            if right_digits:
                result = float("0." + "0" * (right_digits - 1) + "1")
            else:
                result += sys.float_info.epsilon

        if right_digits:
            result = min(result, 10**left_digits - float(f'0.{"0" * (right_digits - 1)}1'))
            result = max(result, -(10**left_digits + float(f'0.{"0" * (right_digits - 1)}1')))
        else:
            result = min(result, 10**left_digits - 1)
            result = max(result, -(10**left_digits + 1))

        # It's possible for the result to end up > than max_value or < than min_value
        # When this happens we introduce some variance so we're not always exactly at the boundary.
        # When only one bound is set, use the exceeded amount as the variance bound.
        # When positive=True without explicit min_value, treat min_value as 0.
        effective_min = 0 if (positive and min_value is None) else min_value
        if max_value is not None and result > max_value:
            exceeded_by = result - max_value
            if effective_min is not None:
                variance_bound = max_value - effective_min
            else:
                variance_bound = exceeded_by
            result = result - (exceeded_by + self.generator.random.uniform(0, variance_bound))
        if effective_min is not None and result < effective_min:
            below_by = effective_min - result
            if max_value is not None:
                variance_bound = max_value - effective_min
            else:
                variance_bound = below_by
            result = result + (below_by + self.generator.random.uniform(0, variance_bound))

        return result

    def _safe_random_int(self, min_value: float, max_value: float, positive: bool) -> int:
        orig_min_value = min_value
        orig_max_value = max_value

        if min_value is None:
            min_value = max_value - self.random_int()
        if max_value is None:
            max_value = min_value + self.random_int()
        if positive:
            min_value = max(min_value, 0)

        if min_value == max_value:
            # Only retry if at least one bound was randomly chosen (None originally);
            # when both bounds were explicitly supplied and still collapse to the same
            # integer after adjustment, retrying would recurse infinitely with the same
            # inputs.  Return the collapsed value directly in that case.
            if orig_min_value is None or orig_max_value is None:
                return self._safe_random_int(orig_min_value, orig_max_value, positive)
            return int(min_value)
        else:
            min_value = int(min_value)
            max_value = int(max_value - 1)
            if max_value < min_value:
                max_value += 1
            return self.random_int(min_value, max_value)

    def pyint(self, min_value: int = 0, max_value: int = 9999, step: int = 1) -> int:
        return self.generator.random_int(min_value, max_value, step=step)

    def _random_int_of_length(self, length: int) -> int:
        """Generate a random integer of a given length

        If length is 0, so is the number. Otherwise the first digit must not be 0.
        """

        if length < 0:
            raise ValueError("Length must be a non-negative integer.")
        elif length == 0:
            return 0
        else:
            min_value = 10 ** (length - 1)
            max_value = (10**length) - 1
            return self.pyint(min_value=min_value, max_value=max_value)

    def pydecimal(
        self,
        left_digits: Optional[int] = None,
        right_digits: Optional[int] = None,
        positive: Optional[bool] = None,
        min_value: Optional[BasicNumber] = None,
        max_value: Optional[BasicNumber] = None,
    ) -> Decimal:
        if left_digits is not None and left_digits < 0:
            raise ValueError("A decimal number cannot have less than 0 digits in its integer part")
        if right_digits is not None and right_digits < 0:
            raise ValueError("A decimal number cannot have less than 0 digits in its fractional part")
        if (left_digits is not None and left_digits == 0) and (right_digits is not None and right_digits == 0):
            raise ValueError("A decimal number cannot have 0 digits in total")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError("Min value cannot be greater than max value")
        if min_value is not None and max_value is not None and min_value == max_value:
            raise ValueError("Min and max value cannot be the same")
        if positive and min_value is not None and min_value <= 0:
            raise ValueError("Cannot combine positive=True with negative or zero min_value")
        if positive and max_value is not None and max_value <= 0:
            raise ValueError("Cannot combine positive=True with negative or zero max_value")
        if (
            left_digits is not None
            and max_value
            and math.ceil(math.log10(abs(max_value))) > left_digits  # type: ignore[arg-type]
        ):
            raise ValueError("Max value must fit within left digits")
        if (
            left_digits is not None
            and min_value
            and math.ceil(math.log10(abs(min_value))) > left_digits  # type: ignore[arg-type]
        ):
            raise ValueError("Min value must fit within left digits")

        # if either left or right digits are not specified we randomly choose a length
        max_random_digits = 100
        # Because if min_value is bigger than 10**100
        max_digits_from_value = max(
            math.ceil(math.log10(abs(min_value or 1))),  # type: ignore[arg-type]
            math.ceil(math.log10(abs(max_value or 1))),  # type: ignore[arg-type]
        )
        max_left_random_digits = max(max_random_digits, max_digits_from_value + 10)

        if min_value is not None and min_value >= 0:
            sign = "+"
        elif max_value is not None and max_value <= 0:
            sign = "-"
        else:
            if positive is None:
                sign = self.random_element(("+", "-"))
            else:
                sign = "+" if positive else "-"

        if sign == "+":
            if max_value is not None:
                left_number = str(self.random_int(int(max(min_value or 0, 0)), int(max_value)))
            else:
                min_left_digits = math.ceil(math.log10(max(min_value or 1, 1)))
                if left_digits is None:
                    left_digits = self.random_int(min_left_digits, max_left_random_digits)
                left_number = str(self._random_int_of_length(left_digits))
        else:
            if min_value is not None:
                left_number = str(
                    self.random_int(
                        int(abs(min(max_value or 0, 0))),  # type: ignore[arg-type,call-overload]
                        int(abs(min_value)),  # type: ignore[arg-type,call-overload]
                    )
                )
            else:
                min_left_digits = math.ceil(math.log10(abs(min(max_value or 1, 1))))  # type: ignore[arg-type]
                if left_digits is None:
                    left_digits = self.random_int(min_left_digits, max_left_random_digits)
                left_number = str(self._random_int_of_length(left_digits))

        if right_digits is None:
            right_digits = self.random_int(0, max_random_digits)

        right_number = "".join([str(self.random_digit()) for _ in range(0, right_digits)])

        result = Decimal(f"{sign}{left_number}.{right_number}")

        # Because the random result might have the same number of decimals as max_value the random number
        # might be above max_value or below min_value
        if max_value is not None and result > max_value:
            result = Decimal(str(max_value))
        if min_value is not None and result < min_value:
            result = Decimal(str(min_value))

        return result

    def pytuple(
        self,
        nb_elements: int = 10,
        variable_nb_elements: bool = True,
        value_types: Optional[TypesSpec] = None,
        allowed_types: Optional[TypesSpec] = None,
    ) -> Tuple[Any, ...]:
        return tuple(
            self._pyiterable(
                nb_elements=nb_elements,
                variable_nb_elements=variable_nb_elements,
                value_types=value_types,
                allowed_types=allowed_types,
            )
        )

    def pyset(
        self,
        nb_elements: int = 10,
        variable_nb_elements: bool = True,
        value_types: Optional[TypesSpec] = None,
        allowed_types: Optional[TypesSpec] = None,
    ) -> Set[Any]:
        return set(
            self._pyiterable(
                nb_elements=nb_elements,
                variable_nb_elements=variable_nb_elements,
                value_types=value_types,
                allowed_types=allowed_types,
            )
        )

    def pylist(
        self,
        nb_elements: int = 10,
        variable_nb_elements: bool = True,
        value_types: Optional[TypesSpec] = None,
        allowed_types: Optional[TypesSpec] = None,
    ) -> List[Any]:
        return list(
            self._pyiterable(
                nb_elements=nb_elements,
                variable_nb_elements=variable_nb_elements,
                value_types=value_types,
                allowed_types=allowed_types,
            )
        )

    @no_type_check
    def pyiterable(
        self,
        nb_elements: int = 10,
        variable_nb_elements: bool = True,
        value_types: Optional[TypesSpec] = None,
        allowed_types: Optional[TypesSpec] = None,
    ) -> Iterable[Any]:
        _value_types: TypesSpec = self._check_signature(value_types, allowed_types)
        return self.random_element([self.pylist, self.pytuple, self.pyset])(
            nb_elements=nb_elements,
            variable_nb_elements=variable_nb_elements,
            value_types=_value_types,
            allowed_types=allowed_types,
        )

    def _random_type(self, type_list: List[str]) -> str:
        value_type: str = self.random_element(type_list)

        method_name = f"py{value_type}"
        if hasattr(self, method_name):
            value_type = method_name

        return self.generator.format(value_type)

    def _pyiterable(
        self,
        nb_elements: int = 10,
        variable_nb_elements: bool = True,
        value_types: Optional[TypesSpec] = None,
        allowed_types: Optional[TypesSpec] = None,
    ) -> Iterator:
        _value_types: TypesSpec = self._check_signature(value_types, allowed_types)

        _value_types: TypesNames = [
            t if isinstance(t, str) else getattr(t, "__name__", type(t).__name__).lower()
            for t in _value_types
            # avoid recursion
            if t not in ["iterable", "list", "tuple", "dict", "set"]
        ]
        if not _value_types:
            _value_types = self.default_value_types  # type: ignore

        if variable_nb_elements:
            nb_elements = self.randomize_nb_elements(nb_elements, min=1)

        for _ in range(nb_elements):
            yield self._random_type(_value_types)

    def pydict(
        self,
        nb_elements: int = 10,
        variable_nb_elements: bool = True,
        value_types: Optional[TypesSpec] = None,
        allowed_types: Optional[TypesSpec] = None,
    ) -> Dict[Any, Any]:
        """
        Returns a dictionary.

        :nb_elements: number of elements for dictionary
        :variable_nb_elements: is use variable number of elements for dictionary
        :value_types: type of dictionary values
        """

        words_list_count = len(self.generator.get_words_list())

        if variable_nb_elements:
            nb_elements = self.randomize_nb_elements(nb_elements, min=1)

        if nb_elements > words_list_count:
            warnings.warn(
                f"Number of nb_elements is greater than the number of words in the list."
                f" {words_list_count} words will be used.",
                RuntimeWarning,
            )
            nb_elements = words_list_count

        return dict(
            zip(
                self.generator.words(nb_elements, unique=True),
                self._pyiterable(
                    nb_elements=nb_elements,
                    variable_nb_elements=False,
                    value_types=value_types,
                    allowed_types=allowed_types,
                ),
            )
        )

    def pystruct(
        self,
        count: int = 10,
        value_types: Optional[TypesSpec] = None,
        allowed_types: Optional[TypesSpec] = None,
    ) -> Tuple[List, Dict, Dict]:
        _value_types: TypesSpec = self._check_signature(value_types, allowed_types)

        _value_types: TypesNames = [
            t if isinstance(t, str) else getattr(t, "__name__", type(t).__name__).lower()
            for t in _value_types
            # avoid recursion
            if t != "struct"
        ]
        if not _value_types:
            _value_types = self.default_value_types  # type: ignore

        types = []
        d = {}
        nd = {}
        for i in range(count):
            d[self.generator.word()] = self._random_type(_value_types)
            types.append(self._random_type(_value_types))
            nd[self.generator.word()] = {
                i: self._random_type(_value_types),
                i
                + 1: [
                    self._random_type(_value_types),
                    self._random_type(_value_types),
                    self._random_type(_value_types),
                ],
                i
                + 2: {
                    i: self._random_type(_value_types),
                    i + 1: self._random_type(_value_types),
                    i
                    + 2: [
                        self._random_type(_value_types),
                        self._random_type(_value_types),
                    ],
                },
            }
        return types, d, nd

    def enum(self, enum_cls: Type[TEnum]) -> TEnum:
        """
        Returns a random enum of the provided input `Enum` type.

        :param enum_cls: The `Enum` type to produce the value for.
        :returns: A randomly selected enum value.
        """

        if enum_cls is None:
            raise ValueError("'enum_cls' cannot be None")

        if not issubclass(enum_cls, Enum):
            raise TypeError("'enum_cls' must be an Enum type")

        members: List[TEnum] = list(cast(Iterable[TEnum], enum_cls))

        if len(members) < 1:
            raise EmptyEnumException(f"The provided Enum: '{enum_cls.__name__}' has no members.")

        return self.random_element(members)

    max_correlated_dimensions: int = 16

    def correlated_sample(
        self,
        correlation_matrix: Sequence[Sequence[float]],
        quantile_functions: Sequence[Callable[[float], Any]],
    ) -> Tuple[Any, ...]:
        """
        Generates one vector of correlated values using a Gaussian copula.

        Semantics: a vector ``Z ~ N(0, R)`` is drawn from the zero-mean
        multivariate normal distribution whose covariance matrix is the given
        correlation matrix ``R``. Each component is mapped to a uniform
        variate ``U_i = Phi(Z_i)`` via the standard normal CDF, and the final
        values are ``X_i = Q_i(U_i)`` where ``Q_i`` is the quantile (inverse
        CDF) function of the i-th marginal. Ready-made marginals are available
        as :func:`normal_quantile`, :func:`lognormal_quantile` and
        :func:`discrete_quantile`, but any callable mapping ``(0, 1)`` to a
        value may be used.

        The correlation matrix must be symmetric, have ones on the diagonal
        and be positive semidefinite. Singular (yet valid) matrices, e.g. for
        perfectly correlated columns, are supported. Invalid matrices are
        rejected with a ``ValueError`` instead of being silently repaired.

        Note that the correlation of the generated values equals the input
        coefficients only in special cases (e.g. normal marginals). Non-linear
        or discrete marginals attenuate correlations, so after discretization
        the output correlation is generally not equal to the input
        coefficient. This is inherent to the copula transform, not a bug.

        All randomness is drawn from this Faker instance's random generator,
        so seeding the instance makes samples reproducible.

        :param correlation_matrix: symmetric, positive semidefinite matrix
            with unit diagonal; at most ``max_correlated_dimensions`` (16)
            columns.
        :param quantile_functions: one quantile (inverse CDF) function per
            column of the correlation matrix.
        :returns: a tuple with one generated value per column.
        :raises ValueError: if the matrix is malformed or not a valid
            correlation matrix, or the number of marginals does not match.
        """
        matrix, eigenvalues, eigenvectors = self._validate_correlation_matrix(correlation_matrix)
        dimension = len(matrix)
        if len(quantile_functions) != dimension:
            raise ValueError(
                f"`quantile_functions` must provide one quantile function per column "
                f"({dimension} expected, got {len(quantile_functions)})"
            )
        for quantile_function in quantile_functions:
            if not callable(quantile_function):
                raise ValueError("every entry of `quantile_functions` must be callable")

        random = self.generator.random
        standard_normals = [random.gauss(0.0, 1.0) for _ in range(dimension)]
        sqrt_two = math.sqrt(2.0)
        sample = []
        for i in range(dimension):
            z_i = 0.0
            row = eigenvectors[i]
            for k in range(dimension):
                if eigenvalues[k] > 0.0:
                    z_i += row[k] * math.sqrt(eigenvalues[k]) * standard_normals[k]
            u_i = 0.5 * (1.0 + math.erf(z_i / sqrt_two))
            sample.append(quantile_functions[i](u_i))
        return tuple(sample)

    def _validate_correlation_matrix(
        self, correlation_matrix: Sequence[Sequence[float]]
    ) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        tolerance = 1e-9
        try:
            matrix = [[float(entry) for entry in row] for row in correlation_matrix]
        except (TypeError, ValueError):
            raise ValueError("`correlation_matrix` must be a square matrix of real numbers") from None
        dimension = len(matrix)
        if dimension == 0:
            raise ValueError("`correlation_matrix` must not be empty")
        if dimension > self.max_correlated_dimensions:
            raise ValueError(
                f"`correlation_matrix` dimension {dimension} exceeds the limit of {self.max_correlated_dimensions}"
            )
        if any(len(row) != dimension for row in matrix):
            raise ValueError("`correlation_matrix` must be square")
        for i in range(dimension):
            if abs(matrix[i][i] - 1.0) > tolerance:
                raise ValueError("`correlation_matrix` must have ones on the diagonal")
            for j in range(i + 1, dimension):
                if abs(matrix[i][j] - matrix[j][i]) > tolerance:
                    raise ValueError("`correlation_matrix` must be symmetric")
        eigenvalues, eigenvectors = self._jacobi_eigen_decomposition(matrix)
        if min(eigenvalues) < -tolerance:
            raise ValueError("`correlation_matrix` must be positive semidefinite")
        return matrix, eigenvalues, eigenvectors

    @staticmethod
    def _jacobi_eigen_decomposition(matrix: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """
        Computes the eigen decomposition of a small symmetric matrix with the
        cyclic Jacobi method (standard library only).

        :returns: a tuple ``(eigenvalues, eigenvectors)`` where column ``k``
            of ``eigenvectors`` is the eigenvector for ``eigenvalues[k]``.
        """
        dimension = len(matrix)
        a = [row[:] for row in matrix]
        vectors = [[1.0 if i == j else 0.0 for j in range(dimension)] for i in range(dimension)]
        for _ in range(50):
            off_diagonal_norm = math.sqrt(
                sum(a[i][j] ** 2 for i in range(dimension) for j in range(dimension) if i != j)
            )
            if off_diagonal_norm <= 1e-12:
                break
            for p in range(dimension - 1):
                for q in range(p + 1, dimension):
                    apq = a[p][q]
                    if apq == 0.0:
                        continue
                    theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                    sign = 1.0 if theta >= 0.0 else -1.0
                    t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                    c = 1.0 / math.sqrt(t * t + 1.0)
                    s = t * c
                    for k in range(dimension):
                        akp, akq = a[k][p], a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    for k in range(dimension):
                        apk, aqk = a[p][k], a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
                    for k in range(dimension):
                        vkp, vkq = vectors[k][p], vectors[k][q]
                        vectors[k][p] = c * vkp - s * vkq
                        vectors[k][q] = s * vkp + c * vkq
        eigenvalues = [a[i][i] for i in range(dimension)]
        return eigenvalues, vectors
