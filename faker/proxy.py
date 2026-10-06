from __future__ import annotations

import copy
import functools
import re

from collections.abc import Iterable, Mapping
from collections import OrderedDict
from random import Random
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Pattern, Sequence, TypeVar

from .config import DEFAULT_LOCALE
from .exceptions import UniquenessException
from .factory import Factory
from .generator import Generator
from .typing import SeedType
from .utils.distribution import choices_distribution

if TYPE_CHECKING:
    from .providers import BaseProvider

_UNIQUE_ATTEMPTS = 1000

RetType = TypeVar("RetType")


class Faker:
    """Proxy class capable of supporting multiple locales"""

    cache_pattern: Pattern = re.compile(r"^_cached_\w*_mapping$")
    generator_attrs = [
        attr for attr in dir(Generator) if not attr.startswith("__") and attr not in ["seed", "seed_instance", "random"]
    ]

    def __init__(
        self,
        locale: str | Sequence[str] | dict[str, int | float] | None = None,
        providers: list[str] | None = None,
        generator: Generator | None = None,
        includes: list[str] | None = None,
        use_weighting: bool = True,
        **config: Any,
    ) -> None:
        self._factory_map: OrderedDict[str, Generator | Faker] = OrderedDict()
        self._weights = None
        self._unique_proxy = UniqueProxy(self)
        self._optional_proxy = OptionalProxy(self)
        self._record_scopes: list[_RecordScope] = []

        if isinstance(locale, str):
            locales = [locale.replace("-", "_")]

        # This guarantees a FIFO ordering of elements in `locales` based on the final
        # locale string while discarding duplicates after processing
        elif isinstance(locale, (list, tuple, set)):
            locales = []
            for code in locale:
                if not isinstance(code, str):
                    raise TypeError(f'The locale "{str(code)}" must be a string.')
                final_locale = code.replace("-", "_")
                if final_locale not in locales:
                    locales.append(final_locale)

        elif isinstance(locale, (OrderedDict, dict)):
            assert all(isinstance(v, (int, float)) for v in locale.values())
            odict = OrderedDict()
            for k, v in locale.items():
                key = k.replace("-", "_")
                odict[key] = v
            locales = list(odict.keys())
            self._weights = list(odict.values())

        else:
            locales = [DEFAULT_LOCALE]

        if len(locales) == 1:
            self._factory_map[locales[0]] = Factory.create(
                locales[0],
                providers,
                generator,
                includes,
                use_weighting=use_weighting,
                **config,
            )
        else:
            for locale in locales:
                self._factory_map[locale] = Faker(
                    locale,
                    providers,
                    generator,
                    includes,
                    use_weighting=use_weighting,
                    **config,
                )

        self._locales = locales
        self._factories = list(self._factory_map.values())

    def __dir__(self):
        attributes = set(super().__dir__())
        for factory in self.factories:
            attributes |= {attr for attr in dir(factory) if not attr.startswith("_")}
        return sorted(attributes)

    def __getitem__(self, locale: str) -> Faker:
        if locale.replace("-", "_") in self.locales and len(self.locales) == 1:
            return self
        instance = self._factory_map[locale.replace("-", "_")]
        assert isinstance(instance, Faker)  # for mypy
        return instance

    def __getattribute__(self, attr: str) -> Any:
        """
        Handles the "attribute resolution" behavior for declared members of this proxy class

        The class method `seed` cannot be called from an instance.

        :param attr: attribute name
        :return: the appropriate attribute
        """
        if attr == "seed":
            msg = "Calling `.seed()` on instances is deprecated. Use the class method `Faker.seed()` instead."
            raise TypeError(msg)
        else:
            return super().__getattribute__(attr)

    def __getattr__(self, attr: str) -> Any:
        """
        Handles cache access and proxying behavior

        :param attr: attribute name
        :return: the appropriate attribute
        """
        try:
            record_scopes = object.__getattribute__(self, "_record_scopes")
        except AttributeError:
            record_scopes = []

        if record_scopes:
            return self._get_record_attribute(record_scopes[-1], attr)

        if len(self._factories) == 1:
            return getattr(self._factories[0], attr)
        elif attr in self.generator_attrs:
            msg = "Proxying calls to `%s` is not implemented in multiple locale mode." % attr
            raise NotImplementedError(msg)
        elif self.cache_pattern.match(attr):
            msg = "Cached attribute `%s` does not exist" % attr
            raise AttributeError(msg)
        else:
            factory = self._select_factory(attr)
            return getattr(factory, attr)

    def __deepcopy__(self, memodict):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        result._locales = copy.deepcopy(self._locales, memodict)
        result._factory_map = copy.deepcopy(self._factory_map, memodict)
        result._factories = list(result._factory_map.values())
        result._weights = copy.deepcopy(self._weights, memodict)
        result._record_scopes = []
        result._unique_proxy = UniqueProxy(result)
        result._unique_proxy._seen = {k: {result._unique_proxy._sentinel} for k in self._unique_proxy._seen.keys()}
        result._optional_proxy = OptionalProxy(result)
        return result

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state)
        self._record_scopes = getattr(self, "_record_scopes", [])

    def record(
        self,
        providers: str | Iterable[str] | None = None,
        *,
        locale: str | None = None,
        provider_names: str | Iterable[str] | None = None,
        required_providers: str | Iterable[str] | None = None,
        provider_methods: str | Iterable[str] | None = None,
    ) -> ContextManager["_RecordScope"]:
        """Return a scope in which all provider calls use one locale."""
        provider_args = [
            providers,
            provider_names,
            required_providers,
            provider_methods,
        ]
        if sum(arg is not None for arg in provider_args) > 1:
            raise ValueError("Only one collection of required record providers may be supplied.")

        required = next((arg for arg in provider_args if arg is not None), None)
        required = self._normalize_record_providers(required)
        record_locale, factory = self._select_record_factory(locale, required)
        return _RecordScope(self, record_locale, factory)

    record_scope = record

    @staticmethod
    def _normalize_record_providers(providers: str | Iterable[str] | None) -> list[str]:
        if providers is None:
            return []
        if isinstance(providers, str):
            provider_names = [providers]
        elif isinstance(providers, Iterable) and not isinstance(providers, (bytes, bytearray)):
            provider_names = list(providers)
        else:
            raise TypeError("Required record providers must be a string or an iterable of strings.")

        if not all(isinstance(provider_name, str) for provider_name in provider_names):
            raise TypeError("Required record provider names must be strings.")

        normalized_names = []
        for provider_name in provider_names:
            if provider_name not in normalized_names:
                normalized_names.append(provider_name)
        return normalized_names

    def _select_record_factory(
        self,
        locale: str | None,
        required_providers: Sequence[str],
    ) -> tuple[str, Generator | Faker]:
        if locale is not None:
            if not isinstance(locale, str):
                raise TypeError("Record locale must be a string.")
            normalized_locale = locale.replace("-", "_")
            if normalized_locale not in self._factory_map:
                msg = f"Invalid configuration for record locale `{normalized_locale}`"
                raise AttributeError(msg)

            factory = self._factory_map[normalized_locale]
            missing_providers = [
                provider_name
                for provider_name in required_providers
                if not self._record_supports(factory, normalized_locale, provider_name)
            ]
            if missing_providers:
                details = self._format_missing_report({normalized_locale: missing_providers})
                msg = f"No configured locale provides all required record providers: {details}"
                raise AttributeError(msg)
            return normalized_locale, factory

        eligible_factories = []
        missing_by_locale = OrderedDict()
        for candidate_locale, factory in self._factory_map.items():
            missing_providers = [
                provider_name
                for provider_name in required_providers
                if not self._record_supports(factory, candidate_locale, provider_name)
            ]
            if missing_providers:
                missing_by_locale[candidate_locale] = missing_providers
            else:
                eligible_factories.append(factory)

        if not eligible_factories:
            details = self._format_missing_report(missing_by_locale)
            msg = f"No configured locale provides all required record providers: {details}"
            raise AttributeError(msg)

        if len(eligible_factories) == 1:
            factory = eligible_factories[0]
        elif self._weights:
            random_factory = eligible_factories[0]
            if isinstance(random_factory, Faker) and len(random_factory.factories) == 1:
                random_factory = random_factory.factories[0]
            weights = [
                weight
                for candidate_factory, weight in zip(self._factories, self._weights)
                if any(candidate_factory is eligible_factory for eligible_factory in eligible_factories)
            ]
            factory = choices_distribution(eligible_factories, weights, random_factory.random, length=1)[0]
        else:
            factory = self._select_factory_choice(eligible_factories)

        record_locale = next(
            candidate_locale
            for candidate_locale, candidate_factory in self._factory_map.items()
            if candidate_factory is factory
        )
        return record_locale, factory

    @staticmethod
    def _format_missing_report(missing_by_locale: Mapping[str, Sequence[str]]) -> str:
        return "; ".join(
            f"{locale} is missing {', '.join(missing_providers)}"
            for locale, missing_providers in missing_by_locale.items()
        )

    @classmethod
    def _record_supports(cls, factory: Generator | Faker, locale: str, provider_name: str) -> bool:
        if isinstance(factory, Faker):
            if len(factory.factories) != 1:
                return False
            factory = factory.factories[0]

        if "." in provider_name:
            provider = None
            if hasattr(factory, "provider"):
                provider = factory.provider(provider_name)
            if provider is None:
                return False
            provider_language = getattr(provider, "__lang__", None)
            return provider_language is None or provider_language == locale

        method = getattr(factory, provider_name, None)
        if not callable(method):
            return False

        owner = getattr(method, "__self__", None)
        if owner is factory:
            factory_language = getattr(factory, "__lang__", None)
            return factory_language is None or factory_language == locale

        for provider in cls._record_providers(factory):
            if getattr(provider, provider_name, None) != method:
                continue
            provider_language = getattr(provider, "__lang__", None)
            return provider_language is None or provider_language == locale
        return False

    @staticmethod
    def _record_providers(factory: Generator | Faker) -> list[BaseProvider]:
        get_providers = getattr(factory, "get_providers", None)
        if not callable(get_providers):
            return []
        providers = get_providers()
        return list(providers) if providers is not None else []

    @classmethod
    def _get_record_attribute(cls, scope: "_RecordScope", attr: str) -> Any:
        try:
            value = getattr(scope.factory, attr)
        except AttributeError as exc:
            if not cls._record_supports(scope.factory, scope.locale, attr):
                msg = f"Record locale {scope.locale!r} does not provide {attr!r}."
                raise AttributeError(msg) from exc
            raise

        if callable(value) and not cls._record_supports(scope.factory, scope.locale, attr):
            msg = f"Record locale {scope.locale!r} does not provide {attr!r}."
            raise AttributeError(msg)
        return value

    @property
    def unique(self) -> UniqueProxy:
        return self._unique_proxy

    @property
    def optional(self) -> OptionalProxy:
        return self._optional_proxy

    def _select_factory(self, method_name: str) -> Factory:
        """
        Returns a random factory that supports the provider method

        :param method_name: Name of provider method
        :return: A factory that supports the provider method
        """

        factories, weights = self._map_provider_method(method_name)

        if len(factories) == 0:
            msg = f"No generator object has attribute {method_name!r}"
            raise AttributeError(msg)
        elif len(factories) == 1:
            return factories[0]

        if weights:
            factory = self._select_factory_distribution(factories, weights)
        else:
            factory = self._select_factory_choice(factories)
        return factory

    def _select_factory_distribution(self, factories, weights):
        return choices_distribution(factories, weights, self.factories[0].random, length=1)[0]

    def _select_factory_choice(self, factories):
        return self._factories[0].random.choice(factories)

    def _map_provider_method(self, method_name: str) -> tuple[list[Factory], list[float] | None]:
        """
        Creates a 2-tuple of factories and weights for the given provider method name

        The first element of the tuple contains a list of compatible factories.
        The second element of the tuple contains a list of distribution weights.

        :param method_name: Name of provider method
        :return: 2-tuple (factories, weights)
        """

        # Return cached mapping if it exists for given method
        attr = f"_cached_{method_name}_mapping"
        if hasattr(self, attr):
            return getattr(self, attr)

        # Create mapping if it does not exist
        if self._weights:
            value = [
                (factory, weight)
                for factory, weight in zip(self.factories, self._weights)
                if hasattr(factory, method_name)
            ]
            factories, weights = zip(*value)
            mapping = list(factories), list(weights)
        else:
            value = [factory for factory in self.factories if hasattr(factory, method_name)]  # type: ignore
            mapping = value, None  # type: ignore

        # Then cache and return results
        setattr(self, attr, mapping)
        return mapping

    @classmethod
    def seed(cls, seed: SeedType | None = None) -> None:
        """
        Hashables the shared `random.Random` object across all factories

        :param seed: seed value
        """
        Generator.seed(seed)

    def seed_instance(self, seed: SeedType | None = None) -> None:
        """
        Creates and seeds a new `random.Random` object for each factory

        :param seed: seed value
        """
        for factory in self._factories:
            factory.seed_instance(seed)

    def seed_locale(self, locale: str, seed: SeedType | None = None) -> None:
        """
        Creates and seeds a new `random.Random` object for the factory of the specified locale

        :param locale: locale string
        :param seed: seed value
        """
        self._factory_map[locale.replace("-", "_")].seed_instance(seed)

    @property
    def random(self) -> Random:
        """
        Proxies `random` getter calls

        In single locale mode, this will be proxied to the `random` getter
        of the only internal `Generator` object. Subclasses will have to
        implement desired behavior in multiple locale mode.
        """

        if self._record_scopes:
            return self._record_scopes[-1].factory.random

        if len(self._factories) == 1:
            return self._factories[0].random
        else:
            msg = "Proxying `random` getter calls is not implemented in multiple locale mode."
            raise NotImplementedError(msg)

    @random.setter
    def random(self, value: Random) -> None:
        """
        Proxies `random` setter calls

        In single locale mode, this will be proxied to the `random` setter
        of the only internal `Generator` object. Subclasses will have to
        implement desired behavior in multiple locale mode.
        """

        if self._record_scopes:
            self._record_scopes[-1].factory.random = value
            return

        if len(self._factories) == 1:
            self._factories[0].random = value
        else:
            msg = "Proxying `random` setter calls is not implemented in multiple locale mode."
            raise NotImplementedError(msg)

    @property
    def locales(self) -> list[str]:
        return list(self._locales)

    @property
    def weights(self) -> list[int | float] | None:
        return self._weights

    @property
    def factories(self) -> list[Generator | Faker]:
        return self._factories

    def items(self) -> list[tuple[str, Generator | Faker]]:
        return list(self._factory_map.items())


class _RecordScope:
    def __init__(self, proxy: Faker, locale: str, factory: Generator | Faker) -> None:
        self._proxy = proxy
        self._locale = locale
        self._factory = factory

    def __enter__(self) -> "_RecordScope":
        self._proxy._record_scopes.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        scopes = self._proxy._record_scopes
        if not scopes or scopes[-1] is not self:
            msg = "Record scope exited in the wrong order."
            raise RuntimeError(msg)
        scopes.pop()

    def __getattr__(self, attr: str) -> Any:
        try:
            scopes = object.__getattribute__(self, "_proxy")._record_scopes
        except AttributeError as exc:
            raise AttributeError(attr) from exc

        if not scopes or scopes[-1] is not self:
            msg = "Record scope is not active."
            raise RuntimeError(msg)
        return Faker._get_record_attribute(self, attr)

    @property
    def proxy(self) -> Faker:
        return self._proxy

    @property
    def locale(self) -> str:
        return self._locale

    @property
    def record_locale(self) -> str:
        return self._locale

    @property
    def factory(self) -> Generator | Faker:
        return self._factory

    @property
    def unique(self) -> UniqueProxy:
        return self._proxy.unique

    @property
    def optional(self) -> OptionalProxy:
        return self._proxy.optional

    def record(self, *args: Any, **kwargs: Any) -> ContextManager["_RecordScope"]:
        return self._proxy.record(*args, **kwargs)

    record_scope = record


class UniqueProxy:
    def __init__(self, proxy: Faker, excluded_types: tuple[type, ...] = ()):
        self._proxy = proxy
        self._seen: dict = {}
        self._sentinel = object()
        self._excluded_types = excluded_types

    def clear(self) -> None:
        self._seen = {}

    def exclude_types(self, types: list[type]) -> UniqueProxy:
        """Return new UniqueProxy excluding specified types from uniqueness checks.

        Args:
            types: List of types to exclude from uniqueness enforcement

        Returns:
            New UniqueProxy instance with excluded types configured

        Example:
            >>> fake = Faker()
            >>> # Bools won't enforce uniqueness, but other types will
            >>> proxy = fake.unique.exclude_types([bool])
            >>> proxy.pybool()  # Can return duplicates
            >>> proxy.name()  # Still enforces uniqueness
        """
        new_proxy = UniqueProxy(self._proxy, tuple(types))
        new_proxy._seen = self._seen
        new_proxy._sentinel = self._sentinel
        return new_proxy

    def __getitem__(self, locale: str) -> UniqueProxy:
        locale_proxy = self._proxy[locale]
        unique_proxy = UniqueProxy(locale_proxy, self._excluded_types)
        unique_proxy._seen = self._seen
        unique_proxy._sentinel = self._sentinel
        return unique_proxy

    def __getattr__(self, name: str) -> Any:
        obj = getattr(self._proxy, name)
        if callable(obj):
            return self._wrap(name, obj)
        else:
            raise TypeError("Accessing non-functions through .unique is not supported.")

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _make_hashable(self, value: Any) -> Any:
        """Convert unhashable types (e.g., dict) to a hashable representation."""
        if isinstance(value, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in value.items()))
        elif isinstance(value, list):
            return tuple(self._make_hashable(v) for v in value)
        elif isinstance(value, set):
            return frozenset(self._make_hashable(v) for v in value)
        return value

    def _wrap(self, name: str, function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # If types are excluded, call function once to check return type
            if self._excluded_types:
                retval = function(*args, **kwargs)
                # Skip uniqueness check if type is excluded
                if isinstance(retval, self._excluded_types):
                    return retval
                # If not excluded, continue with normal uniqueness logic
                # but we already have a value, so we'll use it if unique
                hashable_retval = self._make_hashable(retval)
                key = (name, args, tuple(sorted(kwargs.items())))
                generated = self._seen.setdefault(key, {self._sentinel})

                # Check if this first value is unique
                if hashable_retval not in generated:
                    generated.add(hashable_retval)
                    return retval
                # Not unique, continue with normal loop below
            else:
                # No exclusions, use original logic
                key = (name, args, tuple(sorted(kwargs.items())))
                generated = self._seen.setdefault(key, {self._sentinel})
                retval = self._sentinel
                hashable_retval = self._make_hashable(retval)

            # Original uniqueness logic (with potential first attempt already done)
            for i in range(_UNIQUE_ATTEMPTS):
                if hashable_retval not in generated:
                    break
                retval = function(*args, **kwargs)
                hashable_retval = self._make_hashable(retval)
            else:
                raise UniquenessException(f"Got duplicated values after {_UNIQUE_ATTEMPTS:,} iterations.")

            generated.add(hashable_retval)

            return retval

        return wrapper


class OptionalProxy:
    """
    Return either a fake value or None, with a customizable probability.
    """

    def __init__(self, proxy: Faker):
        self._proxy = proxy

    def __getattr__(self, name: str) -> Any:
        obj = getattr(self._proxy, name)
        if callable(obj):
            return self._wrap(name, obj)
        else:
            raise TypeError("Accessing non-functions through .optional is not supported.")

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _wrap(self, name: str, function: Callable[..., RetType]) -> Callable[..., RetType | None]:
        @functools.wraps(function)
        def wrapper(*args: Any, prob: float = 0.5, **kwargs: Any) -> RetType | None:
            if not 0 < prob <= 1.0:
                raise ValueError("prob must be between 0 and 1")
            record_scopes = getattr(self._proxy, "_record_scopes", [])
            if record_scopes:
                enabled = record_scopes[-1].factory.random.randint(1, 100) <= prob * 100
                return function(*args, **kwargs) if enabled else None

            return function(*args, **kwargs) if self._proxy.boolean(chance_of_getting_true=int(prob * 100)) else None

        return wrapper
