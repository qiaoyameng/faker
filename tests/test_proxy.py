import copy
import pickle
import random

from collections import OrderedDict
from unittest.mock import PropertyMock, patch

import pytest

from faker import Faker
from faker.config import DEFAULT_LOCALE
from faker.generator import Generator


class AlphaRecordFactory:
    def __init__(self, locale):
        self.random = random.Random()
        setattr(self, "__lang__", locale)

    def get_providers(self):
        return [self]

    def alpha(self):
        return getattr(self, "__lang__")

    def shared(self):
        return getattr(self, "__lang__")


class BetaRecordFactory:
    def __init__(self, locale):
        self.random = random.Random()
        setattr(self, "__lang__", locale)

    def get_providers(self):
        return [self]

    def beta(self):
        return getattr(self, "__lang__")

    def shared(self):
        return getattr(self, "__lang__")


def custom_record_fake(locales, weights=None):
    fake = Faker(list(locales))
    alpha_locales, beta_locales = locales
    fake._factory_map = OrderedDict(
        [
            (alpha_locales, AlphaRecordFactory(alpha_locales)),
            (beta_locales, BetaRecordFactory(beta_locales)),
        ]
    )
    fake._locales = list(locales)
    fake._factories = list(fake._factory_map.values())
    fake._weights = weights
    return fake


class TestFakerProxyClass:
    """Test Faker proxy class"""

    def test_unspecified_locale(self):
        fake = Faker()
        assert len(fake.locales) == 1
        assert len(fake.factories) == 1
        assert fake.locales[0] == DEFAULT_LOCALE

    def test_locale_as_string(self):
        locale = "en_US"
        fake = Faker(locale)
        assert len(fake.locales) == 1
        assert len(fake.factories) == 1
        assert fake.locales[0] == locale

    def test_locale_as_list(self):
        locale = ["en-US", "en_PH", "ja_JP", "de-DE"]
        expected = ["en_US", "en_PH", "ja_JP", "de_DE"]
        fake = Faker(locale)
        assert fake.locales == expected
        assert len(fake.factories) == len(expected)

        locale = ["en-US", "en_PH", "ja_JP", "de-DE", "ja-JP", "de_DE", "en-US"] * 3
        expected = ["en_US", "en_PH", "ja_JP", "de_DE"]
        fake = Faker(locale)
        assert fake.locales == expected
        assert len(fake.factories) == len(expected)

    def test_locale_as_list_invalid_value_type(self):
        locale = [1, 2]
        with pytest.raises(TypeError) as exc:
            Faker(locale)
        assert str(exc.value) == 'The locale "1" must be a string.'

    def test_locale_as_ordereddict(self):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )

        fake = Faker(locale)
        assert fake.locales == ["de_DE", "en_US", "en_PH", "ja_JP"]
        assert len(fake.factories) == 4
        assert fake.weights == [3, 2, 1, 5]

        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
                ("de-DE", 4),
                ("ja-JP", 2),
                ("en-US", 1),
            ]
        )
        fake = Faker(locale)
        assert fake.locales == ["de_DE", "en_US", "en_PH", "ja_JP"]
        assert len(fake.factories) == 4
        assert fake.weights == [4, 1, 1, 2]

    def test_invalid_locale(self):
        with pytest.raises(AttributeError):
            Faker("foo_Bar")

        with pytest.raises(AttributeError):
            Faker(["en_US", "foo_Bar"])

        with pytest.raises(AttributeError):
            Faker(
                OrderedDict(
                    [
                        ("de_DE", 3),
                        ("en-US", 2),
                        ("en-PH", 1),
                        ("foo_Bar", 5),
                    ]
                )
            )

    def test_items(self):
        locale = ["de_DE", "en-US", "en-PH", "ja_JP", "de-DE", "ja-JP", "en-US"]
        processed_locale = list({code.replace("-", "_") for code in locale})
        fake = Faker(locale)
        for locale_name, factory in fake.items():
            assert locale_name in processed_locale
            assert isinstance(factory, (Generator, Faker))

    def test_dunder_getitem(self):
        locale = ["de_DE", "en-US", "en-PH", "ja_JP"]
        fake = Faker(locale)

        for code in locale:
            assert isinstance(fake[code], (Generator, Faker))

        with pytest.raises(KeyError):
            fake["en_GB"]

    def test_seed_classmethod(self):
        fake = Faker()

        # Verify `seed()` is not callable from a class instance
        with pytest.raises(TypeError):
            fake.seed(0)

        # Verify calls to `seed()` from a class object are proxied properly
        with patch("faker.generator.Generator.seed") as mock_seed:
            mock_seed.assert_not_called()
            Faker.seed(0)
            mock_seed.assert_called_once_with(0)

    def test_seed_class_locales(self):
        Faker.seed(2043)
        count = 5
        fake = Faker(["en_GB", "fr_FR", "en_IN"])
        first_list = [fake.name() for _ in range(count)]
        # We convert the list to a set to remove duplicates and ensure
        # that we have exactly `count` unique fake values
        assert len(set(first_list)) == count

        Faker.seed(2043)
        fake = Faker(["en_GB", "fr_FR", "en_IN"])
        second_list = [fake.name() for _ in range(count)]

        assert first_list == second_list

    def test_seed_instance(self):
        locale = ["de_DE", "en-US", "en-PH", "ja_JP"]
        fake = Faker(locale)

        with patch("faker.generator.Generator.seed_instance") as mock_seed_instance:
            mock_seed_instance.assert_not_called()
            fake.seed_instance(0)

            # Verify `seed_instance(0)` was called 4 times (one for each locale)
            calls = mock_seed_instance.call_args_list
            assert len(calls) == 4
            for call in calls:
                args, kwargs = call
                assert args == (0,)
                assert kwargs == {}

    def test_seed_locale(self):
        from faker.generator import random as shared_random_instance

        locale = ["de_DE", "en-US", "en-PH", "ja_JP"]
        fake = Faker(locale)

        # Get current state of each factory's random instance
        states = {}
        for locale, factory in fake.items():
            states[locale] = factory.random.getstate()

        # Create a new random instance for en_US factory with seed value
        fake.seed_locale("en_US", 0)

        for locale, factory in fake.items():
            # en_US factory should have changed
            if locale == "en_US":
                assert factory.random != shared_random_instance
                assert factory.random.getstate() != states[locale]

            # There should be no changes for the rest
            else:
                assert factory.random == shared_random_instance
                assert factory.random.getstate() == states[locale]

    def test_single_locale_proxy_behavior(self):
        fake = Faker()
        internal_factory = fake.factories[0]

        # Test if `Generator` attributes are proxied properly
        for attr in fake.generator_attrs:
            assert getattr(fake, attr) == getattr(internal_factory, attr)

        # Test if `random` getter and setter are proxied properly
        tmp_random = fake.random
        assert internal_factory.random != 1
        fake.random = 1
        assert internal_factory.random == 1
        fake.random = tmp_random

        # Test if a valid provider method is proxied properly
        # Factory selection logic should not be triggered
        with patch("faker.proxy.Faker._select_factory") as mock_select_factory:
            mock_select_factory.assert_not_called()
            assert fake.name == internal_factory.name
            fake.name()
            mock_select_factory.assert_not_called()

    def test_multiple_locale_proxy_behavior(self):
        fake = Faker(["de-DE", "en-US", "en-PH", "ja-JP"])

        # `Generator` attributes are not implemented
        for attr in fake.generator_attrs:
            with pytest.raises(NotImplementedError):
                getattr(fake, attr)

        # The `random` getter is not implemented
        with pytest.raises(NotImplementedError):
            random = fake.random
            random.seed(0)

        # The `random` setter is not implemented
        with pytest.raises(NotImplementedError):
            fake.random = 1

    def test_multiple_locale_caching_behavior(self):
        fake = Faker(["de_DE", "en-US", "en-PH", "ja_JP"])

        with patch("faker.proxy.Faker._map_provider_method", wraps=fake._map_provider_method) as mock_map_method:
            mock_map_method.assert_not_called()
            assert not hasattr(fake, "_cached_name_mapping")

            # Test cache creation
            fake.name()
            assert hasattr(fake, "_cached_name_mapping")
            mock_map_method.assert_called_once_with("name")

            # Test subsequent cache access
            with patch.object(Faker, "_cached_name_mapping", create=True, new_callable=PropertyMock) as mock_cached_map:
                # Keep test fast by patching the cached mapping to return something simpler
                mock_cached_map.return_value = [fake["en_US"]], [1]
                for _ in range(100):
                    fake.name()

                # Python's hasattr() internally calls getattr()
                # So each call to name() accesses the cached mapping twice
                assert mock_cached_map.call_count == 200

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_no_weights(self, mock_factory_distribution, mock_factory_choice):
        fake = Faker(["de_DE", "en-US", "en-PH", "ja_JP"])

        # There are no distribution weights, so factory selection logic will use `random.choice`
        # if multiple factories have the specified provider method
        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            mock_select_factory.assert_not_called()
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()

            # All factories for the listed locales have the `name` provider method
            fake.name()
            mock_select_factory.assert_called_once_with("name")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_called_once_with(fake.factories)
            mock_select_factory.reset_mock()
            mock_factory_distribution.reset_mock()
            mock_factory_choice.reset_mock()

            # Only `en_PH` factory has provider method `luzon_province`, so there is no
            # need for `random.choice` factory selection logic to run
            fake.luzon_province()
            mock_select_factory.assert_called_with("luzon_province")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()
            mock_select_factory.reset_mock()
            mock_factory_distribution.reset_mock()
            mock_factory_choice.reset_mock()

            # Both `en_US` and `ja_JP` factories have provider method `zipcode`
            fake.zipcode()
            mock_select_factory.assert_called_once_with("zipcode")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_called_once_with(
                [fake["en_US"], fake["ja_JP"]],
            )

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_with_weights(self, mock_factory_distribution, mock_factory_choice):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale)
        mock_factory_distribution.assert_not_called()
        mock_factory_choice.assert_not_called()

        # Distribution weights have been specified, so factory selection logic will use
        # `choices_distribution` if multiple factories have the specified provider method
        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            # All factories for the listed locales have the `name` provider method
            fake.name()
            mock_select_factory.assert_called_once_with("name")
            mock_factory_distribution.assert_called_once_with(fake.factories, fake.weights)
            mock_factory_choice.assert_not_called()

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_single_provider(self, mock_factory_distribution, mock_factory_choice):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale)

        # Distribution weights have been specified, so factory selection logic will use
        # `choices_distribution` if multiple factories have the specified provider method
        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            # Only `en_PH` factory has provider method `luzon_province`, so there is no
            # need for `choices_distribution` factory selection logic to run
            fake.luzon_province()
            mock_select_factory.assert_called_once_with("luzon_province")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_shared_providers(self, mock_factory_distribution, mock_factory_choice):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale)

        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            # Both `en_US` and `ja_JP` factories have provider method `zipcode`
            fake.zipcode()
            mock_select_factory.assert_called_once_with("zipcode")
            mock_factory_distribution.assert_called_once_with([fake["en_US"], fake["ja_JP"]], [2, 5])
            mock_factory_choice.assert_not_called()

    def test_multiple_locale_factory_selection_unsupported_method(self):
        fake = Faker(["en_US", "en_PH"])
        with pytest.raises(AttributeError):
            fake.obviously_invalid_provider_method_a23f()

    @patch("random.Random.choice")
    @patch("random.Random.choices")
    def test_weighting_disabled_single_choice(self, mock_choices_fn, mock_choice_fn):
        fake = Faker(use_weighting=False)
        fake.first_name()
        mock_choice_fn.assert_called()
        mock_choices_fn.assert_not_called()

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_disabled_with_locales(self, mock_choices_fn, mock_choice_fn):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale, use_weighting=False)
        fake.first_name()
        mock_choices_fn.assert_called()  # select provider
        mock_choice_fn.assert_called()  # select within provider

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_disabled_multiple_locales(self, mock_choices_fn, mock_choice_fn):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale, use_weighting=False)
        fake.first_name()
        mock_choices_fn.assert_called()  # select provider
        mock_choice_fn.assert_called()  # select within provider

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_disabled_multiple_choices(self, mock_choices_fn, mock_choice_fn):
        fake = Faker(use_weighting=False)
        fake.uri_path(deep=3)

        assert mock_choices_fn.mock_calls[0][2]["k"] == 3
        assert mock_choices_fn.mock_calls[0][2]["weights"] is None
        mock_choice_fn.assert_not_called()

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_enabled_multiple_choices(self, mock_choices_fn, mock_choice_fn):
        fake = Faker(use_weighting=True)
        fake.uri_path(deep=3)

        assert mock_choices_fn.mock_calls[0][2]["k"] == 3
        assert mock_choices_fn.mock_calls[0][2]["weights"] is None
        mock_choice_fn.assert_not_called()

    def test_dir_include_all_providers_attribute_in_list(self):
        fake = Faker(["en_US", "en_PH"])
        expected = set(
            dir(Faker)
            + [
                "_factories",
                "_locales",
                "_factory_map",
                "_weights",
                "_unique_proxy",
                "_optional_proxy",
                "_record_scopes",
            ]
        )
        for factory in fake.factories:
            expected |= {attr for attr in dir(factory) if not attr.startswith("_")}
        expected = sorted(expected)
        attributes = dir(fake)
        assert attributes == expected

    def test_select_factory_distribution_uses_instance_random(self):
        from faker.utils.distribution import choices_distribution

        locale = OrderedDict([("de_DE", 3), ("en_US", 2), ("ja_JP", 5)])
        fake = Faker(locale)
        fake.seed_instance(12345)

        instance_random = fake._factories[0].random
        with patch("faker.proxy.choices_distribution", wraps=choices_distribution) as mock_dist:
            fake.name()
            mock_dist.assert_called_once()
            args = mock_dist.call_args[0]
            assert args[2] is instance_random

    def test_seed_instance_deterministic_multi_locale_no_weights(self):
        fake = Faker(["en_US", "ja_JP", "de_DE"])
        fake.seed_instance(12345)
        first_run = [fake.name() for _ in range(20)]

        fake2 = Faker(["en_US", "ja_JP", "de_DE"])
        fake2.seed_instance(12345)
        second_run = [fake2.name() for _ in range(20)]

        assert first_run == second_run

    def test_seed_instance_deterministic_multi_locale_with_weights(self):
        locale = OrderedDict([("de_DE", 3), ("en_US", 2), ("ja_JP", 5)])
        fake = Faker(locale)
        fake.seed_instance(12345)
        first_run = [fake.name() for _ in range(20)]

        fake2 = Faker(locale)
        fake2.seed_instance(12345)
        second_run = [fake2.name() for _ in range(20)]

        assert first_run == second_run

    def test_copy(self):
        fake = Faker("it_IT")
        fake2 = copy.deepcopy(fake)
        assert fake.locales == fake2.locales
        assert fake.locales is not fake2.locales
        assert fake2.factories[0] is fake2._factory_map["it_IT"]

    def test_copy_rebinds_single_locale_proxies(self):
        fake = Faker("en_US")
        fake2 = copy.deepcopy(fake)

        assert fake2.unique._proxy is fake2
        assert fake2.optional._proxy is fake2
        assert fake2.factories[0] is fake2._factory_map["en_US"]
        assert fake2.optional.name(prob=1.0)

    def test_copy_rebinds_multiple_locale_proxies(self):
        fake = Faker(["en_US", "ja_JP"])
        fake2 = copy.deepcopy(fake)

        assert fake2.unique._proxy is fake2
        assert fake2.optional._proxy is fake2
        assert fake2.factories[0] is fake2["en_US"]
        assert fake2.factories[1] is fake2["ja_JP"]
        assert fake2.optional.name(prob=1.0)

    def test_copy_unique_uses_copied_proxy_state(self):
        source = Faker("en_US")
        source.seed_instance(999)
        clone_a = copy.deepcopy(source)
        clone_b = copy.deepcopy(source)

        clone_a.seed_instance(200)
        clone_b.seed_instance(200)

        assert clone_a.unique._proxy is clone_a
        assert clone_b.unique._proxy is clone_b
        assert clone_a.unique.name() == clone_b.unique.name()

    def test_pickle(self):
        fake = Faker()
        pickled = pickle.dumps(fake)
        pickle.loads(pickled)


class TestFakerRecordScope:
    def test_record_scope_uses_weighted_eligible_factory_once(self):
        fake = custom_record_fake(("de_DE", "ja_JP"), weights=[3, 2])
        de_factory = fake._factory_map["de_DE"]
        ja_factory = fake._factory_map["ja_JP"]

        with patch("faker.proxy.choices_distribution", return_value=[de_factory]) as mock_dist:
            with fake.record(["shared"]) as record:
                assert record.locale == "de_DE"
                assert record.shared() == "de_DE"
                assert fake.shared() == "de_DE"
                assert mock_dist.call_count == 1

        with patch("faker.proxy.choices_distribution", return_value=[ja_factory]) as mock_dist:
            with fake.record(["shared"]) as record:
                assert record.locale == "ja_JP"
                assert record.shared() == "ja_JP"
                assert fake.shared() == "ja_JP"
                assert mock_dist.call_count == 1

        args = mock_dist.call_args[0]
        assert args[0] == [de_factory, ja_factory]
        assert args[1] == [3, 2]

    def test_record_scope_filters_by_custom_provider_coverage(self):
        fake = custom_record_fake(("de_DE", "ja_JP"), weights=[3, 2])

        with fake.record(["alpha"]) as record:
            assert record.locale == "de_DE"
            assert record.alpha() == "de_DE"
            assert record.shared() == "de_DE"
            with pytest.raises(AttributeError, match="does not provide 'beta'"):
                record.beta()

        with fake.record(["beta"]) as record:
            assert record.locale == "ja_JP"
            assert record.beta() == "ja_JP"
            assert record.shared() == "ja_JP"
            with pytest.raises(AttributeError, match="does not provide 'alpha'"):
                record.alpha()

    def test_record_scope_uses_unweighted_eligible_factory(self):
        fake = custom_record_fake(("de_DE", "ja_JP"))

        with patch.object(random.Random, "choice", return_value=fake._factory_map["ja_JP"]) as mock_choice:
            with fake.record(["shared"]) as record:
                assert record.locale == "ja_JP"
                assert record.shared() == "ja_JP"
            mock_choice.assert_called_once_with(fake.factories)

    def test_record_scope_reports_missing_required_providers(self):
        fake = custom_record_fake(("de_DE", "ja_JP"), weights=[3, 2])

        with pytest.raises(AttributeError) as exc_info:
            with fake.record(["alpha", "beta"]):
                pass

        message = str(exc_info.value)
        assert "de_DE is missing beta" in message
        assert "ja_JP is missing alpha" in message
        assert fake._record_scopes == []

    def test_record_scope_reports_unsupported_method_after_entry(self):
        fake = Faker(["en_US", "fr_FR", "en_PH"])

        with fake.record(["luzon_province"]) as record:
            with pytest.raises(AttributeError, match="does not provide 'name'"):
                record.name()

    def test_record_scope_accepts_explicit_hyphenated_locale(self):
        fake = custom_record_fake(("de_DE", "ja_JP"))

        with fake.record(provider_names=["shared"], locale="ja-JP") as record:
            assert record.locale == "ja_JP"
            assert record.shared() == "ja_JP"

        with pytest.raises(AttributeError, match="Invalid configuration for record locale `fr_FR`"):
            fake.record(locale="fr_FR")

    def test_record_scope_restores_method_selection_after_exit_and_exception(self):
        fake = Faker(["en_US", "fr_FR", "en_PH"])

        with patch.object(Faker, "_select_factory", return_value=fake["en_PH"]) as mock_select:
            with fake.record(locale="en_PH"):
                assert fake._record_scopes[-1].locale == "en_PH"
                assert fake.luzon_province()
            mock_select.assert_not_called()

            assert fake._record_scopes == []
            fake.luzon_province()
            mock_select.assert_called_once_with("luzon_province")

        with pytest.raises(RuntimeError, match="boom"):
            with fake.record(locale="fr_FR"):
                raise RuntimeError("boom")

        assert fake._record_scopes == []

    def test_nested_record_scopes_restore_outer_record(self):
        fake = Faker(["en_US", "fr_FR", "en_PH"])

        with fake.record(locale="fr_FR") as outer:
            assert outer.locale == "fr_FR"
            assert fake.name()

            with outer.record(locale="en_PH") as inner:
                assert inner.locale == "en_PH"
                assert inner.luzon_province()
                assert fake.luzon_province()
                with pytest.raises(AttributeError, match="does not provide 'name'"):
                    inner.name()

            assert fake._record_scopes[-1] is outer
            assert fake.name()
            with pytest.raises(AttributeError, match="does not provide 'luzon_province'"):
                fake.luzon_province()

        assert fake._record_scopes == []

    def test_nested_record_scope_exception_restores_outer_record(self):
        fake = Faker(["en_US", "fr_FR", "en_PH"])

        with fake.record(locale="fr_FR") as outer:
            with pytest.raises(ValueError, match="inner failure"):
                with outer.record(locale="en_PH"):
                    raise ValueError("inner failure")

            assert fake._record_scopes[-1] is outer
            assert fake.name()

    def test_record_scope_rejects_fallback_locale_but_outside_keeps_old_behavior(self):
        fake = Faker(["en"])

        with patch("faker.proxy.Factory") as mock_factory:
            outside_address = fake.address()
            mock_factory.assert_not_called()

        assert outside_address

        with fake.record(locale="en") as record:
            with pytest.raises(AttributeError, match="Record locale 'en' does not provide 'address'"):
                record.address()

        with fake.record(required_providers=["faker.providers.file"]) as record:
            assert record.file_name()

    def test_record_scope_unique_and_optional_keep_selected_locale(self):
        fake = Faker(["en_US", "fr_FR"])

        with fake.record(locale="fr_FR"):
            assert fake.unique.name()
            assert fake.optional.name(prob=1.0)
