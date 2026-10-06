import math
import re
import unittest

from decimal import Decimal

from faker import Faker
from faker.providers.geo.pt_PT import Provider as PtPtProvider


class TestArDz(unittest.TestCase):
    def setUp(self):
        self.fake = Faker("ar_DZ")
        Faker.seed(0)

    def test_latitude(self):
        lat = self.fake.latitude()
        assert isinstance(lat, Decimal)

    def test_longitude(self):
        long = self.fake.longitude()
        assert isinstance(long, Decimal)

    def test_latlng(self):
        loc = self.fake.latlng()
        assert isinstance(loc, tuple)
        assert len(loc) == 2
        assert isinstance(loc[0], Decimal)
        assert isinstance(loc[1], Decimal)

    def test_coordinate(self):
        loc = self.fake.coordinate()
        assert isinstance(loc, Decimal)

    def test_coordinate_centered(self):
        loc = self.fake.coordinate(center=23)
        assert round(loc) == 23

    def test_coordinate_rounded(self):
        loc = self.fake.coordinate(center=23, radius=3)
        assert 20 <= round(loc) <= 26

    def test_location_on_land(self):
        loc = self.fake.location_on_land()
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])  # Should be able to cast first two elements of tuple to Decimal
        assert Decimal(loc[1])
        assert isinstance(loc[2], str)  # Place is a string
        assert isinstance(loc[3], str)  # Country code is a string
        assert len(loc[3]) == 2  # Country code is two letters
        assert isinstance(loc[4], str)  # Timezone is a string

    def test_location_on_land_coords_only(self):
        loc = self.fake.location_on_land(coords_only=True)
        assert isinstance(loc, tuple)
        assert len(loc) == 2
        assert Decimal(loc[0])  # Should be able to cast first two elements of tuple to Decimal
        assert Decimal(loc[1])


class TestCsCz(unittest.TestCase):
    def setUp(self):
        self.fake = Faker("cs_CZ")
        Faker.seed(0)

    def test_location_on_land(self):
        loc = self.fake.location_on_land()
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])  # Should be able to cast first two elements of tuple to Decimal
        assert Decimal(loc[1])
        assert isinstance(loc[2], str)  # Place is a string
        assert isinstance(loc[3], str)  # Country code is a string
        assert len(loc[3]) == 2  # Country code is two letters
        assert isinstance(loc[4], str)  # Timezone is a string


class TestDeAT(unittest.TestCase):
    """Tests in addresses in the de_AT locale"""

    def setUp(self):
        self.fake = Faker("de_AT")
        Faker.seed(0)

    def test_local_latitude(self):
        local_latitude = self.fake.local_latitude()
        assert re.match(r"4[6-8]\.\d+", str(local_latitude))

    def test_local_longitude(self):
        local_longitude = self.fake.local_longitude()
        assert re.match(r"1[1-5]\.\d+", str(local_longitude))


class TestEnUS(unittest.TestCase):
    """Tests geographic locations in the en_US locale"""

    def setUp(self):
        self.fake = Faker("en_US")
        Faker.seed(0)

    def test_latitude(self):
        lat = self.fake.latitude()
        assert isinstance(lat, Decimal)

    def test_longitude(self):
        long = self.fake.longitude()
        assert isinstance(long, Decimal)

    def test_latlng(self):
        loc = self.fake.latlng()
        assert isinstance(loc, tuple)
        assert len(loc) == 2
        assert isinstance(loc[0], Decimal)
        assert isinstance(loc[1], Decimal)

    def test_coordinate(self):
        loc = self.fake.coordinate()
        assert isinstance(loc, Decimal)

    def test_coordinate_centered(self):
        loc = self.fake.coordinate(center=23)
        assert round(loc) == 23

    def test_coordinate_rounded(self):
        loc = self.fake.coordinate(center=23, radius=3)
        assert 20 <= round(loc) <= 26

    def test_location_on_land(self):
        loc = self.fake.location_on_land()
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])  # Should be able to cast first two elements of tuple to Decimal
        assert Decimal(loc[1])
        assert isinstance(loc[2], str)  # Place is a string
        assert isinstance(loc[3], str)  # Country code is a string
        assert len(loc[3]) == 2  # Country code is two letters
        assert isinstance(loc[4], str)  # Timezone is a string

    def test_location_on_land_coords_only(self):
        loc = self.fake.location_on_land(coords_only=True)
        assert isinstance(loc, tuple)
        assert len(loc) == 2
        assert Decimal(loc[0])  # Should be able to cast first two elements of tuple to Decimal
        assert Decimal(loc[1])


class TestEnIe(TestEnUS):
    def setUp(self):
        self.fake = Faker("en_IE")
        Faker.seed(0)


class TestFrDz(TestEnUS):
    def setUp(self):
        self.fake = Faker("fr_DZ")
        Faker.seed(0)


class TestGlobal(unittest.TestCase):
    """Tests geographic locations regardless of locale"""

    def setUp(self):
        self.fake = Faker()  # No locale specified, gets global for this provider
        Faker.seed(0)

    def test_local_latlng(self):
        loc = self.fake.local_latlng(country_code="US")
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])
        assert Decimal(loc[1])

        loc_short = self.fake.local_latlng(country_code="US", coords_only=True)
        assert len(loc_short) == 2
        assert Decimal(loc_short[0])
        assert Decimal(loc_short[1])


class TestRegionSampling(unittest.TestCase):
    def setUp(self):
        self.fake = Faker()
        self.fake.seed_instance(12345)

    @staticmethod
    def spherical_area(south, north, longitude_width):
        return math.radians(longitude_width) * (
            math.sin(math.radians(north)) - math.sin(math.radians(south))
        )

    def test_region_area_overlaps_once_and_removes_exclusions(self):
        equatorial_band = (-10, 10, -180, 180)
        polar_band = (60, 80, -180, 180)
        cross_dateline_band = (-30, -20, 170, -170)
        dateline_hole = (-27, -23, 176, -176)

        equatorial_area = self.spherical_area(-10, 10, 360)
        polar_area = self.spherical_area(60, 80, 360)
        cross_area = self.spherical_area(-30, -20, 20)
        hole_area = self.spherical_area(-27, -23, 8)

        assert self.fake.region_area([equatorial_band, equatorial_band]) == equatorial_area
        assert self.fake.region_area(cross_dateline_band, dateline_hole) == cross_area - hole_area

        total_area = equatorial_area + polar_area + cross_area - hole_area
        region = [equatorial_band, polar_band, cross_dateline_band]
        assert math.isclose(self.fake.region_area(region, dateline_hole), total_area)

    def test_samples_match_ranges_holes_and_spherical_area_proportions(self):
        equatorial_band = (-10, 10, -180, 180)
        polar_band = (60, 80, -180, 180)
        cross_dateline_band = (-30, -20, 170, -170)
        dateline_hole = (-27, -23, 176, -176)
        region = [equatorial_band, polar_band, cross_dateline_band]

        point = self.fake.latlng_in_region(cross_dateline_band, dateline_hole)
        assert isinstance(point, tuple)
        assert all(isinstance(coordinate, Decimal) for coordinate in point)

        sample_size = 30000
        points = self.fake.latlngs_in_region(region, exclusions=dateline_hole, count=sample_size)
        assert len(points) == sample_size

        counts = {
            "equatorial": 0,
            "polar": 0,
            "cross_dateline": 0,
        }
        seen_east_of_dateline = False
        seen_west_of_dateline = False

        for latitude, longitude in points:
            latitude = float(latitude)
            longitude = float(longitude)
            assert -90 <= latitude <= 90
            assert -180 <= longitude <= 180

            if -10 <= latitude <= 10:
                counts["equatorial"] += 1
            elif 60 <= latitude <= 80:
                counts["polar"] += 1
            elif -30 <= latitude <= -20 and (longitude >= 170 or longitude <= -170):
                counts["cross_dateline"] += 1
            else:
                raise AssertionError("Coordinate was sampled outside the configured region")

            in_hole = -27 <= latitude <= -23 and (longitude >= 176 or longitude <= -176)
            assert not in_hole
            if -30 <= latitude <= -20:
                seen_east_of_dateline = seen_east_of_dateline or longitude >= 170
                seen_west_of_dateline = seen_west_of_dateline or longitude <= -170

        assert seen_east_of_dateline
        assert seen_west_of_dateline

        areas = {
            "equatorial": self.spherical_area(-10, 10, 360),
            "polar": self.spherical_area(60, 80, 360),
            "cross_dateline": self.spherical_area(-30, -20, 20)
            - self.spherical_area(-27, -23, 8),
        }
        total_area = sum(areas.values())
        tolerance = 0.012
        for name, area in areas.items():
            assert abs(counts[name] / sample_size - area / total_area) < tolerance

    def test_zero_area_and_fully_excluded_region(self):
        assert self.fake.region_area((90, 90, -180, 180)) == 0
        assert self.fake.region_area((0, 1, -180, 180), (0, 1, -180, 180)) == 0
        assert self.fake.latlngs_in_region((0, 0, -180, 180), count=0) == []

        with self.assertRaises(ValueError):
            self.fake.latlng_in_region((90, 90, -180, 180))
        with self.assertRaises(ValueError):
            self.fake.latlng_in_region(
                (0, 1, -180, 180),
                (0, 1, -180, 180),
            )

    def test_pole_region_and_invalid_rectangles(self):
        points = self.fake.latlngs_in_region((89, 90, -180, 180), count=100)
        assert all(89 <= float(latitude) <= 90 for latitude, _ in points)

        with self.assertRaises(ValueError):
            self.fake.latlng_in_region((91, 90, -180, 180))
        with self.assertRaises(ValueError):
            self.fake.latlng_in_region((0, 1, -181, 180))
        with self.assertRaises(TypeError):
            self.fake.latlng_in_region((0, 1, 0))


class TestPlPl(unittest.TestCase):
    def setUp(self):
        self.fake = Faker("pl_PL")
        Faker.seed(0)

    def test_location_on_land(self):
        loc = self.fake.location_on_land()
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])  # Should be able to cast first two elements of tuple to Decimal
        assert Decimal(loc[1])
        assert isinstance(loc[2], str)  # Place is a string
        assert isinstance(loc[3], str)  # Country code is a string
        assert len(loc[3]) == 2  # Country code is two letters
        assert isinstance(loc[4], str)  # Timezone is a string


class TestPtPT(unittest.TestCase):
    def setUp(self):
        self.fake = Faker("pt_PT")
        Faker.seed(0)

    def test_nationality(self):
        nationality = self.fake.nationality()
        assert isinstance(nationality, str)
        assert nationality in PtPtProvider.nationalities


class TestSkSk(unittest.TestCase):
    def setUp(self):
        self.fake = Faker("sk_SK")
        Faker.seed(0)

    def test_location_on_land(self):
        loc = self.fake.location_on_land()
        assert isinstance(loc, tuple)
        assert len(loc) == 5
        assert Decimal(loc[0])  # Should be able to cast first two elements of tuple to Decimal
        assert Decimal(loc[1])
        assert isinstance(loc[2], str)  # Place is a string
        assert isinstance(loc[3], str)  # Country code is a string
        assert len(loc[3]) == 2  # Country code is two letters
        assert isinstance(loc[4], str)  # Timezone is a string


class TestTrTr(TestEnUS):
    def setUp(self):
        self.fake = Faker("tr_TR")
        Faker.seed(0)


class TestMkMk(unittest.TestCase):
    """Tests geo provider for mk_MK locale"""

    def setUp(self):
        self.fake = Faker("mk_MK")
        Faker.seed(0)

    def test_local_latitude(self):
        for _ in range(100):
            lat = self.fake.local_latitude()
            assert 40.85 <= float(lat) <= 42.37, f"Latitude {lat} out of MK bounds"

    def test_local_longitude(self):
        for _ in range(100):
            lon = self.fake.local_longitude()
            assert 20.45 <= float(lon) <= 23.03, f"Longitude {lon} out of MK bounds"
