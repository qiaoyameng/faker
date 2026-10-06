import math
import re
import unittest

from decimal import Decimal

from faker import Faker
from faker.providers.geo import GeoRect, GeoRegion
from faker.providers.geo.region import EARTH_RADIUS_KM
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


class TestGeoRegion(unittest.TestCase):
    def setUp(self):
        self.fake = Faker()
        Faker.seed(123456)

    def test_invalid_rectangles(self):
        with self.assertRaises(ValueError):
            GeoRect(-91, 0, 0, 1)
        with self.assertRaises(ValueError):
            GeoRect(0, 91, 0, 1)
        with self.assertRaises(ValueError):
            GeoRect(0, 1, -181, 0)
        with self.assertRaises(ValueError):
            GeoRect(0, 1, 0, 181)
        with self.assertRaises(ValueError):
            GeoRect(1, 0, 0, 1)
        with self.assertRaises(TypeError):
            GeoRegion([(0, 1, 0)])

    def test_zero_area_region(self):
        region = GeoRegion([GeoRect(0, 0, 0, 1)])
        self.assertTrue(region.is_empty)
        self.assertEqual(region.area_km2, 0)
        self.assertEqual(self.fake.latlngs_in_region(region, 5), [])
        with self.assertRaises(ValueError):
            self.fake.latlng_in_region(region)

    def test_entire_region_removed(self):
        region = GeoRegion(
            include=[GeoRect(10, 20, -180, 180)],
            exclude=[GeoRect(10, 20, -180, 180)],
        )
        self.assertTrue(region.is_empty)
        self.assertEqual(region.cells, ())
        self.assertEqual(self.fake.region_latlng(region, count=3), [])
        with self.assertRaises(ValueError):
            self.fake.region_latlng(region)

    def test_tuple_rectangles_and_default_region(self):
        region = GeoRegion(include=[(-10, 10, -180, 180)])
        self.assertEqual(region.include, (GeoRect(-10, 10, -180, 180),))

        for _ in range(20):
            lat, lon = self.fake.latlng_in_region()
            self.assertTrue(-90 <= float(lat) <= 90)
            self.assertTrue(-180 <= float(lon) <= 180)

    def test_area_union_and_exclusion(self):
        included = GeoRegion(
            [
                GeoRect(-10, 10, 0, 90),
                GeoRect(-10, 10, 60, 120),
            ]
        )
        expected_longitude = math.radians(120)
        expected_latitude = math.sin(math.radians(10)) - math.sin(math.radians(-10))
        self.assertAlmostEqual(included.solid_angle, expected_longitude * expected_latitude)

        holed = GeoRegion(
            include=[GeoRect(-10, 10, -180, 180)],
            exclude=[GeoRect(-10, 10, 0, 180)],
        )
        expected_holed_longitude = math.radians(180)
        self.assertAlmostEqual(holed.solid_angle, expected_holed_longitude * expected_latitude)

    def test_antimeridian_rectangle(self):
        region = GeoRegion([GeoRect(-10, 10, 170, -170)])
        self.assertTrue(region.contains(0, 175))
        self.assertTrue(region.contains(0, -175))
        self.assertFalse(region.contains(0, 0))
        self.assertAlmostEqual(
            region.solid_angle,
            math.radians(20)
            * (math.sin(math.radians(10)) - math.sin(math.radians(-10))),
        )

        for _ in range(100):
            lat, lon = self.fake.latlng_in_region(region)
            self.assertTrue(-10 <= lat <= 10)
            self.assertTrue(float(lon) >= 170 or float(lon) <= -170)

    def test_pole_regions(self):
        north = GeoRegion([GeoRect(80, 90, -180, 180)])
        south = GeoRegion([GeoRect(-90, -80, -180, 180)])
        expected = math.radians(360) * (1 - math.sin(math.radians(80)))
        self.assertAlmostEqual(north.solid_angle, expected)
        self.assertAlmostEqual(south.solid_angle, expected)
        self.assertAlmostEqual(north.area_km2, expected * EARTH_RADIUS_KM**2)
        self.assertTrue(north.contains(90, 0))
        self.assertTrue(south.contains(-90, 180))

        for _ in range(100):
            lat, _ = self.fake.latlng_in_region(north)
            self.assertTrue(80 <= float(lat) <= 90)

    def test_region_samples_match_area_proportions(self):
        equatorial = GeoRect(-10, 10, -180, 180)
        high_latitude = GeoRect(60, 80, -180, 180)
        crossing_with_hole = GeoRect(-30, -20, 170, -170)
        hole = GeoRect(-30, -25, 175, -175)
        region = GeoRegion(
            include=[equatorial, high_latitude, crossing_with_hole],
            exclude=[hole],
        )

        areas = {
            "equatorial": GeoRegion([equatorial]).solid_angle,
            "high_latitude": GeoRegion([high_latitude]).solid_angle,
            "crossing": GeoRegion([crossing_with_hole], [hole]).solid_angle,
        }
        total_area = sum(areas.values())
        self.assertAlmostEqual(region.solid_angle, total_area)

        samples = self.fake.latlngs_in_region(region, 20000)
        counts = {"equatorial": 0, "high_latitude": 0, "crossing": 0}
        for lat, lon in samples:
            self.assertTrue(region.contains(float(lat), float(lon)))
            self.assertFalse(hole.contains(float(lat), float(lon)))
            if -10 <= float(lat) <= 10:
                counts["equatorial"] += 1
            elif 60 <= float(lat) <= 80:
                counts["high_latitude"] += 1
            elif -30 <= float(lat) <= -20 and (float(lon) >= 170 or float(lon) <= -170):
                counts["crossing"] += 1

        self.assertEqual(sum(counts.values()), len(samples))
        for name, area in areas.items():
            with self.subTest(zone=name):
                self.assertAlmostEqual(
                    counts[name] / len(samples),
                    area / total_area,
                    delta=0.015,
                )
