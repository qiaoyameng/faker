import math
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple, Union


EARTH_RADIUS_KM = 6371.0088


class GeoRect:
    """An axis-aligned rectangle bounded by latitude and longitude lines."""

    __slots__ = ("_min_lat", "_max_lat", "_min_lon", "_max_lon")

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def __new__(
        cls,
        min_lat: Any,
        max_lat: Any,
        min_lon: Any,
        max_lon: Any,
    ) -> "GeoRect":
        min_lat = _validate_coordinate(min_lat, -90, 90, "min_lat")
        max_lat = _validate_coordinate(max_lat, -90, 90, "max_lat")
        min_lon = _validate_coordinate(min_lon, -180, 180, "min_lon")
        max_lon = _validate_coordinate(max_lon, -180, 180, "max_lon")

        if min_lat > max_lat:
            raise ValueError("min_lat cannot be greater than max_lat")

        rect = object.__new__(cls)
        rect._min_lat = min_lat
        rect._max_lat = max_lat
        rect._min_lon = min_lon
        rect._max_lon = max_lon
        return rect

    def __iter__(self) -> Iterator[float]:
        return iter((self.min_lat, self.max_lat, self.min_lon, self.max_lon))

    def __repr__(self) -> str:
        return (
            f"GeoRect(min_lat={self.min_lat}, max_lat={self.max_lat}, "
            f"min_lon={self.min_lon}, max_lon={self.max_lon})"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GeoRect) and tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    @property
    def min_lat(self) -> float:
        return self._min_lat

    @property
    def max_lat(self) -> float:
        return self._max_lat

    @property
    def min_lon(self) -> float:
        return self._min_lon

    @property
    def max_lon(self) -> float:
        return self._max_lon

    @property
    def longitude_width(self) -> float:
        if self.min_lon <= self.max_lon:
            return self.max_lon - self.min_lon
        return 360 + self.max_lon - self.min_lon

    def longitude_intervals(self) -> Tuple[Tuple[float, float], ...]:
        """Split an antimeridian-spanning rectangle into linear intervals."""

        if self.min_lon <= self.max_lon:
            if self.max_lon == self.min_lon:
                return ()
            return ((self.min_lon, self.max_lon),)
        return (
            (self.min_lon, 180),
            (-180, self.max_lon),
        )

    def contains(self, lat: float, lon: float) -> bool:
        if not self.min_lat <= lat <= self.max_lat:
            return False

        if (lat == 90 and self.max_lat == 90) or (lat == -90 and self.min_lat == -90):
            return True

        if self.min_lon <= self.max_lon:
            return self.min_lon <= lon <= self.max_lon
        return lon >= self.min_lon or lon <= self.max_lon


RectLike = Union[GeoRect, Sequence[Any]]
Cell = Tuple[float, float, float, float]
Interval = Tuple[float, float]


class GeoRegion:
    """A union of latitude/longitude rectangles with rectangular holes."""

    def __init__(
        self,
        include: Optional[Iterable[RectLike]] = None,
        exclude: Optional[Iterable[RectLike]] = None,
    ) -> None:
        self.include = tuple(_as_rect(rect) for rect in include or ())
        self.exclude = tuple(_as_rect(rect) for rect in exclude or ())
        self._cells = _partition(self.include, self.exclude)
        self._solid_angle = sum(solid_angle(cell) for cell in self._cells)

    @property
    def solid_angle(self) -> float:
        """Region area as a fraction-free solid angle in steradians."""

        return self._solid_angle

    @property
    def area_km2(self) -> float:
        """Spherical area in square kilometres."""

        return self._solid_angle * EARTH_RADIUS_KM**2

    @property
    def is_empty(self) -> bool:
        return self._solid_angle == 0

    @property
    def cells(self) -> Tuple[Cell, ...]:
        """Non-overlapping cells whose union is this region."""

        return self._cells

    def contains(self, lat: float, lon: float) -> bool:
        included = any(rect.contains(lat, lon) for rect in self.include)
        excluded = any(rect.contains(lat, lon) for rect in self.exclude)
        return included and not excluded


def _validate_coordinate(value: Any, minimum: float, maximum: float, name: str) -> float:
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise ValueError(f"{name} must be finite")
    if not minimum <= coordinate <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return coordinate


def _as_rect(value: RectLike) -> GeoRect:
    if isinstance(value, GeoRect):
        return value
    if isinstance(value, Sequence) and len(value) == 4:
        return GeoRect(*value)
    raise TypeError("region rectangles must be GeoRect instances or four-value sequences")


def _partition(include: Iterable[GeoRect], exclude: Iterable[GeoRect]) -> Tuple[Cell, ...]:
    rectangles = tuple(include) + tuple(exclude)
    latitude_cuts = {-90.0, 90.0}
    for rect in rectangles:
        latitude_cuts.add(rect.min_lat)
        latitude_cuts.add(rect.max_lat)

    cells = []
    cuts = sorted(latitude_cuts)
    for lat_min, lat_max in zip(cuts, cuts[1:]):
        included = _merge_intervals(
            interval
            for rect in include
            if rect.min_lat < lat_max and rect.max_lat > lat_min
            for interval in rect.longitude_intervals()
        )
        holes = _merge_intervals(
            interval
            for rect in exclude
            if rect.min_lat < lat_max and rect.max_lat > lat_min
            for interval in rect.longitude_intervals()
        )

        for lon_min, lon_max in included:
            for interval in _subtract_intervals((lon_min, lon_max), holes):
                cells.append((lat_min, lat_max, interval[0], interval[1]))

    return tuple(cells)


def _merge_intervals(intervals: Iterable[Interval]) -> Tuple[Interval, ...]:
    merged = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return tuple((start, end) for start, end in merged)


def _subtract_intervals(interval: Interval, holes: Iterable[Interval]) -> Tuple[Interval, ...]:
    remaining = [interval]
    for hole_start, hole_end in holes:
        next_remaining = []
        for start, end in remaining:
            overlap_start = max(start, hole_start)
            overlap_end = min(end, hole_end)
            if overlap_start >= overlap_end:
                next_remaining.append((start, end))
                continue

            if start < overlap_start:
                next_remaining.append((start, overlap_start))
            if overlap_end < end:
                next_remaining.append((overlap_end, end))
        remaining = next_remaining
    return tuple(remaining)


def solid_angle(cell: Cell) -> float:
    lat_min, lat_max, lon_min, lon_max = cell
    longitude_span = math.radians(lon_max - lon_min)
    return longitude_span * (
        math.sin(math.radians(lat_max)) - math.sin(math.radians(lat_min))
    )
