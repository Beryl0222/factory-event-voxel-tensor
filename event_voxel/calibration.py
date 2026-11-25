"""Calibration versions, projection and spatial ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class OwnerRegion:
    sensor: str
    y0: float
    y1: float
    x0: float
    x1: float
    priority: int = 0

    def contains(self, y: float, x: float) -> bool:
        return self.y0 <= y < self.y1 and self.x0 <= x < self.x1

    @classmethod
    def from_box(cls, sensor: str, box: SequenceLike, priority: int = 0) -> "OwnerRegion":
        if len(box) != 4:
            raise ValueError(f"owner box must be [y0,y1,x0,x1], got {box!r}")
        return cls(sensor, float(box[0]), float(box[1]), float(box[2]), float(box[3]), priority)


SequenceLike = tuple | list | np.ndarray


@dataclass(frozen=True)
class Calibration:
    sensor: str
    version: str
    effective_lo: float
    effective_hi: float
    scale_y: float
    scale_x: float
    shift_y: float
    shift_x: float
    owner: OwnerRegion
    offset: float = 0.0
    line_delay: float = 0.0

    def effective_contains(self, raw_t: float) -> bool:
        return self.effective_lo <= raw_t < self.effective_hi

    def project_y(self, raw_y: Any) -> Any:
        return raw_y * self.scale_y + self.shift_y

    def project_x(self, raw_x: Any) -> Any:
        return raw_x * self.scale_x + self.shift_x

    def corrected_time(self, raw_t: Any, raw_y: Any) -> Any:
        # Important: line delay uses the *original* row, not projected row.
        return raw_t + self.offset + raw_y * self.line_delay

    def owns(self, projected_y: float, projected_x: float) -> bool:
        return self.owner.contains(projected_y, projected_x)

    def raw_owned_row_bounds(self, snap_tol: float = 1e-9) -> tuple[float, float]:
        """Continuous raw-row interval whose projection is in the owner box."""

        def invert(bound: float) -> float:
            return (bound - self.shift_y) / self.scale_y

        a = invert(self.owner.y0)
        b = invert(self.owner.y1)
        return (min(a, b), max(a, b))

    def raw_owned_rows(self, snap_tol: float = 1e-9) -> np.ndarray:
        """Integer original rows that can project into this owner region."""

        lo, hi = self.raw_owned_row_bounds()

        def snap_near_int(v: float) -> float:
            r = round(v)
            if abs(v - r) <= snap_tol:
                return float(r)
            return v

        lo = snap_near_int(lo)
        hi = snap_near_int(hi)
        first = math.ceil(lo - snap_tol)
        stop = math.ceil(hi - snap_tol)
        if stop < first:
            stop = first
        return np.arange(first, stop, dtype=np.int64)

    def delay_extrema(self, snap_tol: float = 1e-9) -> tuple[float, float]:
        rows = self.raw_owned_rows(snap_tol)
        if rows.size == 0:
            lo, hi = self.raw_owned_row_bounds(snap_tol)
            candidates = np.asarray([lo, hi], dtype=np.float64)
        else:
            candidates = rows.astype(np.float64)
        delays = self.offset + candidates * self.line_delay
        return float(np.min(delays)), float(np.max(delays))

    def completeness_horizon(self, progress_local: float, snap_tol: float = 1e-9) -> float:
        """Exclusive global time through which all events are promised.

        Progress ``p`` promises raw events with ``t < p``.  For row ``r`` the
        correction constant is ``offset + r*line_delay``.  The slowest row
        (minimum correction constant) determines the common global horizon.
        The calibration's half-open effective interval also caps the horizon
        until a following calibration is supplied.
        """

        min_delay, _ = self.delay_extrema(snap_tol)
        promised_local = min(float(progress_local), float(self.effective_hi))
        return promised_local + min_delay


class CalibrationBook:
    """Versioned calibrations for all read heads."""

    def __init__(
        self,
        versions: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        *,
        active_version: str | None = None,
        overlap_owner: str | Mapping[tuple[float, float, float, float], str] | None = None,
    ) -> None:
        self._raw_versions: dict[str, dict[str, Any]] = {
            str(v): dict(sensors) for v, sensors in (versions or {}).items()
        }
        if active_version is None and self._raw_versions:
            active_version = next(iter(self._raw_versions))
        self.active_version: str | None = None if active_version is None else str(active_version)
        self.version_history: list[tuple[int, str | None, str]] = []
        self._revision = 0
        self.overlap_owner = overlap_owner

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def versions(self) -> tuple[str, ...]:
        return tuple(self._raw_versions)

    def activate(self, version: str, *, sensors: Iterable[str] | None = None) -> None:
        version = str(version)
        if version not in self._raw_versions:
            raise KeyError(f"unknown calibration version {version!r}; have {self.versions}")
        # The checked-in exchange uses one global active correction.  Keeping
        # this hook allows a future per-head correction without changing API.
        self.active_version = version
        self._revision += 1
        self.version_history.append((self._revision, version, ",".join(sensors or ())))

    def add_version(self, version: str, calibrations: Mapping[str, Mapping[str, Any]]) -> None:
        self._raw_versions[str(version)] = {k: dict(v) for k, v in calibrations.items()}

    def _raw_cal(self, sensor: str, version: str | None = None) -> Mapping[str, Any]:
        ver = str(version or self.active_version)
        if ver is None:
            raise ValueError("no active calibration version")
        if ver not in self._raw_versions:
            raise KeyError(f"unknown calibration version {ver!r}")
        sensors = self._raw_versions[ver]
        if sensor not in sensors:
            raise KeyError(f"sensor {sensor!r} absent from calibration {ver!r}")
        return sensors[sensor]

    def get(
        self,
        sensor: str,
        raw_t: float | None = None,
        *,
        version: str | None = None,
    ) -> Calibration:
        ver = str(version or self.active_version)
        raw = self._raw_cal(sensor, ver)
        effective = raw.get("effective", raw.get("effective_local", (-np.inf, np.inf)))
        scale = raw.get("scale", raw.get("scale_yx", (1.0, 1.0)))
        shift = raw.get("shift", raw.get("shift_yx", (0.0, 0.0)))
        owner_raw = raw.get("owner", raw.get("owner_box"))
        if owner_raw is None:
            raise ValueError(f"calibration {ver}/{sensor} has no owner box")
        owner = OwnerRegion.from_box(sensor, owner_raw)
        cal = Calibration(
            sensor=sensor,
            version=ver,
            effective_lo=float(effective[0]),
            effective_hi=float(effective[1]),
            scale_y=float(scale[0]),
            scale_x=float(scale[1]),
            shift_y=float(shift[0]),
            shift_x=float(shift[1]),
            owner=owner,
            offset=float(raw.get("offset", 0.0)),
            line_delay=float(raw.get("line_delay", raw.get("row_delay", 0.0))),
        )
        if raw_t is not None and not cal.effective_contains(float(raw_t)):
            # Search other supplied versions for an effective calibration.
            candidates = []
            for other_ver, sensors in self._raw_versions.items():
                if sensor not in sensors:
                    continue
                try:
                    other = self.get(sensor, None, version=other_ver)
                except Exception:
                    continue
                if other.effective_contains(float(raw_t)):
                    candidates.append(other)
            if len(candidates) == 1:
                return candidates[0]
            raise ValueError(
                f"raw t={raw_t} is outside effective range {effective} for "
                f"{sensor}/{ver}; matching versions={[c.version for c in candidates]}"
            )
        return cal

    def owner_regions(self, version: str | None = None) -> list[OwnerRegion]:
        ver = str(version or self.active_version)
        out: list[OwnerRegion] = []
        for sensor, raw in self._raw_versions[ver].items():
            box = raw.get("owner", raw.get("owner_box"))
            if box is not None:
                out.append(OwnerRegion.from_box(sensor, box))
        return out

    def owner_at(
        self, y: float, x: float, *, version: str | None = None
    ) -> str | None:
        """Return the spatial owner at a projected coordinate."""

        matches = [r for r in self.owner_regions(version) if r.contains(y, x)]
        if not matches:
            return None
        if len({r.sensor for r in matches}) == 1:
            return matches[0].sensor
        if isinstance(self.overlap_owner, str):
            return self.overlap_owner
        # Explicit priority can be supplied via OwnerRegion.priority.
        return sorted(matches, key=lambda r: r.priority, reverse=True)[0].sensor

    def owner_array(
        self, y: np.ndarray, x: np.ndarray, *, version: str | None = None
    ) -> np.ndarray:
        """Vectorized spatial owner lookup for coordinate arrays."""

        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        owners = np.full(y.shape, None, dtype=object)
        claimed = np.zeros(y.shape, dtype=bool)
        regions = sorted(
            self.owner_regions(version), key=lambda r: r.priority, reverse=True
        )
        for region in regions:
            matches = (
                (y >= region.y0)
                & (y < region.y1)
                & (x >= region.x0)
                & (x < region.x1)
            )
            assign = matches & ~claimed
            owners[assign] = region.sensor
            claimed |= matches

        if isinstance(self.overlap_owner, str):
            counts = np.zeros(y.shape, dtype=np.int16)
            for region in self.owner_regions(version):
                matches = (
                    (y >= region.y0)
                    & (y < region.y1)
                    & (x >= region.x0)
                    & (x < region.x1)
                )
                counts += matches
            owners[counts > 1] = self.overlap_owner
        return owners

    def coverage_map(
        self,
        H: int,
        W: int,
        *,
        origin_y: float = 0.0,
        origin_x: float = 0.0,
        step_y: float = 1.0,
        step_x: float = 1.0,
        version: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """Rasterize half-open owner boxes onto grid points."""

        ys = origin_y + np.arange(H, dtype=np.float64) * step_y
        xs = origin_x + np.arange(W, dtype=np.float64) * step_x
        coverage = np.zeros((H, W), dtype=bool)
        owner_index = np.full((H, W), -1, dtype=np.int16)
        sensor_ids: dict[str, int] = {}

        for region in self.owner_regions(version):
            sensor = region.sensor
            if sensor not in sensor_ids:
                sensor_ids[sensor] = len(sensor_ids)
            sid = sensor_ids[sensor]
            ym = (ys >= region.y0) & (ys < region.y1)
            xm = (xs >= region.x0) & (xs < region.x1)
            mask = ym[:, None] & xm[None, :]
            coverage |= mask
            # Preserve an already assigned higher-priority overlap owner.
            empty = owner_index < 0
            owner_index[mask & empty] = sid
            if self.overlap_owner == sensor:
                owner_index[mask] = sid
        return coverage, owner_index, sensor_ids
