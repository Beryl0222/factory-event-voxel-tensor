"""Streaming ingestion, versioned revisions and inference handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Iterable, Iterator, Mapping
import uuid

import numpy as np

from .calibration import CalibrationBook
from .config import VoxelConfig
from .voxelizer import project_events, splat_prepared, target_window_indices
from .windows import ActiveWindowLimitError, CallableArray, WindowBank, WindowState


class WindowNotReadyError(RuntimeError):
    """Raised when a caller tries to hand off a non-inferable window."""


class ProgressBoard(dict):
    """Mapping of latest local progress that is also callable.

    ``stream.progress["east"]`` reads the raw local promise while
    ``stream.progress("east", 20)`` submits a progress update.
    """

    def __init__(self, owner: "StreamingVoxelizer") -> None:
        super().__init__()
        self._owner = owner

    def __call__(self, sensor: str | None = None, value: float | int | None = None):
        if sensor is None:
            return dict(self)
        if value is None:
            return self.get(str(sensor))
        return self._owner.update_progress(str(sensor), value)



@dataclass(slots=True, frozen=True)
class EventKey:
    sensor: str
    event_id: int


@dataclass(slots=True)
class EventRecord:
    key: EventKey
    y: int
    x: int
    t_local: int
    polarity: int
    batch_id: str | None = None


@dataclass
class IngestReport:
    sensor: str
    received: int = 0
    accepted: int = 0
    duplicates: int = 0
    owner_rejected: int = 0
    calibration_pending: int = 0
    boundary_loss: float = 0.0
    window_indices: list[int] = field(default_factory=list)
    duplicate_ids: list[int] = field(default_factory=list)
    rejected_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sensor": self.sensor,
            "received": self.received,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "owner_rejected": self.owner_rejected,
            "calibration_pending": self.calibration_pending,
            "boundary_loss": self.boundary_loss,
            "window_indices": self.window_indices,
            "duplicate_ids": self.duplicate_ids,
            "rejected_ids": self.rejected_ids,
        }


class ConflictMarker:
    """Shared conflict state visible to an immutable handoff delivery."""

    def __init__(
        self,
        window_index: int,
        delivery_id: str,
        ready_at_handoff: bool = True,
    ) -> None:
        self.window_index = int(window_index)
        self.delivery_id = delivery_id
        self.ready_at_handoff = bool(ready_at_handoff)
        self.conflicts: list[dict[str, Any]] = []

    @property
    def conflicted(self) -> bool:
        return bool(self.conflicts)

    # Common synonym.
    has_conflict = conflicted

    def add(self, kind: str, **details: Any) -> dict[str, Any]:
        entry = {"kind": kind, **details}
        self.conflicts.append(entry)
        return entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "delivery_id": self.delivery_id,
            "conflicted": self.conflicted,
            "conflicts": list(self.conflicts),
        }


@dataclass
class WindowDelivery:
    """Immutable tensor package transferred to inference."""

    window_index: int
    delivery_id: str
    revision: int
    calibration_version: str | None
    voxels: np.ndarray
    coverage: np.ndarray
    boundary_loss: float
    boundary_loss_by_time: np.ndarray
    progress: dict[str, float]
    horizons: dict[str, float]
    window_end: float
    ready_at_handoff: bool
    marker: ConflictMarker
    dtype: Any = None

    def __post_init__(self) -> None:
        self.voxels.setflags(write=False)
        self.coverage.setflags(write=False)
        self.boundary_loss_by_time.setflags(write=False)
        if self.dtype is None:
            self.dtype = self.voxels.dtype

    @property
    def conflict(self) -> bool:
        return self.marker.conflicted

    @property
    def conflicted(self) -> bool:
        return self.marker.conflicted

    @property
    def conflicts(self) -> list[dict[str, Any]]:
        return self.marker.conflicts

    @property
    def usable(self) -> bool:
        """An original delivery is usable only until a revision conflict."""

        return self.ready_at_handoff and not self.conflicted

    @property
    def boundary_allocation_loss(self) -> float:
        return float(self.boundary_loss)

    @property
    def clipped_weight(self) -> float:
        return float(self.boundary_loss)

    @property
    def positive(self) -> np.ndarray:
        return self.voxels[1]

    @property
    def negative(self) -> np.ndarray:
        return self.voxels[0]

    @property
    def voxels_tchw(self) -> np.ndarray:
        """Time-first layout: ``(T, 2, H, W)``."""

        return np.transpose(self.voxels, (1, 0, 2, 3))

    @property
    def voxels_thwc(self) -> np.ndarray:
        """Channels-last layout: ``(T, H, W, 2)``."""

        return np.transpose(self.voxels, (1, 2, 3, 0))

    @property
    def signed_voxels_thw(self) -> np.ndarray:
        """Signed time-first grid: positive weights minus negative weights."""

        return self.voxels[1] - self.voxels[0]

    @property
    def effective_coverage(self) -> CallableArray:
        if not self.ready_at_handoff or self.conflicted:
            return CallableArray(np.zeros_like(self.coverage))
        return CallableArray(self.coverage)

    @property
    def zero_coverage(self) -> np.ndarray:
        return self.effective_coverage

    @property
    def coverage_status(self) -> np.ndarray:
        """0=uncovered, 1=valid zero-transition cover, -2=stale handed rev."""

        status = np.zeros(self.coverage.shape, dtype=np.int8)
        status[self.coverage] = 1 if self.usable else -2
        return status

    @property
    def mask(self) -> np.ndarray:
        return self.effective_coverage

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "delivery_id": self.delivery_id,
            "revision": self.revision,
            "calibration_version": self.calibration_version,
            "voxels": self.voxels,
            "coverage": self.coverage,
            "effective_coverage": self.effective_coverage,
            "zero_coverage": self.zero_coverage,
            "coverage_status": self.coverage_status,
            "boundary_loss": self.boundary_loss,
            "boundary_loss_by_time": self.boundary_loss_by_time,
            "progress": dict(self.progress),
            "horizons": dict(self.horizons),
            "window_end": self.window_end,
            "ready_at_handoff": self.ready_at_handoff,
            "usable": self.usable,
            "conflict": self.conflict,
            "conflicts": list(self.conflicts),
        }

    def __iter__(self) -> Iterator[Any]:
        # Convenience for ``voxels, coverage = claim(...)``.
        yield self.voxels
        yield self.coverage

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass
class WindowView:
    """Current operator state for one active window."""

    index: int
    revision: int
    calibration_version: str | None
    state: WindowState
    complete: bool
    usable: bool
    horizons: dict[str, float]
    window_end: float
    reasons: list[str]
    delivery: WindowDelivery | None = None
    conflict: bool = False
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def voxels(self) -> np.ndarray:
        return self.state.voxels

    @property
    def current_voxels(self) -> np.ndarray:
        return self.state.voxels

    @property
    def original_voxels(self) -> np.ndarray | None:
        return None if self.delivery is None else self.delivery.voxels

    @property
    def tensor(self) -> np.ndarray:
        return self.state.voxels

    @property
    def positive(self) -> np.ndarray:
        return self.state.voxels[1]

    @property
    def negative(self) -> np.ndarray:
        return self.state.voxels[0]

    @property
    def voxels_tchw(self) -> np.ndarray:
        return np.transpose(self.voxels, (1, 0, 2, 3))

    @property
    def voxels_thwc(self) -> np.ndarray:
        return np.transpose(self.voxels, (1, 2, 3, 0))

    @property
    def signed_voxels_thw(self) -> np.ndarray:
        return self.voxels[1] - self.voxels[0]

    @property
    def coverage(self) -> np.ndarray:
        return self.state.coverage

    @property
    def spatial_coverage(self) -> np.ndarray:
        return self.state.coverage

    @property
    def effective_coverage(self) -> CallableArray:
        return CallableArray(self.state.effective_coverage(self.complete))

    @property
    def zero_coverage(self) -> np.ndarray:
        return self.effective_coverage

    @property
    def coverage_status(self) -> np.ndarray:
        """0=uncovered, 1=valid zero-transition cover, -1=progress incomplete."""

        status = np.zeros(self.coverage.shape, dtype=np.int8)
        status[self.coverage] = 1 if self.complete else -1
        return status

    @property
    def mask(self) -> np.ndarray:
        return self.effective_coverage

    @property
    def boundary_loss(self) -> float:
        return self.state.boundary_loss

    @property
    def boundary_loss_by_time(self) -> np.ndarray:
        return self.state.boundary_loss_time

    @property
    def boundary_allocation_loss(self) -> float:
        return self.boundary_loss

    @property
    def clipped_weight(self) -> float:
        return self.boundary_loss

    @property
    def ready(self) -> bool:
        return self.complete

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_index": self.index,
            "revision": self.revision,
            "calibration_version": self.calibration_version,
            "voxels": self.voxels,
            "coverage": self.coverage,
            "effective_coverage": self.effective_coverage,
            "zero_coverage": self.zero_coverage,
            "coverage_status": self.coverage_status,
            "boundary_loss": self.boundary_loss,
            "boundary_loss_by_time": self.boundary_loss_by_time,
            "complete": self.complete,
            "ready": self.ready,
            "usable": self.usable,
            "horizons": dict(self.horizons),
            "window_end": self.window_end,
            "reasons": list(self.reasons),
            "conflict": self.conflict,
            "conflicts": list(self.conflicts),
            "delivery_id": None if self.delivery is None else self.delivery.delivery_id,
        }

    def __iter__(self) -> Iterator[Any]:
        yield self.voxels
        yield self.coverage

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _chunks(items: Mapping[Any, EventRecord] | list[EventRecord], size: int):
    if isinstance(items, Mapping):
        iterator = iter(items.values())
    else:
        iterator = iter(items)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


class StreamingVoxelizer:
    """Event-sourced bounded window operator and handoff coordinator."""

    def __init__(
        self,
        config: VoxelConfig | Mapping[str, Any],
        calibrations: CalibrationBook | Mapping[str, Any] | None = None,
        *,
        active_calibration: str | None = None,
        overlap_owner: str | Mapping[tuple[float, float, float, float], str] | None = None,
    ) -> None:
        if isinstance(config, Mapping):
            config = VoxelConfig.from_mapping(config)
        self.config = config
        if isinstance(calibrations, CalibrationBook):
            self.book = calibrations
        else:
            versions = None
            if calibrations is not None:
                cal_map = dict(calibrations)  # type: ignore[arg-type]
                if overlap_owner is None and "overlap_owner" in cal_map:
                    overlap_owner = cal_map["overlap_owner"]
                if "calibrations" in cal_map:
                    versions = cal_map["calibrations"]
                elif "records" in cal_map and isinstance(cal_map["records"], Mapping):
                    versions = cal_map["records"].get("calibrations")
                elif "owner" in cal_map or "owner_box" in cal_map:
                    # A single calibration mapping.
                    versions = {"default": {"default": cal_map}}
                    if active_calibration is None:
                        active_calibration = "default"
                elif cal_map and all(
                    isinstance(v, Mapping) and ("owner" in v or "owner_box" in v)
                    for v in cal_map.values()
                ):
                    # sensor -> calibration, without explicit versions.
                    versions = {"default": dict(cal_map)}
                    if active_calibration is None:
                        active_calibration = "default"
                else:
                    # Accept either {"v1": {sensor: cal}} or records containing
                    # a "calibrations" key.
                    versions = cal_map
            self.book = CalibrationBook(
                versions,
                active_version=active_calibration,
                overlap_owner=overlap_owner,
            )
        if active_calibration is not None:
            self.book.activate(active_calibration)

        self.bank = WindowBank(config)
        # Compact retained raw batches (int64 columns) avoid a second Python
        # object graph for every event.  The identity sets remain exact.
        self.event_batches: dict[str, list[np.ndarray]] = {}
        self.seen: dict[str, set[int]] = {}
        self.progress = ProgressBoard(self)
        self.progress_history: list[tuple[int, str, float | None, float]] = []
        self.submission_log: list[Any] = []
        self.ingest_reports: list[IngestReport] = []

        self.deliveries: dict[str, WindowDelivery] = {}
        self.delivery_by_window: dict[int, list[str]] = {}
        self.markers_by_window: dict[int, ConflictMarker] = {}
        self.next_window = 0
        self._sequence = 0
        self._coverage_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, int]]] = {}

    def retained_event_bytes(self) -> int:
        return int(sum(batch.nbytes for batches in self.event_batches.values() for batch in batches))

    def workspace_bytes(self, *, include_archived_deliveries: bool = True) -> int:
        total = self.bank.dense_bytes() + self.retained_event_bytes()
        if include_archived_deliveries:
            for delivery in self.deliveries.values():
                total += int(delivery.voxels.nbytes)
                total += int(delivery.coverage.nbytes)
                total += int(delivery.boundary_loss_by_time.nbytes)
        return total

    # ------------------------------------------------------------------
    # Configuration/calibration/progress inputs
    # ------------------------------------------------------------------
    @property
    def active_calibration(self) -> str | None:
        return self.book.active_version

    def activate_calibration(self, version: str) -> list[int]:
        """Activate a correction and rebuild all retained active windows."""

        old_version = self.book.active_version
        old_revision = self.book.revision
        self.book.activate(version)
        affected = self._rebuild_active_windows(
            reason="calibration",
            old_version=old_version,
            new_version=version,
            old_revision=old_revision,
        )
        self.submission_log.append(["calibration", version])
        return affected

    # Friendly alias matching the exchange record verb.
    calibration = activate_calibration

    def update_progress(self, sensor: str, local_timestamp: float | int) -> WindowView | None:
        """Set the latest local progress promise; lower values are corrections."""

        self._sequence += 1
        old = self.progress.get(sensor)
        value = float(local_timestamp)
        self.progress[sensor] = value
        self.progress_history.append((self._sequence, sensor, old, value))
        self.submission_log.append(["progress", sensor, local_timestamp])

        # A backwards/changed promise can turn a handed sealed window into a
        # stale revision.  Check all known delivery windows.
        for window_index, delivery_ids in list(self.delivery_by_window.items()):
            end = self.config.window_end_time(window_index)
            sealed_now, horizons, _ = self._readiness(window_index, end=end)
            marker = self._latest_marker(window_index)
            if marker is None:
                continue
            if marker.ready_at_handoff and not sealed_now:
                marker.add(
                    "progress_revision",
                    sensor=sensor,
                    old_local_progress=old,
                    new_local_progress=value,
                    horizons=horizons,
                    window_end=end,
                )
        if self.next_window in self.bank:
            return self.window(self.next_window)
        return None

    progress_update = update_progress
    on_progress = update_progress

    def add_calibration(
        self, version: str, calibrations: Mapping[str, Mapping[str, Any]]
    ) -> None:
        self.book.add_version(version, calibrations)

    def on_calibration(self, version: str) -> list[int]:
        return self.activate_calibration(version)

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------
    def ingest_events(
        self,
        sensor: str,
        events: np.ndarray | list[list[Any]],
        batch_id: str | None = None,
    ) -> IngestReport:
        self._sequence += 1
        sensor = str(sensor)
        array = np.asarray(events)
        if array.ndim != 2 or array.shape[1] < 5:
            raise ValueError("events must have shape (N, 5+) with id,y,x,t,polarity")

        report = IngestReport(sensor=sensor, received=int(array.shape[0]))
        seen = self.seen.setdefault(sensor, set())

        # De-duplicate against committed identities and within this batch, but
        # do not commit the new identities until window capacity is guaranteed.
        keep = np.ones(int(array.shape[0]), dtype=bool)
        batch_ids: set[int] = set()
        for i, row in enumerate(array):
            event_id = int(row[0])
            if event_id in seen or event_id in batch_ids:
                keep[i] = False
                report.duplicates += 1
                report.duplicate_ids.append(event_id)
            else:
                batch_ids.add(event_id)

        if np.any(keep):
            # Compact unique-event copy; this is the retained event source for
            # calibration rebuilds and is at most 40 bytes/event for int64 x5.
            new_array = np.ascontiguousarray(array[keep], dtype=np.int64)
            required = self._preflight_capacity(sensor, new_array)
            limit = self.config.max_active_windows
            if limit is not None:
                existing = set(self.bank.indices())
                new_windows = required - existing
                if len(existing) + len(new_windows) > limit:
                    raise ActiveWindowLimitError(
                        f"ingest needs windows {sorted(new_windows)}; active limit "
                        f"{limit} reached by {sorted(existing)}. Claim/ack or drop first"
                    )

            seen.update(int(v) for v in new_array[:, 0])
            self.event_batches.setdefault(sensor, []).append(new_array)

            before = set(self.bank.indices())
            splat_report = self._splat_sensor_array(sensor, new_array, report)
            report.boundary_loss = float(splat_report.get("boundary_loss", 0.0))
            touched = set(self.bank.indices()) - before
            report.window_indices = sorted(
                set(report.window_indices) | touched | splat_report.get("windows_touched", set())
            )
            for win in self.bank.states():
                self._apply_coverage(win.index)

        self.ingest_reports.append(report)
        self.submission_log.append(["events", sensor, batch_id])
        return report

    add_events = ingest_events
    on_events = ingest_events
    process_events = ingest_events
    submit_events = ingest_events
    add_batch = ingest_events
    push_events = ingest_events

    def _preflight_capacity(self, sensor: str, array: np.ndarray) -> set[int]:
        """Calculate target windows without allocating or mutating state."""

        targets: set[int] = set()
        cal, version_groups, _ = self._calibration_for_rows(sensor, array)
        if cal is not None:
            prepared = project_events(array, cal, snap_tol=self.config.snap_tol)
            prepared = self._apply_global_ownership(sensor, prepared, cal.version)
            targets |= target_window_indices(
                prepared, self.config, valid=prepared["valid"]
            )
            return targets

        if version_groups is None:
            return targets
        for version in set(v for v in version_groups if v is not None):
            sel = version_groups == version
            try:
                group_cal = self.book.get(sensor, version=version)
            except Exception:
                continue
            prepared = project_events(
                array[sel], group_cal, snap_tol=self.config.snap_tol
            )
            prepared = self._apply_global_ownership(
                sensor, prepared, group_cal.version
            )
            targets |= target_window_indices(
                prepared, self.config, valid=prepared["valid"]
            )
        return targets

    def rebuild(self) -> list[int]:
        """Rebuild all retained events under the active calibrations."""

        return self._rebuild_active_windows(reason="manual_rebuild")

    def _calibration_for_rows(
        self, sensor: str, array: np.ndarray
    ) -> tuple[CalibrationLike | None, np.ndarray | None, IngestReport | None]:
        """Return one calibration if all rows share it, else row grouping."""

        raw_t = array[:, 3].astype(np.float64)
        try:
            cal = self.book.get(sensor)
        except Exception:
            cal = None
        if cal is not None and np.all((raw_t >= cal.effective_lo) & (raw_t < cal.effective_hi)):
            return cal, None, None

        # Group by the unique effective calibration selected by the book.
        versions: list[str | None] = []
        for t in raw_t:
            try:
                versions.append(self.book.get(sensor, float(t)).version)
            except Exception:
                versions.append(None)
        return None, np.asarray(versions, dtype=object), None

    def _apply_global_ownership(
        self, sensor: str, prepared: dict[str, np.ndarray], version: str | None
    ) -> dict[str, np.ndarray]:
        """Resolve overlap ownership through the calibration book."""

        owners = self.book.owner_array(
            prepared["y"], prepared["x"], version=version
        )
        owner_valid = owners == sensor
        prepared["owner_valid"] = owner_valid
        prepared["valid"] = owner_valid & prepared["effective_valid"]
        return prepared

    def _splat_sensor_array(
        self, sensor: str, array: np.ndarray, report: IngestReport | None = None
    ) -> dict[str, Any]:
        cal, version_groups, _ = self._calibration_for_rows(sensor, array)
        total_stats: dict[str, Any] = {"boundary_loss": 0.0, "windows_touched": set()}

        if cal is not None:
            chunks = [array] if len(array) <= self.config.chunk_size else list(self._iter_array_chunks(array))
            for chunk in chunks:
                prepared = project_events(chunk, cal, snap_tol=self.config.snap_tol)
                prepared = self._apply_global_ownership(
                    sensor, prepared, cal.version
                )
                if report is not None:
                    owner_bad = ~prepared["owner_valid"]
                    report.owner_rejected += int(np.count_nonzero(owner_bad))
                    report.calibration_pending += int(
                        np.count_nonzero(~prepared["effective_valid"])
                    )
                    report.accepted += int(np.count_nonzero(prepared["valid"]))
                    if np.any(owner_bad):
                        report.rejected_ids.extend(
                            int(v) for v in prepared["event_id"][owner_bad]
                        )
                _, stats = splat_prepared(
                    prepared,
                    self.config,
                    self.bank,
                    valid=prepared["valid"],
                    version=cal.version,
                )
                total_stats["boundary_loss"] += stats["boundary_loss"]
                total_stats["windows_touched"] |= stats["windows_touched"]
            return total_stats

        # Mixed/missing calibration rows: group by selected version.  Missing
        # rows remain in EventLog and are picked up by a later rebuild.
        if version_groups is None:
            return total_stats
        for version in set(v for v in version_groups if v is not None):
            sel = version_groups == version
            try:
                cal = self.book.get(sensor, version=version)
            except Exception:
                continue
            prepared = project_events(array[sel], cal, snap_tol=self.config.snap_tol)
            prepared = self._apply_global_ownership(
                sensor, prepared, cal.version
            )
            if report is not None:
                owner_bad = ~prepared["owner_valid"]
                report.owner_rejected += int(np.count_nonzero(owner_bad))
                report.calibration_pending += int(
                    np.count_nonzero(~prepared["effective_valid"])
                )
                report.accepted += int(np.count_nonzero(prepared["valid"]))
                if np.any(owner_bad):
                    report.rejected_ids.extend(
                        int(v) for v in prepared["event_id"][owner_bad]
                    )
            _, stats = splat_prepared(
                prepared,
                self.config,
                self.bank,
                valid=prepared["valid"],
                version=cal.version,
            )
            total_stats["boundary_loss"] += stats["boundary_loss"]
            total_stats["windows_touched"] |= stats["windows_touched"]
        if report is not None:
            report.calibration_pending += int(np.count_nonzero(version_groups == None))  # noqa: E711
        return total_stats

    def _iter_array_chunks(self, array: np.ndarray) -> Iterator[np.ndarray]:
        n = len(array)
        step = int(self.config.chunk_size)
        for start in range(0, n, step):
            yield array[start : start + step]

    # ------------------------------------------------------------------
    # Coverage and rebuilding
    # ------------------------------------------------------------------
    def _coverage_for_active(self) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        version = self.book.active_version or "none"
        if version not in self._coverage_cache:
            if self.book.active_version is None:
                coverage = np.zeros((self.config.H, self.config.W), dtype=bool)
                owner_map = np.full((self.config.H, self.config.W), -1, dtype=np.int16)
                ids: dict[str, int] = {}
            else:
                coverage, owner_map, ids = self.book.coverage_map(
                    self.config.H,
                    self.config.W,
                    origin_y=self.config.origin_y,
                    origin_x=self.config.origin_x,
                    step_y=self.config.step_y,
                    step_x=self.config.step_x,
                    version=self.book.active_version,
                )
            self._coverage_cache[version] = (coverage, owner_map, ids)
        return self._coverage_cache[version]

    def _apply_coverage(self, window_index: int) -> None:
        if window_index not in self.bank:
            return
        state = self.bank[window_index]
        coverage, owner_map, ids = self._coverage_for_active()
        state.coverage = coverage.copy()
        state.owner_map = owner_map.copy()
        state.sensor_ids = dict(ids)
        state.revision = self.book.revision

    def _rebuild_active_windows(self, reason: str, **change: Any) -> list[int]:
        old_indices = set(self.bank.indices())

        # Fail atomically if the corrected projection would require more active
        # blocks than the configured real-time data space.  This preflight runs
        # before zeroing any existing tensor.
        target_indices: set[int] = set()
        for sensor, batches in self.event_batches.items():
            for batch in batches:
                for chunk in self._iter_array_chunks(batch):
                    target_indices |= self._preflight_capacity(sensor, chunk)
        retained_indices = old_indices | target_indices
        limit = self.config.max_active_windows
        if limit is not None and len(retained_indices) > limit:
            raise ActiveWindowLimitError(
                f"calibration rebuild needs windows {sorted(retained_indices)}, "
                f"exceeding max_active_windows={limit}; acknowledge/drop old windows first"
            )

        # Every previously handed revision gets a conflict marker, even if its
        # dense active state was transferred to inference and dropped.
        for window_index, delivery_ids in self.delivery_by_window.items():
            marker = self._latest_marker(window_index)
            if marker is not None:
                marker.add(
                    "calibration_revision",
                    reason=reason,
                    old_active_version=change.get("old_version"),
                    new_active_version=change.get("new_version"),
                    **{
                        k: v
                        for k, v in change.items()
                        if k not in {"old_version", "new_version"}
                    },
                )

        for state in self.bank.states():
            state.reset_values()

        # Re-sensor-by-sensor/batch to preserve vectorized calibration constants.
        for sensor, batches in self.event_batches.items():
            for batch in batches:
                for chunk in self._iter_array_chunks(batch):
                    self._splat_sensor_array(sensor, chunk, report=None)

        new_indices = set(self.bank.indices())
        for idx in old_indices | new_indices:
            if idx in self.bank:
                self._apply_coverage(idx)
                self.bank[idx].dirty = True
        for window_index, delivery_ids in self.delivery_by_window.items():
            if window_index in self.bank and delivery_ids:
                self.bank[window_index].handed_delivery_id = delivery_ids[-1]
        return sorted(old_indices | new_indices)

    # ------------------------------------------------------------------
    # Readiness / views
    # ------------------------------------------------------------------
    def _sensor_name_for_id(self, sensor_id: int, ids: dict[str, int]) -> str | None:
        for name, sid in ids.items():
            if sid == sensor_id:
                return name
        return None

    def horizons(self) -> dict[str, float]:
        _, owner_map, ids = self._coverage_for_active()
        present_ids = set(int(v) for v in np.unique(owner_map) if int(v) >= 0)
        result: dict[str, float] = {}
        for sid in present_ids:
            sensor = self._sensor_name_for_id(sid, ids)
            if sensor is None:
                continue
            p = self.progress.get(sensor)
            if p is None or self.book.active_version is None:
                result[sensor] = float("-inf")
                continue
            try:
                cal = self.book.get(sensor)
                result[sensor] = cal.completeness_horizon(p, self.config.snap_tol)
            except Exception:
                result[sensor] = float("-inf")
        return result

    def global_progress(self, sensor: str | None = None) -> float:
        """Safe exclusive global horizon for one sensor or all owned coverage."""

        if sensor is not None:
            p = self.progress.get(sensor)
            if p is None:
                return float("-inf")
            cal = self.book.get(sensor)
            return cal.completeness_horizon(p, self.config.snap_tol)
        horizons = self.horizons()
        if not horizons:
            return float("inf")
        return float(min(horizons.values()))

    watermark = global_progress

    def _readiness(
        self, window_index: int, end: float | None = None
    ) -> tuple[bool, dict[str, float], list[str]]:
        if end is None:
            end = self.config.window_end_time(window_index)
        horizons = self.horizons()
        reasons: list[str] = []
        tol = 1e-8
        for sensor, horizon in horizons.items():
            if not np.isfinite(horizon):
                reasons.append(f"{sensor}: no progress/calibration")
            elif horizon + tol < end:
                reasons.append(
                    f"{sensor}: global horizon {horizon:g} < window end {end:g}"
                )
        # If there are no owners at all, spatial coverage is empty.  The window
        # can be numerically sealed but has no effective zero coverage.
        return not reasons, horizons, reasons

    def window(self, index: int, *, create: bool = True) -> WindowView | None:
        index = int(index)
        complete, horizons, reasons = self._readiness(index)
        marker = self._latest_marker(index)
        conflicts = [] if marker is None else list(marker.conflicts)

        if index not in self.bank:
            delivery = self.handed_window(index)
            if delivery is not None:
                # The active state was transferred/dropped at handoff.  Keep
                # the original delivery queryable rather than allocating a
                # blank replacement tensor.
                state = WindowState(
                    index=index,
                    config=self.config,
                    voxels=delivery.voxels,
                    boundary_loss_time=delivery.boundary_loss_by_time,
                    coverage=delivery.coverage,
                    accepted_events=0,
                    handed_delivery_id=delivery.delivery_id,
                    revision=delivery.revision,
                )
                return WindowView(
                    index=index,
                    revision=delivery.revision,
                    calibration_version=delivery.calibration_version,
                    state=state,
                    complete=delivery.ready_at_handoff,
                    usable=delivery.usable,
                    horizons=horizons,
                    window_end=self.config.window_end_time(index),
                    reasons=reasons,
                    delivery=delivery,
                    conflict=delivery.conflict,
                    conflicts=conflicts,
                )
            if not create:
                return None
            state = self.bank.get(index, create=True)
        else:
            state = self.bank[index]

        self._apply_coverage(index)
        delivery_id = state.handed_delivery_id
        delivery = None if delivery_id is None else self.deliveries.get(delivery_id)
        return WindowView(
            index=index,
            revision=self.book.revision,
            calibration_version=self.book.active_version,
            state=state,
            complete=complete,
            # The current rebuilt distribution is usable whenever progress is
            # complete.  ``conflict`` separately reports that an already-handed
            # revision is stale; the original delivery itself becomes unusable.
            usable=complete,
            horizons=horizons,
            window_end=self.config.window_end_time(index),
            reasons=reasons,
            delivery=delivery,
            conflict=bool(conflicts),
            conflicts=conflicts,
        )

    get_window = window
    peek_window = window
    view_window = window

    def ready(self, index: int) -> bool:
        complete, _, _ = self._readiness(index)
        return complete

    def ready_indices(self, max_scan: int = 64) -> list[int]:
        out = []
        start = self.next_window
        for idx in range(start, start + max_scan):
            if self.ready(idx):
                out.append(idx)
            else:
                break
        return out

    # ------------------------------------------------------------------
    # Handoff
    # ------------------------------------------------------------------
    def _new_delivery_id(self) -> str:
        return f"w{uuid.uuid4().hex[:12]}"

    def _latest_marker(self, window_index: int) -> ConflictMarker | None:
        return self.markers_by_window.get(int(window_index))

    def _marker_for_delivery(self, delivery_id: str) -> ConflictMarker | None:
        for marker in self.markers_by_window.values():
            if marker.delivery_id == delivery_id:
                return marker
        return None

    def claim_window(
        self,
        index: int | None = None,
        *,
        require_ready: bool = True,
        copy: bool = False,
        archive: bool | None = None,
    ) -> WindowDelivery:
        if index is None:
            return self.claim_next(require_ready=require_ready, copy=copy, archive=archive)  # type: ignore[misc]
        index = int(index)
        view = self.window(index, create=True)
        assert view is not None
        if require_ready and not view.complete:
            raise WindowNotReadyError(
                f"window {index} is not ready for inference: {'; '.join(view.reasons)}"
            )

        state = self.bank[index]
        delivery_id = self._new_delivery_id()
        marker = ConflictMarker(index, delivery_id, ready_at_handoff=view.complete)
        self.markers_by_window[index] = marker

        if copy:
            voxels = state.voxels.copy()
            coverage = state.coverage.copy()
            loss_t = state.boundary_loss_time.copy()
        else:
            # True handoff/transfer: the dense active array ownership moves to
            # the immutable delivery and this operator drops the window state.
            voxels = state.voxels
            coverage = state.coverage
            loss_t = state.boundary_loss_time

        delivery = WindowDelivery(
            window_index=index,
            delivery_id=delivery_id,
            revision=self.book.revision,
            calibration_version=self.book.active_version,
            voxels=voxels,
            coverage=coverage,
            boundary_loss=state.boundary_loss,
            boundary_loss_by_time=loss_t,
            progress=dict(self.progress),
            horizons=dict(view.horizons),
            window_end=view.window_end,
            ready_at_handoff=view.complete,
            marker=marker,
            dtype=self.config.dtype,
        )

        state.handed_delivery_id = delivery_id
        self.delivery_by_window.setdefault(index, []).append(delivery_id)
        if not copy:
            self.bank.drop(index)

        should_archive = self.config.retain_deliveries if archive is None else archive
        if should_archive:
            self.deliveries[delivery_id] = delivery
        # If not archived, the caller's returned object and shared marker are
        # enough.  Marker remains small and lets future corrections be tracked.
        if index == self.next_window:
            self.next_window = index + 1
        elif index > self.next_window:
            # Out-of-order claim: do not skip unclaimed lower windows.
            pass
        return delivery

    handoff_window = claim_window
    take_window = claim_window
    handoff = claim_window
    take = claim_window

    def claim_next(
        self,
        *,
        require_ready: bool = True,
        copy: bool = False,
        archive: bool | None = None,
    ) -> WindowDelivery | None:
        index = self.next_window
        # Ensure an empty but covered window can still be handed off.
        view = self.window(index, create=True)
        if require_ready and not view.complete:
            return None
        return self.claim_window(
            index, require_ready=require_ready, copy=copy, archive=archive
        )

    take_next = claim_next
    handoff_next = claim_next

    def claim_ready(self, *, copy: bool = False) -> list[WindowDelivery]:
        deliveries = []
        while True:
            d = self.claim_next(copy=copy)
            if d is None:
                break
            deliveries.append(d)
        return deliveries

    def get_delivery(self, delivery_id: str) -> WindowDelivery | None:
        return self.deliveries.get(delivery_id)

    def handed_window(self, index: int, revision: int | None = None) -> WindowDelivery | None:
        ids = self.delivery_by_window.get(int(index), [])
        if not ids:
            return None
        if revision is None:
            return self.deliveries.get(ids[-1])
        for did in ids:
            d = self.deliveries.get(did)
            if d is not None and d.revision == revision:
                return d
        return None

    get_handed_window = handed_window
    get_handoff = handed_window

    def ack(self, delivery_or_id: str | WindowDelivery, *, drop_window: bool = False) -> None:
        did = delivery_or_id if isinstance(delivery_or_id, str) else delivery_or_id.delivery_id
        self.deliveries.pop(did, None)
        if drop_window:
            for win, ids in list(self.delivery_by_window.items()):
                if did in ids:
                    self.bank.drop(win)

    # ------------------------------------------------------------------
    # Exchange replay
    # ------------------------------------------------------------------
    def submit(self, submission: Any, tensors: Mapping[str, np.ndarray] | None = None) -> Any:
        """Process one exchange submission record."""

        if isinstance(submission, Mapping):
            kind = submission.get("kind") or submission.get("type")
            if kind == "events":
                return self.ingest_events(
                    str(submission["sensor"]),
                    submission["events"],
                    submission.get("batch_id"),
                )
            if kind == "progress":
                return self.update_progress(submission["sensor"], submission["local_timestamp"])
            if kind == "calibration":
                return self.activate_calibration(str(submission["version"]))
            raise ValueError(f"unknown submission kind {kind!r}")

        kind = submission[0]
        if kind == "events":
            _, sensor, batch_id, tensor_name = submission
            if tensors is None or tensor_name not in tensors:
                raise KeyError(f"tensor {tensor_name!r} was not provided")
            return self.ingest_events(sensor, tensors[tensor_name], batch_id)
        if kind == "progress":
            _, sensor, value = submission
            return self.update_progress(sensor, value)
        if kind == "calibration":
            return self.activate_calibration(str(submission[1]))
        raise ValueError(f"unknown submission {submission!r}")

    process_submission = submit
    process_record = submit
    handle_record = submit

    def replay(
        self,
        submissions: Iterable[Any],
        tensors: Mapping[str, np.ndarray] | None = None,
        *,
        auto_claim: bool = False,
    ) -> list[WindowDelivery]:
        claimed: list[WindowDelivery] = []
        for submission in submissions:
            self.submit(submission, tensors)
            if auto_claim:
                claimed.extend(self.claim_ready())
        return claimed


# Avoid circular typing import at runtime.
CalibrationLike = Any
