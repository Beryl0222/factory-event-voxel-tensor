"""Pure NumPy event-to-voxel splatting operator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .calibration import Calibration, CalibrationBook
from .config import VoxelConfig
from .windows import WindowBank, WindowState


EVENT_COLUMNS = ("event_id", "y", "x", "t_local", "polarity")


def floor_and_fraction(values: np.ndarray, snap_tol: float = 1e-9):
    """Floor and fractional part with small floating-point snapping."""

    values = np.asarray(values, dtype=np.float64)
    nearest = np.round(values)
    near_integer = np.abs(values - nearest) <= snap_tol
    snapped = np.where(near_integer, nearest, values)
    index = np.floor(snapped).astype(np.int64)
    frac = snapped - index
    frac = np.where(np.abs(frac) <= snap_tol, 0.0, frac)
    frac = np.where(np.abs(frac - 1.0) <= snap_tol, 1.0, frac)
    return index, frac


def polarity_channel(polarity: np.ndarray) -> np.ndarray:
    """Map -1 to channel 0 and +1 to channel 1."""

    polarity = np.asarray(polarity)
    return (polarity > 0).astype(np.intp)


def project_events(
    events: np.ndarray,
    calibration: Calibration,
    *,
    snap_tol: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Project raw event rows using one calibration."""

    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] < 5:
        raise ValueError("events must have shape (N, 5+) with id,y,x,t,polarity")
    raw_y = events[:, 1].astype(np.float64, copy=False)
    raw_x = events[:, 2].astype(np.float64, copy=False)
    raw_t = events[:, 3].astype(np.float64, copy=False)
    polarity = events[:, 4]

    projected_y = calibration.project_y(raw_y)
    projected_x = calibration.project_x(raw_x)
    global_t = calibration.corrected_time(raw_t, raw_y)
    owner_valid = (
        (projected_y >= calibration.owner.y0)
        & (projected_y < calibration.owner.y1)
        & (projected_x >= calibration.owner.x0)
        & (projected_x < calibration.owner.x1)
    )
    effective_valid = (raw_t >= calibration.effective_lo) & (
        raw_t < calibration.effective_hi
    )
    return {
        "event_id": events[:, 0],
        "raw_y": raw_y,
        "raw_x": raw_x,
        "raw_t": raw_t,
        "y": np.asarray(projected_y, dtype=np.float64),
        "x": np.asarray(projected_x, dtype=np.float64),
        "t": np.asarray(global_t, dtype=np.float64),
        "channel": polarity_channel(polarity),
        "owner_valid": owner_valid,
        "effective_valid": effective_valid,
        "valid": owner_valid & effective_valid,
    }


# Descriptive alias used by the package root and integrations.
calibrate_array = project_events


def _add_spatial_corners_for_time(
    state: WindowState,
    config: VoxelConfig,
    *,
    channels: np.ndarray,
    y_index: np.ndarray,
    x_index: np.ndarray,
    wy0: np.ndarray,
    wx0: np.ndarray,
    time_grid_index: np.ndarray,
    time_weight: np.ndarray,
    snap_tol: float,
) -> tuple[float, int]:
    """Splat one temporal endpoint with four bilinear spatial corners."""

    B = config.B
    local_t = np.mod(time_grid_index, B).astype(np.intp)
    total_loss = 0.0
    clipped_components = 0

    for iy, wy_base in ((0, wy0), (1, 1.0 - wy0)):
        yy = y_index + iy
        for ix, wx_base in ((0, wx0), (1, 1.0 - wx0)):
            xx = x_index + ix
            weight = wy_base * wx_base * time_weight
            inside = (yy >= 0) & (yy < config.H) & (xx >= 0) & (xx < config.W)
            outside = ~inside

            if np.any(outside):
                loss_weights = weight[outside]
                total_loss += float(np.sum(loss_weights, dtype=np.float64))
                clipped_components += int(
                    np.count_nonzero(np.abs(loss_weights) > snap_tol)
                )
                # Loss is attributable to this window's temporal grid point.
                np.add.at(
                    state.boundary_loss_time,
                    local_t[outside],
                    loss_weights.astype(np.float64, copy=False),
                )

            if np.any(inside):
                sel = inside
                np.add.at(
                    state.voxels,
                    (
                        channels[sel],
                        local_t[sel],
                        yy[sel].astype(np.intp),
                        xx[sel].astype(np.intp),
                    ),
                    weight[sel],
                )

    return total_loss, clipped_components


def splat_prepared(
    prepared: Mapping[str, np.ndarray],
    config: VoxelConfig,
    bank: WindowBank | None = None,
    *,
    valid: np.ndarray | None = None,
    version: str | None = None,
) -> tuple[WindowBank, dict[str, Any]]:
    """Splat prepared/projected events into a bounded window bank."""

    if bank is None:
        bank = WindowBank(config)

    if valid is None:
        valid = np.asarray(prepared.get("valid", np.ones_like(prepared["y"], dtype=bool)))
    valid = np.asarray(valid, dtype=bool)

    stats = {
        "input_events": int(valid.shape[0]),
        "accepted_events": int(np.count_nonzero(valid)),
        "boundary_loss": 0.0,
        "clipped_components": 0,
        "windows_touched": set(),
    }
    if not np.any(valid):
        return bank, stats

    y = prepared["y"][valid]
    x = prepared["x"][valid]
    t = prepared["t"][valid]
    channels = prepared["channel"][valid]

    gy = (y - config.origin_y) / config.step_y
    gx = (x - config.origin_x) / config.step_x
    gt = (t - config.origin_t) / config.time_step

    ky, fy = floor_and_fraction(gy, config.snap_tol)
    kx, fx = floor_and_fraction(gx, config.snap_tol)
    kt, ft = floor_and_fraction(gt, config.snap_tol)
    wy0 = 1.0 - fy
    wx0 = 1.0 - fx

    # Process only non-zero temporal endpoints.  An exact grid time has no
    # meaningful zero-weight neighbor and must not allocate an adjacent window.
    for time_side, raw_weight in ((0, 1.0 - ft), (1, ft)):
        active = np.abs(raw_weight) > config.snap_tol
        if not np.any(active):
            continue

        grid_t = (kt + time_side)[active]
        time_weight = raw_weight[active]
        window_index = np.floor_divide(grid_t, config.B).astype(np.int64)
        active_channels = channels[active]
        active_ky = ky[active]
        active_kx = kx[active]
        active_wy0 = wy0[active]
        active_wx0 = wx0[active]

        for win in np.unique(window_index):
            win = int(win)
            sel = window_index == win
            state = bank.get(win)
            if version is not None:
                state.calibration_versions.add(str(version))
            state.accepted_events += int(np.count_nonzero(sel))
            stats["windows_touched"].add(win)

            loss, clipped = _add_spatial_corners_for_time(
                state,
                config,
                channels=active_channels[sel],
                y_index=active_ky[sel],
                x_index=active_kx[sel],
                wy0=active_wy0[sel],
                wx0=active_wx0[sel],
                time_grid_index=grid_t[sel],
                time_weight=time_weight[sel],
                snap_tol=config.snap_tol,
            )
            stats["boundary_loss"] += loss
            stats["clipped_components"] += clipped
            state.clipped_event_components += clipped

    return bank, stats


def target_window_indices(
    prepared: Mapping[str, np.ndarray],
    config: VoxelConfig,
    *,
    valid: np.ndarray | None = None,
) -> set[int]:
    """Return dense window blocks that would receive non-zero weights."""

    if valid is None:
        valid = np.asarray(prepared.get("valid", np.ones_like(prepared["y"], dtype=bool)))
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        return set()
    t = prepared["t"][valid]
    gt = (t - config.origin_t) / config.time_step
    kt, ft = floor_and_fraction(gt, config.snap_tol)
    windows = {int(w) for w in np.floor_divide(kt, config.B)}
    split = np.abs(ft) > config.snap_tol
    if np.any(split):
        next_grid = kt[split] + 1
        windows.update(int(w) for w in np.floor_divide(next_grid, config.B))
    return windows


@dataclass
class VoxelizationResult:
    """Return value of the one-shot functional operator."""

    config: VoxelConfig
    bank: WindowBank
    calibration: Calibration | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def windows(self) -> dict[int, np.ndarray]:
        return {i: s.voxels for i, s in self.bank._windows.items()}

    @property
    def voxels_by_window(self) -> dict[int, np.ndarray]:
        return self.windows

    @property
    def coverage_by_window(self) -> dict[int, np.ndarray]:
        return {i: s.coverage for i, s in self.bank._windows.items()}

    @property
    def boundary_loss_by_window(self) -> dict[int, float]:
        return {i: s.boundary_loss for i, s in self.bank._windows.items()}

    def mass_balance_error(self) -> float:
        accepted = float(self.stats.get("accepted_events", 0))
        mass = sum(s.allocated_mass + s.boundary_loss for s in self.bank.states())
        return abs(mass - accepted)


def voxelize_arrays(
    events: np.ndarray | list[list[Any]],
    config: VoxelConfig | None = None,
    calibration: Calibration | CalibrationBook | Mapping[str, Any] | None = None,
    *,
    book: CalibrationBook | None = None,
    sensor: str | None = None,
    version: str | None = None,
    shape_yx: tuple[int, int] | None = None,
    shape: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    H: int | None = None,
    W: int | None = None,
    window_size: int | None = None,
    window_bins: int | None = None,
    bins: int | None = None,
    time_step: float | None = None,
    temporal_step: float | None = None,
    origin: tuple[float, float, float] | Mapping[str, float] | None = None,
    voxel_size: Any = None,
    step: Any = None,
    spatial_step_yx: tuple[float, float] | None = None,
    dtype: Any = np.float64,
    **kwargs: Any,
) -> VoxelizationResult:
    """One-shot event tensor voxelization for tests and offline use.

    Provide either a :class:`VoxelConfig`, or ``shape_yx`` and
    ``window_size``/``window_bins``/``bins``.  Provide either a concrete
    :class:`Calibration`, or a ``book`` plus ``sensor``/``version``.
    Extra keyword aliases are accepted for exploratory callers.
    """

    array = np.asarray(events)
    if config is None:
        if shape_yx is None:
            shape_yx = shape
        if shape_yx is None and (height or H) and (width or W):
            shape_yx = (int(height or H), int(width or W))
        if shape_yx is None:
            raise ValueError("either config or shape_yx/(height,width) is required")
        bin_count = window_bins or window_size or bins or kwargs.get("num_bins")
        if bin_count is None:
            raise ValueError("window_size/window_bins/bins is required")
        time_step = time_step if time_step is not None else temporal_step
        sy = sx = 1.0
        if spatial_step_yx is not None:
            sy, sx = map(float, spatial_step_yx)
        step_like = voxel_size if voxel_size is not None else step
        if step_like is not None:
            step_values = list(step_like)
            if len(step_values) == 3:
                # Same axis convention as origin: (y, x, t), unless a caller
                # uses explicit spatial_step_yx/time_step.
                sy, sx, maybe_t = map(float, step_values)
                time_step = maybe_t if time_step is None else time_step
            elif len(step_values) == 2:
                sy, sx = map(float, step_values)

        oy = ox = ot = 0.0
        if isinstance(origin, Mapping):
            oy = float(origin.get("y", 0.0))
            ox = float(origin.get("x", 0.0))
            ot = float(origin.get("t", origin.get("time", 0.0)))
        elif origin is not None:
            oy, ox, ot = map(float, origin)
        config = VoxelConfig(
            shape_yx=tuple(shape_yx),  # type: ignore[arg-type]
            window_bins=int(bin_count),
            time_step=1.0 if time_step is None else float(time_step),
            origin_y=float(oy),
            origin_x=float(ox),
            origin_t=float(ot),
            step_y=sy,
            step_x=sx,
            dtype=dtype,
            max_active_windows=None,
        )

    if isinstance(calibration, CalibrationBook):
        book = calibration
        calibration = None
    if calibration is None:
        if book is None or sensor is None:
            raise ValueError("calibration or book+sensor is required")
        calibration = book.get(sensor, version=version)
    elif isinstance(calibration, Mapping):
        calibration = CalibrationBook({"_": {sensor or "sensor": calibration}}).get(
            sensor or "sensor", version="_"
        )

    assert isinstance(calibration, Calibration)
    prepared = project_events(array, calibration, snap_tol=config.snap_tol)
    bank = WindowBank(config)
    _, stats = splat_prepared(
        prepared, config, bank, valid=prepared["valid"], version=calibration.version
    )

    coverage, owner_map, ids = CalibrationBook(
        {calibration.version: {calibration.sensor: _calibration_to_mapping(calibration)}}
    ).coverage_map(
        config.H,
        config.W,
        origin_y=config.origin_y,
        origin_x=config.origin_x,
        step_y=config.step_y,
        step_x=config.step_x,
    )
    for state in bank.states():
        state.coverage = coverage.copy()
        state.owner_map = owner_map.copy()
        state.sensor_ids = dict(ids)

    return VoxelizationResult(
        config=config, bank=bank, calibration=calibration, stats=stats
    )


def _calibration_to_mapping(cal: Calibration) -> dict[str, Any]:
    return {
        "effective": [cal.effective_lo, cal.effective_hi],
        "scale": [cal.scale_y, cal.scale_x],
        "shift": [cal.shift_y, cal.shift_x],
        "owner": [cal.owner.y0, cal.owner.y1, cal.owner.x0, cal.owner.x1],
        "offset": cal.offset,
        "line_delay": cal.line_delay,
    }
