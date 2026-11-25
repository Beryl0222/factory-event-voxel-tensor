"""Configuration objects for the event voxel operator."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import json

import numpy as np


def _as_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if np.isscalar(value):
        v = float(value)
        return v, v
    if len(value) != 2:
        raise ValueError(f"expected a pair, got {value!r}")
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class VoxelConfig:
    """Global voxel grid and active-window policy.

    Axis order for :attr:`origin` and spatial steps is ``(y, x)``; time is
    configured separately.  A window contains ``window_bins`` temporal grid
    points.  Grid point ``k`` is at global time
    ``origin_t + k * time_step``; window ``w`` owns indices
    ``[w*window_bins, (w+1)*window_bins)`` and has half-open physical end
    ``origin_t + (w+1)*window_bins*time_step``.
    """

    shape_yx: tuple[int, int]
    window_bins: int
    time_step: float = 1.0
    origin_y: float = 0.0
    origin_x: float = 0.0
    origin_t: float = 0.0
    step_y: float = 1.0
    step_x: float = 1.0
    dtype: Any = np.float64
    max_active_windows: int | None = 4
    chunk_size: int = 4096
    scratch_budget_bytes: int | None = None
    snap_tol: float = 1e-9
    retain_deliveries: bool = True

    def __post_init__(self) -> None:
        h, w = int(self.shape_yx[0]), int(self.shape_yx[1])
        if h <= 0 or w <= 0:
            raise ValueError("shape_yx must be positive")
        if int(self.window_bins) <= 0:
            raise ValueError("window_bins must be positive")
        if float(self.time_step) <= 0:
            raise ValueError("time_step must be positive")
        if float(self.step_y) <= 0 or float(self.step_x) <= 0:
            raise ValueError("spatial steps must be positive")
        if self.max_active_windows is not None and int(self.max_active_windows) <= 0:
            raise ValueError("max_active_windows must be positive or None")
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive")
        if self.scratch_budget_bytes is not None and int(self.scratch_budget_bytes) <= 0:
            raise ValueError("scratch_budget_bytes must be positive or None")

    @property
    def H(self) -> int:
        return int(self.shape_yx[0])

    @property
    def W(self) -> int:
        return int(self.shape_yx[1])

    @property
    def B(self) -> int:
        return int(self.window_bins)

    @property
    def temporal_window_span(self) -> float:
        return self.B * float(self.time_step)

    def window_start_time(self, window_index: int) -> float:
        return self.origin_t + int(window_index) * self.temporal_window_span

    def window_end_time(self, window_index: int) -> float:
        return self.origin_t + (int(window_index) + 1) * self.temporal_window_span

    def with_overrides(self, **kwargs: Any) -> "VoxelConfig":
        return replace(self, **kwargs)

    @classmethod
    def from_mapping(
        cls,
        records: Mapping[str, Any],
        capacity: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> "VoxelConfig":
        """Build config from the exchange ``records`` mapping.

        Recognized aliases include ``window_size``/``window_bins``,
        ``time_step``/``temporal_step`` and ``max_active_windows``.
        A three-value ``origin`` sequence is interpreted as ``(y, x, t)``;
        a mapping may use explicit ``y``/``x``/``t`` keys.
        """

        data = dict(records)
        if "shape_y_x" not in data and isinstance(data.get("records"), Mapping):
            data = dict(data["records"])
        cap = dict(capacity or {})

        shape = data.get("shape_y_x", data.get("shape_yx", data.get("shape")))
        if shape is None:
            raise ValueError("records is missing shape_y_x")

        bins = data.get(
            "window_bins",
            data.get("window_size", cap.get("window_size")),
        )
        if bins is None:
            raise ValueError("window_size/window_bins is required")

        time_step = data.get(
            "time_step",
            data.get("temporal_step", data.get("time_voxel_step", 1.0)),
        )

        origin_y = origin_x = origin_t = 0.0
        origin = data.get("origin", [0.0, 0.0, 0.0])
        if isinstance(origin, Mapping):
            origin_y = float(origin.get("y", origin.get("origin_y", 0.0)))
            origin_x = float(origin.get("x", origin.get("origin_x", 0.0)))
            origin_t = float(origin.get("t", origin.get("time", origin.get("origin_t", 0.0))))
        elif origin is not None:
            origin = list(origin)
            axis_order = data.get("origin_axis_order", ("y", "x", "t"))
            if len(origin) == 3 and tuple(axis_order) == ("y", "x", "t"):
                origin_y, origin_x, origin_t = map(float, origin)
            elif len(origin) == 3 and tuple(axis_order) in (
                ("t", "y", "x"),
                ("time", "y", "x"),
            ):
                origin_t, origin_y, origin_x = map(float, origin)
            elif len(origin) >= 2:
                # The exchange format used by the checked-in case follows
                # event-column order after event id: y, x, t.
                origin_y, origin_x = float(origin[0]), float(origin[1])
                if len(origin) >= 3:
                    origin_t = float(origin[2])

        step_y = data.get("step_y", data.get("voxel_step_y", None))
        step_x = data.get("step_x", data.get("voxel_step_x", None))
        spatial_step = data.get(
            "spatial_step_yx",
            data.get("step_yx", data.get("voxel_size_yx", None)),
        )
        if spatial_step is not None:
            sy, sx = _as_pair(spatial_step, (1.0, 1.0))
            step_y = sy if step_y is None else step_y
            step_x = sx if step_x is None else step_x
        step_y = 1.0 if step_y is None else float(step_y)
        step_x = 1.0 if step_x is None else float(step_x)

        max_active = overrides.pop(
            "max_active_windows",
            data.get("max_active_windows", cap.get("max_active_windows", 4)),
        )
        scratch_budget = overrides.pop(
            "scratch_budget_bytes",
            data.get("peak_scratch_bytes", cap.get("peak_scratch_bytes")),
        )
        chunk_size = overrides.pop("chunk_size", None)
        if chunk_size is None:
            chunk_size = data.get("chunk_size")
        if chunk_size is None:
            if scratch_budget:
                # Conservative upper estimate for projection/floor/weight
                # temporaries: roughly 512 bytes per event in a chunk.
                chunk_size = max(256, int(scratch_budget) // 512)
            else:
                chunk_size = 4096
        retain = overrides.pop("retain_deliveries", True)

        dtype = overrides.pop("dtype", data.get("dtype", np.float64))
        if isinstance(dtype, str):
            dtype = np.dtype(dtype).type

        cfg = cls(
            shape_yx=(int(shape[0]), int(shape[1])),
            window_bins=int(bins),
            time_step=float(time_step),
            origin_y=origin_y,
            origin_x=origin_x,
            origin_t=origin_t,
            step_y=step_y,
            step_x=step_x,
            dtype=dtype,
            max_active_windows=None if max_active is None else int(max_active),
            chunk_size=int(chunk_size),
            scratch_budget_bytes=None if scratch_budget is None else int(scratch_budget),
            retain_deliveries=bool(retain),
        )
        if overrides:
            cfg = replace(cfg, **overrides)
        return cfg


VoxelConfig.from_dict = VoxelConfig.from_mapping  # type: ignore[attr-defined]


def load_case_config(
    capture_path: str | Path,
    capacity_path: str | Path | None = None,
    **overrides: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None, VoxelConfig]:
    """Load records and build a config from capture/capacity JSON files."""

    capture_path = Path(capture_path)
    envelope = json.loads(capture_path.read_text(encoding="utf-8"))
    records = envelope.get("records", envelope)
    capacity = None
    if capacity_path is not None:
        capacity = json.loads(Path(capacity_path).read_text(encoding="utf-8"))
    else:
        candidate = capture_path.parent / "capacity.json"
        if candidate.exists():
            capacity = json.loads(candidate.read_text(encoding="utf-8"))
    return records, capacity, VoxelConfig.from_mapping(records, capacity, **overrides)
