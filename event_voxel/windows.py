"""Active dense window storage.

The bank deliberately knows nothing about the full time axis.  A dense tensor
is allocated only when a window receives data or is explicitly inspected.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .config import VoxelConfig


class ActiveWindowLimitError(RuntimeError):
    """Raised when an additional window would exceed the active-data budget."""


class CallableArray(np.ndarray):
    """Array that is also callable.

    Some integrations expose coverage as a property; older call sites may use
    an ``effective_coverage()`` accessor.  This small adapter supports both.
    """

    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def __call__(self) -> "CallableArray":
        return self


@dataclass
class WindowState:
    index: int
    config: VoxelConfig
    voxels: np.ndarray | None = None
    boundary_loss_time: np.ndarray | None = None
    coverage: np.ndarray | None = None
    owner_map: np.ndarray | None = None
    sensor_ids: dict[str, int] = field(default_factory=dict)
    accepted_events: int = 0
    clipped_event_components: int = 0
    calibration_versions: set[str] = field(default_factory=set)
    revision: int = 0
    handed_delivery_id: str | None = None
    dirty: bool = False

    def __post_init__(self) -> None:
        if self.voxels is None:
            self.voxels = np.zeros(
                (2, self.config.B, self.config.H, self.config.W),
                dtype=self.config.dtype,
            )
        if self.boundary_loss_time is None:
            self.boundary_loss_time = np.zeros(self.config.B, dtype=np.float64)
        if self.coverage is None:
            self.coverage = np.zeros((self.config.H, self.config.W), dtype=bool)
        if self.owner_map is None:
            self.owner_map = np.full((self.config.H, self.config.W), -1, dtype=np.int16)

    @property
    def boundary_loss(self) -> float:
        return float(np.sum(self.boundary_loss_time, dtype=np.float64))

    @property
    def allocated_mass(self) -> float:
        return float(np.sum(self.voxels, dtype=np.float64))

    @property
    def spatial_coverage_fraction(self) -> float:
        return float(np.count_nonzero(self.coverage)) / float(self.coverage.size)

    def reset_values(self) -> None:
        """Zero voxel/loss accumulators while retaining the window object."""

        self.voxels.fill(0)
        self.boundary_loss_time.fill(0)
        self.accepted_events = 0
        self.clipped_event_components = 0
        self.calibration_versions.clear()

    def effective_coverage(self, complete: bool) -> np.ndarray:
        """Coverage where a zero is a valid no-transition observation.

        Spatial ownership is represented by :attr:`coverage`.  A window that
        is not progress-complete cannot support a no-event conclusion anywhere,
        so the effective zero mask is empty in that case while the configured
        spatial mask remains available separately.
        """

        if complete:
            return self.coverage.copy()
        return np.zeros_like(self.coverage)

    def voxel_coverage(self, complete: bool, *, channels: bool = False) -> np.ndarray:
        """Broadcast effective coverage to temporal (and optionally polar) axes."""

        mask = self.effective_coverage(complete)
        temporal = np.broadcast_to(mask[None, :, :], (self.config.B, *mask.shape)).copy()
        if channels:
            return np.broadcast_to(
                temporal[None, :, :, :], (2, self.config.B, *mask.shape)
            ).copy()
        return temporal


class WindowBank:
    """Bounded ordered mapping of active window indices to dense states."""

    def __init__(self, config: VoxelConfig) -> None:
        self.config = config
        self._windows: OrderedDict[int, WindowState] = OrderedDict()

    def __contains__(self, index: object) -> bool:
        return int(index) in self._windows

    def __getitem__(self, index: int) -> WindowState:
        return self._windows[int(index)]

    def __iter__(self) -> Iterator[int]:
        return iter(self._windows)

    def __len__(self) -> int:
        return len(self._windows)

    def indices(self) -> list[int]:
        return list(self._windows.keys())

    def states(self) -> list[WindowState]:
        return list(self._windows.values())

    def get(self, index: int, *, create: bool = True) -> WindowState | None:
        index = int(index)
        if index in self._windows:
            self._windows.move_to_end(index)
            return self._windows[index]
        if not create:
            return None
        limit = self.config.max_active_windows
        if limit is not None and len(self._windows) >= limit:
            raise ActiveWindowLimitError(
                f"active window limit {limit} reached; claim/ack or drop a window first"
            )
        state = WindowState(index=index, config=self.config)
        self._windows[index] = state
        return state

    def drop(self, index: int) -> bool:
        return self._windows.pop(int(index), None) is not None

    def clear(self) -> None:
        self._windows.clear()

    def dense_bytes(self) -> int:
        total = 0
        for state in self._windows.values():
            total += int(state.voxels.nbytes)
            total += int(state.boundary_loss_time.nbytes)
            total += int(state.coverage.nbytes)
            total += int(state.owner_map.nbytes)
        return total
