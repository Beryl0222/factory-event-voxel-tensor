"""Conventional solution entry point re-exporting the public operator API."""

from event_voxel import *  # noqa: F401,F403
from event_voxel import (  # noqa: F401
    Calibration,
    CalibrationBook,
    ConflictMarker,
    StreamingVoxelizer,
    VoxelConfig,
    VoxelizationResult,
    WindowDelivery,
    WindowView,
    voxelize_arrays,
)
from event_voxel.replay import replay_records

EventVoxelizer = StreamingVoxelizer
WindowHandoff = StreamingVoxelizer

__all__ = [
    "Calibration",
    "CalibrationBook",
    "ConflictMarker",
    "EventVoxelizer",
    "StreamingVoxelizer",
    "VoxelConfig",
    "VoxelizationResult",
    "WindowDelivery",
    "WindowHandoff",
    "WindowView",
    "replay_records",
    "voxelize_arrays",
]
