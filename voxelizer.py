"""Compatibility import path for the event voxel operator."""

from event_voxel import *  # noqa: F401,F403
from event_voxel import (  # noqa: F401
    Calibration,
    CalibrationBook,
    ConflictMarker,
    EventToTensorVoxelizer,
    EventVoxelOperator,
    StreamingVoxelizer,
    VoxelConfig,
    VoxelizationResult,
    WindowDelivery,
    WindowHandoff,
    WindowView,
    calibrate_array,
    voxelize_arrays,
)

EventVoxelizer = StreamingVoxelizer
EventStreamVoxelizer = StreamingVoxelizer
VoxelStream = StreamingVoxelizer

__all__ = [
    "Calibration",
    "CalibrationBook",
    "ConflictMarker",
    "EventToTensorVoxelizer",
    "EventVoxelOperator",
    "EventVoxelizer",
    "EventStreamVoxelizer",
    "StreamingVoxelizer",
    "VoxelConfig",
    "VoxelizationResult",
    "VoxelStream",
    "WindowDelivery",
    "WindowHandoff",
    "WindowView",
    "calibrate_array",
    "voxelize_arrays",
]
