"""Compatibility spelling for the event voxelization operator."""

from event_voxel import (
    Calibration,
    CalibrationBook,
    StreamingVoxelizer,
    VoxelConfig,
    VoxelizationResult,
    voxelize_arrays,
    voxelize_events,
)

EventVoxelizationOperator = StreamingVoxelizer

__all__ = [
    "Calibration",
    "CalibrationBook",
    "EventVoxelizationOperator",
    "StreamingVoxelizer",
    "VoxelConfig",
    "VoxelizationResult",
    "voxelize_arrays",
    "voxelize_events",
]
