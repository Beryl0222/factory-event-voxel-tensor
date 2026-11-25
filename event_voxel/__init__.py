"""Streaming event-to-voxel operator and window handoff.

The public objects are intentionally re-exported at package root.  The
implementation never allocates a time axis: only explicitly active window
blocks own dense tensors.
"""

from .calibration import Calibration, CalibrationBook, OwnerRegion
from .config import VoxelConfig, load_case_config
from .stream import (
    ConflictMarker,
    StreamingVoxelizer,
    WindowDelivery,
    WindowView,
)
from .voxelizer import VoxelizationResult, calibrate_array, voxelize_arrays

__all__ = [
    "Calibration",
    "CalibrationBook",
    "Config",
    "ConflictMarker",
    "ContinuousInputHandoff",
    "EventToTensorVoxelizationOperator",
    "EventToTensorVoxelizer",
    "EventVoxelOperator",
    "EventVoxelizer",
    "OwnerRegion",
    "StreamingVoxelizer",
    "VoxelConfig",
    "VoxelizationResult",
    "WindowDelivery",
    "WindowHandoff",
    "WindowHandoffManager",
    "WindowView",
    "calibrate_array",
    "events_to_voxels",
    "load_case_config",
    "voxelize_arrays",
    "voxelize_events",
]

Config = VoxelConfig

# Descriptive aliases used by some integration code.
EventToTensorVoxelizer = StreamingVoxelizer
EventToTensorVoxelizationOperator = StreamingVoxelizer
EventVoxelOperator = StreamingVoxelizer
EventVoxelizer = StreamingVoxelizer
ContinuousInputHandoff = StreamingVoxelizer
WindowHandoff = StreamingVoxelizer
WindowHandoffManager = StreamingVoxelizer
voxelize_events = voxelize_arrays
events_to_voxels = voxelize_arrays
