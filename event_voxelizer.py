"""Descriptive top-level alias for the event-to-voxel operator."""

from voxelizer import *  # noqa: F401,F403
from voxelizer import (  # noqa: F401
    EventVoxelizer,
    StreamingVoxelizer,
    VoxelConfig,
    voxelize_arrays,
)

EventToVoxel = StreamingVoxelizer
EventToTensor = StreamingVoxelizer
EventToTensorVoxelizer = StreamingVoxelizer

__all__ = [
    "EventToVoxel",
    "EventToTensor",
    "EventToTensorVoxelizer",
    "EventVoxelizer",
    "StreamingVoxelizer",
    "VoxelConfig",
    "voxelize_arrays",
]
