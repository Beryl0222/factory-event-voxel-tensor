"""Compatibility spelling for continuous-input window handoff."""

from event_voxel.stream import (
    ConflictMarker,
    StreamingVoxelizer,
    WindowDelivery,
    WindowNotReadyError,
    WindowView,
)
from event_voxel.windows import ActiveWindowLimitError

EventStream = StreamingVoxelizer
StreamHandoff = StreamingVoxelizer

__all__ = [
    "ActiveWindowLimitError",
    "ConflictMarker",
    "EventStream",
    "StreamHandoff",
    "StreamingVoxelizer",
    "WindowDelivery",
    "WindowNotReadyError",
    "WindowView",
]
