"""Compatibility import path for streaming window handoff."""

from event_voxel.stream import (
    ActiveWindowLimitError,
    ConflictMarker,
    StreamingVoxelizer,
    WindowDelivery,
    WindowNotReadyError,
    WindowState,
    WindowView,
)

WindowHandoff = StreamingVoxelizer
StreamHandoff = StreamingVoxelizer
EventHandoff = StreamingVoxelizer

__all__ = [
    "ActiveWindowLimitError",
    "ConflictMarker",
    "EventHandoff",
    "StreamHandoff",
    "StreamingVoxelizer",
    "WindowDelivery",
    "WindowHandoff",
    "WindowNotReadyError",
    "WindowState",
    "WindowView",
]
