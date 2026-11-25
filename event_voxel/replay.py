"""Replay an exchange capture through the streaming operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import VoxelConfig
from .stream import StreamingVoxelizer


def load_exchange(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    records = envelope.get("records", envelope)
    arrays: dict[str, np.ndarray] = {}
    for key, descriptor in envelope.get("tensors", {}).items():
        value = np.asarray(descriptor["values"], dtype=descriptor.get("dtype", "int64"))
        shape = tuple(descriptor.get("shape", value.shape))
        if shape != value.shape:
            raise ValueError(f"tensor {key} has shape {value.shape}, declared {shape}")
        arrays[key] = value
    return records, arrays


def replay_records(
    records: dict[str, Any],
    arrays: dict[str, np.ndarray],
    capacity: dict[str, Any] | None = None,
    *,
    auto_claim: bool = True,
    active_calibration: str | None = None,
    **config_overrides: Any,
) -> StreamingVoxelizer:
    config = VoxelConfig.from_mapping(records, capacity, **config_overrides)
    initial = active_calibration or records.get("active_calibration")
    if initial is None and records.get("calibrations"):
        initial = next(iter(records["calibrations"]))
    stream = StreamingVoxelizer(
        config,
        records.get("calibrations"),
        active_calibration=initial,
        overlap_owner=records.get("overlap_owner"),
    )
    stream.replay(records.get("submissions", []), arrays, auto_claim=auto_claim)
    return stream


def _summary(stream: StreamingVoxelizer, include_arrays: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "active_windows": stream.bank.indices(),
        "next_window": stream.next_window,
        "progress": stream.progress,
        "active_calibration": stream.active_calibration,
        "windows": {},
        "deliveries": {},
    }
    for idx in stream.bank.indices():
        view = stream.window(idx)
        item = {
            "ready": view.ready,
            "usable": view.usable,
            "conflict": view.conflict,
            "reasons": view.reasons,
            "horizons": view.horizons,
            "boundary_loss": view.boundary_loss,
            "coverage_all": bool(view.coverage.all()),
            "coverage_count": int(view.coverage.sum()),
        }
        if include_arrays:
            item["voxels"] = view.voxels.tolist()
            item["coverage"] = view.coverage.tolist()
            item["effective_coverage"] = view.effective_coverage.tolist()
            item["zero_coverage"] = view.zero_coverage.tolist()
            item["coverage_status"] = view.coverage_status.tolist()
            item["boundary_loss_by_time"] = view.boundary_loss_by_time.tolist()
        out["windows"][str(idx)] = item

    for did, delivery in stream.deliveries.items():
        item = delivery.as_dict()
        if not include_arrays:
            item.pop("voxels", None)
            item.pop("coverage", None)
            item.pop("effective_coverage", None)
            item.pop("zero_coverage", None)
            item.pop("coverage_status", None)
            item.pop("boundary_loss_by_time", None)
        else:
            item["voxels"] = delivery.voxels.tolist()
            item["coverage"] = delivery.coverage.tolist()
            item["effective_coverage"] = delivery.effective_coverage.tolist()
            item["zero_coverage"] = delivery.zero_coverage.tolist()
            item["coverage_status"] = delivery.coverage_status.tolist()
            item["boundary_loss_by_time"] = delivery.boundary_loss_by_time.tolist()
        # as_dict already put arrays; make JSON safe.
        out["deliveries"][did] = item
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay an event capture and summarize handoffs")
    parser.add_argument("capture", type=Path)
    parser.add_argument("--capacity", type=Path, default=None)
    parser.add_argument("--no-auto-claim", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    records, arrays = load_exchange(args.capture)
    capacity = None
    cap_path = args.capacity or args.capture.parent / "capacity.json"
    if cap_path.exists():
        capacity = json.loads(cap_path.read_text(encoding="utf-8"))
    stream = replay_records(
        records, arrays, capacity, auto_claim=not args.no_auto_claim
    )
    summary = _summary(stream, include_arrays=not args.summary_only)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
