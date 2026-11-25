"""Behavioral tests for voxel weights, ownership, progress and handoff."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from event_voxel import (
    CalibrationBook,
    StreamingVoxelizer,
    VoxelConfig,
    voxelize_arrays,
)
from event_voxel.windows import ActiveWindowLimitError
from caseio import load_case

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def small_config(**overrides):
    defaults = dict(
        shape_yx=(1, 1),
        window_bins=4,
        time_step=4,
        max_active_windows=4,
    )
    defaults.update(overrides)
    return VoxelConfig(**defaults)


class WeightTests(unittest.TestCase):
    def test_linear_weights_split_across_two_time_windows(self):
        cfg = small_config()
        cal = {
            "effective": [0, 100],
            "scale": [1, 1],
            "shift": [0, 0],
            "owner": [0, 1, 0, 1],
            "offset": 0,
            "line_delay": 0,
        }
        events = np.array([[1, 0, 0, 15, 1]], dtype=np.int64)
        result = voxelize_arrays(events, cfg, calibration=cal, sensor="s")
        self.assertEqual(set(result.windows), {0, 1})
        self.assertAlmostEqual(result.windows[0][1, 3, 0, 0], 0.25)
        self.assertAlmostEqual(result.windows[1][1, 0, 0, 0], 0.75)
        self.assertAlmostEqual(result.stats["boundary_loss"], 0.0)
        self.assertLess(result.mass_balance_error(), 1e-12)

    def test_spatial_clipping_is_counted_not_dropped_silently(self):
        records, tensors = load_case(DATA / "capture.json")
        cfg = VoxelConfig.from_mapping(records)
        east_v1 = records["calibrations"]["v1"]["east"]
        result = voxelize_arrays(
            tensors["a"], cfg, calibration=east_v1, sensor="east"
        )
        # Two y=3.25 events each lose one quarter of their spatial mass.
        self.assertAlmostEqual(result.stats["boundary_loss"], 0.5)
        self.assertAlmostEqual(result.boundary_loss_by_window[0], 0.0625)
        self.assertAlmostEqual(result.boundary_loss_by_window[1], 0.4375)
        self.assertLess(result.mass_balance_error(), 1e-10)


class IdentityAndOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.records, self.tensors = load_case(DATA / "capture.json")
        self.cfg = VoxelConfig.from_mapping(self.records)
        self.stream = StreamingVoxelizer(
            self.cfg, self.records["calibrations"], active_calibration="v1"
        )

    def test_sensor_resend_counts_once_but_other_sensor_ids_do_not(self):
        r1 = self.stream.ingest_events("east", self.tensors["a"], "e1")
        r2 = self.stream.ingest_events("west", self.tensors["b"], "w1")
        r3 = self.stream.ingest_events("east", self.tensors["a"], "e1-resend")

        self.assertEqual(r1.accepted, 3)
        self.assertEqual(r2.received, 3)
        self.assertEqual(r3.duplicates, 3)
        self.assertEqual(len(self.stream.seen["east"]), 3)
        self.assertEqual(len(self.stream.seen["west"]), 3)

    def test_spatial_owner_rule_rejects_non_owner_without_tensor_dedup(self):
        report = self.stream.ingest_events("west", self.tensors["b"], "w1")
        # west event id 2 projects to (0, 3), inside east's owner box.
        self.assertEqual(report.owner_rejected, 1)
        self.assertEqual(report.accepted, 2)
        self.assertIn(2, report.rejected_ids)

        # A different head with the same numeric event id is not an identity
        # duplicate; ownership, not tensor equality, decides inclusion.
        east = self.stream.ingest_events("east", self.tensors["a"], "e1")
        self.assertEqual(east.duplicates, 0)


class ProgressTests(unittest.TestCase):
    def test_slowest_owned_row_controls_no_event_promise(self):
        cfg = small_config(shape_yx=(4, 1))
        book = CalibrationBook(
            {
                "v": {
                    "s": {
                        "effective": [0, 80],
                        "scale": [1, 1],
                        "shift": [0, 0],
                        "owner": [0, 4, 0, 1],
                        "offset": -3,
                        "line_delay": 3,
                    }
                }
            }
        )
        stream = StreamingVoxelizer(cfg, book, active_calibration="v")
        stream.ingest_events("s", np.empty((0, 5), dtype=np.int64))

        stream.progress["s"] = 16
        self.assertFalse(stream.ready(0))
        complete, horizons, reasons = stream._readiness(0)
        self.assertAlmostEqual(horizons["s"], 13.0)
        self.assertTrue(reasons)

        stream.update_progress("s", 19)
        self.assertAlmostEqual(stream.horizons()["s"], 16.0)
        self.assertTrue(stream.ready(0))


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.records, self.tensors = load_case(DATA / "capture.json")
        self.capacity = json.loads((DATA / "capacity.json").read_text())
        self.cfg = VoxelConfig.from_mapping(self.records, self.capacity)
        self.stream = StreamingVoxelizer(
            self.cfg, self.records["calibrations"], active_calibration="v1"
        )

    def _first_five(self):
        for submission in self.records["submissions"][:5]:
            self.stream.submit(submission, self.tensors)

    def test_handoff_has_full_coverage_despite_boundary_clipping(self):
        self._first_five()
        self.assertTrue(self.stream.ready(0))
        delivery = self.stream.claim_window(0)

        self.assertEqual(delivery.voxels.shape, (2, 4, 4, 6))
        self.assertTrue(delivery.coverage.all())
        self.assertTrue(delivery.effective_coverage().all())
        self.assertAlmostEqual(delivery.boundary_loss, 0.0625)
        self.assertTrue(delivery.usable)

        expected_negative_t3 = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.28125, 0.28125, 0],
                [0, 0, 0, 0.09375, 0.09375, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(delivery.voxels[0, 3], expected_negative_t3)
        np.testing.assert_allclose(
            delivery.voxels[1, 3, 3, 2:4], [0.09375, 0.09375]
        )

    def test_handed_revision_is_immutable_and_conflicted(self):
        self._first_five()
        old = self.stream.claim_window(0)
        old_values = old.voxels.copy()

        self.stream.activate_calibration("v2")
        self.stream.update_progress("east", 16)

        # The handed tensor is frozen and remains the old v1 distribution.
        with self.assertRaises(ValueError):
            old.voxels[0, 0, 0, 0] = 999
        np.testing.assert_allclose(old.voxels, old_values)
        self.assertTrue(old.conflict)
        self.assertFalse(old.usable)
        kinds = {c["kind"] for c in old.conflicts}
        self.assertIn("calibration_revision", kinds)
        self.assertIn("progress_revision", kinds)

        # The unhanded/current window is not inferable after the rewind.
        view = self.stream.window(0)
        self.assertFalse(view.complete)
        self.assertFalse(view.usable)

    def test_unhanded_window_is_moved_not_duplicated(self):
        self._first_five()
        self.stream.claim_window(0)
        self.stream.activate_calibration("v2")
        self.stream.update_progress("east", 16)
        self.stream.submit(self.records["submissions"][7], self.tensors)
        self.stream.update_progress("east", 36)
        self.stream.update_progress("west", 37)

        view0 = self.stream.window(0)
        self.assertTrue(view0.complete)
        # v2 moves east event 2 from t=13 fractional splat to exact t=12.
        self.assertAlmostEqual(view0.boundary_loss, 0.0)
        np.testing.assert_allclose(
            view0.voxels[0, 3, 1:3, 3:5],
            np.array([[0.375, 0.375], [0.125, 0.125]]),
        )

        delivery1 = self.stream.claim_window(1, copy=True)
        self.assertAlmostEqual(delivery1.boundary_loss, 0.5)
        self.assertAlmostEqual(float(delivery1.voxels.sum()), 4.5)
        total = sum(
            s.allocated_mass + s.boundary_loss for s in self.stream.bank.states()
        )
        # Four unique east events + two owner-valid west events.
        self.assertAlmostEqual(total, 6.0, places=10)


class CapacityConfigTests(unittest.TestCase):
    def test_capacity_values_drive_active_space_and_chunk_budget(self):
        capacity = json.loads((DATA / "capacity.json").read_text())
        cfg = VoxelConfig.from_mapping(
            {"shape_y_x": [240, 432], "window_size": 9}, capacity
        )
        self.assertEqual(cfg.max_active_windows, 4)
        self.assertEqual(cfg.scratch_budget_bytes, 6 * 1024 * 1024)
        self.assertEqual(cfg.chunk_size, cfg.scratch_budget_bytes // 512)
        self.assertLessEqual(
            cfg.chunk_size * 512, cfg.scratch_budget_bytes + 512
        )


class CoverageSemanticsTests(unittest.TestCase):
    def test_zero_coverage_differs_from_uncovered_and_incomplete(self):
        cfg = small_config(shape_yx=(2, 2), max_active_windows=2)
        book = CalibrationBook(
            {
                "v": {
                    "s": {
                        "effective": [0, 100],
                        "owner": [0, 2, 0, 1],
                    }
                }
            },
            active_version="v",
        )
        stream = StreamingVoxelizer(cfg, book)
        stream.ingest_events(
            "s", np.array([[1, 0, 0, 0, 1]], dtype=np.int64)
        )

        incomplete = stream.window(0)
        np.testing.assert_array_equal(
            incomplete.coverage, [[True, False], [True, False]]
        )
        self.assertFalse(incomplete.effective_coverage.any())
        np.testing.assert_array_equal(
            incomplete.coverage_status, [[-1, 0], [-1, 0]]
        )

        stream.update_progress("s", 100)
        complete = stream.window(0)
        self.assertTrue(complete.complete)
        np.testing.assert_array_equal(
            complete.effective_coverage, [[True, False], [True, False]]
        )
        np.testing.assert_array_equal(
            complete.coverage_status, [[1, 0], [1, 0]]
        )
        # The uncovered column stays zero but is not marked effective coverage.
        self.assertEqual(float(complete.voxels[:, :, :, 1].sum()), 0.0)


class ActiveSpaceTests(unittest.TestCase):
    def test_bank_does_not_allocate_full_timeline(self):
        cfg = VoxelConfig(
            shape_yx=(1, 1),
            window_bins=1,
            time_step=1,
            max_active_windows=1,
        )
        stream = StreamingVoxelizer(
            cfg,
            {
                "v": {
                    "s": {
                        "effective": [0, 100],
                        "scale": [1, 1],
                        "shift": [0, 0],
                        "owner": [0, 1, 0, 1],
                    }
                }
            },
            active_calibration="v",
        )
        stream.ingest_events(
            "s", np.array([[1, 0, 0, 0, 1]], dtype=np.int64)
        )
        self.assertEqual(stream.bank.indices(), [0])

        with self.assertRaises(ActiveWindowLimitError):
            stream.ingest_events(
                "s", np.array([[2, 0, 0, 10, 1]], dtype=np.int64)
            )

        # Transfer window 0 to the caller, freeing the single active slot.
        stream.progress["s"] = 1
        d0 = stream.claim_window(0, require_ready=False, copy=False)
        self.assertNotIn(0, stream.bank)
        self.assertEqual(d0.voxels[1, 0, 0, 0], 1.0)

        # The stored event can now be rebuilt into the new active window.
        stream.ingest_events(
            "s", np.array([[2, 0, 0, 10, 1]], dtype=np.int64)
        )
        self.assertIn(10, stream.bank)
        self.assertNotIn(0, stream.bank)


if __name__ == "__main__":
    unittest.main()
