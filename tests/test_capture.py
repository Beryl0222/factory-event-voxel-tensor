"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        for key in ("a", "b", "late"):
            self.assertEqual(a[key].shape[1], 5)
            self.assertTrue(np.isin(a[key][:, 4], [-1, 1]).all())
        self.assertNotEqual(r["calibrations"]["v1"]["east"]["line_delay"],
                            r["calibrations"]["v2"]["east"]["line_delay"])
