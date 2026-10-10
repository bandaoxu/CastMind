"""Regression: cached case-library load must not NameError on AnalyzeResult."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class LoadCachedAnalysisTests(unittest.TestCase):
    def test_load_cached_analysis_returns_analyze_result(self):
        from castmind.tools.analysis import AnalyzeResult
        from run_experiment import _load_cached_analysis

        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp)
            (lib / "memory.json").write_text(
                json.dumps({"frequency": "h", "periodicity_lag": 24}),
                encoding="utf-8",
            )
            result = _load_cached_analysis(str(lib))
            self.assertIsInstance(result, AnalyzeResult)
            self.assertEqual(result.memory.get("frequency"), "h")
            self.assertEqual(result.case_base, [])
            self.assertEqual(result.case_neighbors, [])


if __name__ == "__main__":
    unittest.main()
