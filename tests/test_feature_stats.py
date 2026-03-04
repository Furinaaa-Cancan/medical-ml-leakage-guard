"""Unit tests for compute_feature_stats (mlgg_pixel) and compute_feature_summary (mlgg_interactive)."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import unittest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── mlgg_pixel.compute_feature_stats ──

class TestComputeFeatureStats(unittest.TestCase):
    """Tests for mlgg_pixel.compute_feature_stats."""

    @classmethod
    def setUpClass(cls):
        # Patch _TEST_MODE before import to avoid terminal side-effects
        os.environ["MLGG_PIXEL_TEST_MODE"] = "1"
        import mlgg_pixel
        cls.compute = staticmethod(mlgg_pixel.compute_feature_stats)

    def _tmp_csv(self, rows):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        self.addCleanup(os.unlink, f.name)
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        f.close()
        return Path(f.name)

    def test_basic_numeric_features(self):
        rows = [
            {"y": "1", "age": "30", "bp": "120"},
            {"y": "0", "age": "40", "bp": "130"},
            {"y": "1", "age": "50", "bp": "140"},
            {"y": "0", "age": "60", "bp": "150"},
            {"y": "1", "age": "35", "bp": "125"},
            {"y": "0", "age": "45", "bp": "135"},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["age", "bp"], "y")
        self.assertIn("age", result)
        self.assertIn("bp", result)
        self.assertTrue(result["age"]["is_numeric"])
        self.assertTrue(result["bp"]["is_numeric"])
        self.assertEqual(result["age"]["missing_pct"], 0.0)
        self.assertIsNotNone(result["age"]["variance"])
        self.assertGreater(result["age"]["variance"], 0)
        self.assertIsNotNone(result["age"]["corr_target"])

    def test_categorical_feature(self):
        rows = [
            {"y": "1", "gender": "M"},
            {"y": "0", "gender": "F"},
            {"y": "1", "gender": "M"},
            {"y": "0", "gender": "F"},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["gender"], "y")
        self.assertFalse(result["gender"]["is_numeric"])
        self.assertIsNone(result["gender"]["variance"])
        self.assertIsNone(result["gender"]["corr_target"])

    def test_missing_values(self):
        rows = [
            {"y": "1", "x": "10"},
            {"y": "0", "x": ""},
            {"y": "1", "x": "30"},
            {"y": "0", "x": ""},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["x"], "y")
        self.assertEqual(result["x"]["missing_pct"], 50.0)

    def test_constant_feature(self):
        rows = [
            {"y": "1", "x": "5"},
            {"y": "0", "x": "5"},
            {"y": "1", "x": "5"},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["x"], "y")
        self.assertTrue(result["x"]["is_numeric"])
        self.assertAlmostEqual(result["x"]["variance"], 0.0)
        self.assertIn("constant", result["x"]["warnings"])

    def test_high_missing_warning(self):
        rows = [{"y": "1", "x": ""}] * 7 + [{"y": "0", "x": "10"}] * 3
        p = self._tmp_csv(rows)
        result = self.compute(p, ["x"], "y")
        self.assertGreater(result["x"]["missing_pct"], 60.0)
        self.assertIn("high_miss", result["x"]["warnings"])

    def test_empty_csv(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        self.addCleanup(os.unlink, f.name)
        f.write("y,x\n")
        f.close()
        result = self.compute(Path(f.name), ["x"], "y")
        self.assertEqual(result, {})

    def test_nonexistent_file(self):
        result = self.compute(Path("/nonexistent/file.csv"), ["x"], "y")
        self.assertEqual(result, {})

    def test_correlation_direction(self):
        rows = [{"y": str(i % 2), "x": str(i)} for i in range(20)]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["x"], "y")
        self.assertIsNotNone(result["x"]["corr_target"])

    def test_distinct_count(self):
        rows = [
            {"y": "1", "x": "a"},
            {"y": "0", "x": "b"},
            {"y": "1", "x": "a"},
            {"y": "0", "x": "c"},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["x"], "y")
        self.assertEqual(result["x"]["distinct"], 3)


# ── mlgg_interactive.compute_feature_summary ──

class TestComputeFeatureSummary(unittest.TestCase):
    """Tests for mlgg_interactive.compute_feature_summary."""

    @classmethod
    def setUpClass(cls):
        import mlgg_interactive
        cls.compute = staticmethod(mlgg_interactive.compute_feature_summary)

    def _tmp_csv(self, rows):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        self.addCleanup(os.unlink, f.name)
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        f.close()
        return f.name

    def test_basic_numeric(self):
        rows = [
            {"y": "1", "age": "30", "bp": "120"},
            {"y": "0", "age": "40", "bp": "130"},
            {"y": "1", "age": "50", "bp": "140"},
            {"y": "0", "age": "60", "bp": "150"},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["age", "bp"], "y")
        self.assertIn("age", result)
        self.assertTrue(result["age"]["is_numeric"])
        self.assertIsNotNone(result["age"]["variance"])

    def test_categorical(self):
        rows = [
            {"y": "1", "gender": "M"},
            {"y": "0", "gender": "F"},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["gender"], "y")
        self.assertFalse(result["gender"]["is_numeric"])

    def test_missing_pct(self):
        rows = [
            {"y": "1", "x": "10"},
            {"y": "0", "x": ""},
            {"y": "1", "x": "30"},
            {"y": "0", "x": ""},
        ]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["x"], "y")
        self.assertEqual(result["x"]["missing_pct"], 50.0)

    def test_correlation(self):
        rows = [{"y": str(i % 2), "x": str(i * 10)} for i in range(10)]
        p = self._tmp_csv(rows)
        result = self.compute(p, ["x"], "y")
        self.assertIsNotNone(result["x"]["corr_target"])

    def test_empty_file(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        self.addCleanup(os.unlink, f.name)
        f.write("y,x\n")
        f.close()
        result = self.compute(f.name, ["x"], "y")
        self.assertEqual(result, {})

    def test_nonexistent(self):
        result = self.compute("/no/such/file.csv", ["x"], "y")
        self.assertEqual(result, {})


# ── _feature_hint_from_stats ──

class TestFeatureHintFromStats(unittest.TestCase):
    """Tests for mlgg_pixel._feature_hint_from_stats."""

    @classmethod
    def setUpClass(cls):
        os.environ["MLGG_PIXEL_TEST_MODE"] = "1"
        import mlgg_pixel
        cls.hint = staticmethod(mlgg_pixel._feature_hint_from_stats)
        mlgg_pixel.LANG = "en"

    def test_numeric_with_stats(self):
        stats = {"x": {"is_numeric": True, "missing_pct": 5.0, "variance": 10.5, "corr_target": 0.3, "distinct": 100, "warnings": []}}
        h = self.hint(stats, "x")
        self.assertIn("num", h)
        self.assertIn("miss=5.0%", h)
        self.assertIn("var=10.5", h)
        self.assertIn("corr=+0.300", h)

    def test_categorical(self):
        stats = {"x": {"is_numeric": False, "missing_pct": 0.0, "variance": None, "corr_target": None, "distinct": 3, "warnings": []}}
        h = self.hint(stats, "x")
        self.assertIn("cat", h)
        self.assertNotIn("var=", h)

    def test_constant_warning(self):
        stats = {"x": {"is_numeric": True, "missing_pct": 0.0, "variance": 0.0, "corr_target": None, "distinct": 1, "warnings": ["constant"]}}
        h = self.hint(stats, "x")
        self.assertIn("CONSTANT", h)

    def test_time_column_hint(self):
        stats = {"time_col": {"is_numeric": True, "missing_pct": 0.0, "variance": 100.0, "corr_target": 0.1, "distinct": 50, "warnings": []}}
        h = self.hint(stats, "time_col", detected_time="time_col")
        self.assertIn("time", h.lower())

    def test_unknown_column(self):
        h = self.hint({}, "unknown_col")
        self.assertEqual(h, "")

    def test_high_missing(self):
        stats = {"x": {"is_numeric": True, "missing_pct": 75.0, "variance": 1.0, "corr_target": None, "distinct": 10, "warnings": ["high_miss"]}}
        h = self.hint(stats, "x")
        self.assertIn("HIGH-MISS", h)


# ── _format_feature_line ──

class TestFormatFeatureLine(unittest.TestCase):
    """Tests for mlgg_interactive._format_feature_line."""

    @classmethod
    def setUpClass(cls):
        import mlgg_interactive
        cls.fmt = staticmethod(mlgg_interactive._format_feature_line)

    def test_numeric_full(self):
        info = {"is_numeric": True, "missing_pct": 3.5, "variance": 42.0, "corr_target": -0.15, "distinct": 50}
        line = self.fmt("age", info)
        self.assertIn("age", line)
        self.assertIn("num", line)
        self.assertIn("miss=3.5%", line)
        self.assertIn("var=42", line)
        self.assertIn("corr=-0.150", line)

    def test_categorical(self):
        info = {"is_numeric": False, "missing_pct": 0.0, "variance": None, "corr_target": None, "distinct": 2}
        line = self.fmt("gender", info)
        self.assertIn("cat", line)
        self.assertNotIn("var=", line)

    def test_constant(self):
        info = {"is_numeric": True, "missing_pct": 0.0, "variance": 0.0, "corr_target": None, "distinct": 1}
        line = self.fmt("c", info)
        self.assertIn("CONST", line)


if __name__ == "__main__":
    unittest.main()
