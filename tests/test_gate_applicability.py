"""Tests for scripts/gate_applicability.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gate_applicability import GateApplicability, SUPPORTED_TYPES


class TestGateApplicabilityInit:
    def test_binary_default(self) -> None:
        ga = GateApplicability()
        assert ga.prediction_type == "binary"

    def test_all_supported_types(self) -> None:
        for pt in SUPPORTED_TYPES:
            ga = GateApplicability(pt)
            assert ga.prediction_type == pt

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="prediction_type"):
            GateApplicability("unknown_type")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            GateApplicability("")


class TestGateModeAndFiltering:
    def test_binary_all_gates_full(self) -> None:
        ga = GateApplicability("binary")
        # All 31 gates should be FULL for binary
        applicable = ga.applicable_gates()
        assert len(applicable) == 31

    def test_binary_no_skipped(self) -> None:
        ga = GateApplicability("binary")
        assert ga.skipped_gates() == []

    def test_regression_has_na_gate(self) -> None:
        ga = GateApplicability("regression")
        skipped = ga.skipped_gates()
        assert len(skipped) >= 1, "Regression should have at least one N/A gate"

    def test_multiclass_fewer_applicable_than_binary(self) -> None:
        ga_bin = GateApplicability("binary")
        ga_mc = GateApplicability("multiclass")
        assert len(ga_mc.applicable_gates()) < len(ga_bin.applicable_gates())

    def test_gate_mode_returns_string(self) -> None:
        ga = GateApplicability("binary")
        mode = ga.gate_mode("leakage_gate")
        assert isinstance(mode, str)
        assert mode in ("FULL", "ADAPTED", "REPLACED", "NEW", "NA", "CONDITIONAL", "UNKNOWN")

    def test_unknown_gate_returns_unknown(self) -> None:
        ga = GateApplicability("binary")
        assert ga.gate_mode("nonexistent_gate_xyz") == "UNKNOWN"

    def test_skip_gate_true_for_na(self) -> None:
        ga = GateApplicability("regression")
        # At least one gate is N/A for regression
        skipped = ga.skipped_gates()
        for g in skipped:
            assert ga.skip_gate(g) is True

    def test_is_runnable_true_for_full(self) -> None:
        ga = GateApplicability("binary")
        assert ga.is_runnable("leakage_gate") is True

    def test_replaced_gates_not_runnable(self) -> None:
        ga = GateApplicability("multiclass")
        replaced = ga.replaced_gates()
        for g in replaced:
            assert ga.is_runnable(g) is False

    def test_summary_fields(self) -> None:
        ga = GateApplicability("multiclass")
        s = ga.summary()
        assert "prediction_type" in s
        assert "total_gates" in s
        assert "applicable" in s
        assert "skipped_na" in s
        assert "replaced_planned" in s
        assert "by_mode" in s

    def test_summary_totals_consistent(self) -> None:
        for pt in SUPPORTED_TYPES:
            ga = GateApplicability(pt)
            s = ga.summary()
            total = s["total_gates"]
            # Sum of all modes should equal total gates
            by_mode_count = sum(len(v) for v in s["by_mode"].values())
            assert by_mode_count == total, f"{pt}: mode counts {by_mode_count} != total {total}"

    def test_adaptation_notes_string(self) -> None:
        ga = GateApplicability("multiclass")
        # request_contract_gate has adaptation notes for multiclass
        notes = ga.adaptation_notes("request_contract_gate")
        assert isinstance(notes, str)

    def test_adaptation_notes_empty_for_binary(self) -> None:
        ga = GateApplicability("binary")
        # Binary gates need no adaptation
        notes = ga.adaptation_notes("leakage_gate")
        # Could be empty string for FULL gates
        assert isinstance(notes, str)

    def test_survival_applicable_count(self) -> None:
        ga = GateApplicability("survival")
        # Survival has some adapted gates but none should be N/A
        applicable = ga.applicable_gates()
        assert len(applicable) >= 28  # FULL + ADAPTED + CONDITIONAL
