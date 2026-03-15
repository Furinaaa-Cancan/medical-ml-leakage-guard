"""
gate_applicability.py — MLGG gate applicability resolver.

Given a prediction type (binary/multiclass/regression/survival), determines
which of the 31 gates are applicable and in what mode (FULL/ADAPTED/NA/etc.).

Used by:
- generate_audit_report.py: filter audit to relevant gates
- run_dag_pipeline.py: skip inapplicable gates instead of failing
- mlgg.py workflow: adjust gate ordering per prediction type

Usage:
    from gate_applicability import GateApplicability
    ga = GateApplicability("multiclass")
    print(ga.applicable_gates())       # gates to run
    print(ga.gate_mode("leakage_gate"))  # "FULL" | "ADAPTED" | "NA" | ...
    print(ga.skip_gate("clinical_metrics_gate"))  # True if NA
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_REFERENCES_DIR = Path(__file__).parent.parent / "references"
_MATRIX_PATH = _REFERENCES_DIR / "gate-applicability-matrix.json"

# Valid prediction types
SUPPORTED_TYPES = frozenset(["binary", "multiclass", "regression", "survival"])

# Applicability codes that mean "run this gate"
_RUNNABLE = frozenset(["FULL", "ADAPTED", "CONDITIONAL"])
# Codes that mean "skip entirely"
_SKIP = frozenset(["NA"])
# Codes that mean "replaced by a different gate (planned)"
_REPLACED = frozenset(["REPLACED", "NEW"])


class GateApplicability:
    """Resolver for MLGG gate applicability given a prediction type."""

    def __init__(self, prediction_type: str = "binary") -> None:
        if prediction_type not in SUPPORTED_TYPES:
            raise ValueError(
                f"prediction_type must be one of {sorted(SUPPORTED_TYPES)}, "
                f"got {prediction_type!r}"
            )
        self.prediction_type = prediction_type
        self._matrix: Optional[Dict[str, Dict[str, str]]] = None

    def _load_matrix(self) -> Dict[str, Dict[str, str]]:
        if self._matrix is None:
            try:
                with _MATRIX_PATH.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                self._matrix = raw.get("core_gates", {})
            except Exception:
                self._matrix = {}
        return self._matrix

    def gate_mode(self, gate_name: str) -> str:
        """Return the applicability code for this gate and prediction type.

        Returns one of: FULL, ADAPTED, REPLACED, NEW, NA, CONDITIONAL, UNKNOWN.
        """
        matrix = self._load_matrix()
        gate_info = matrix.get(gate_name)
        if gate_info is None:
            return "UNKNOWN"
        return str(gate_info.get(self.prediction_type, "UNKNOWN"))

    def skip_gate(self, gate_name: str) -> bool:
        """Return True if this gate should be skipped for the current prediction type."""
        mode = self.gate_mode(gate_name)
        return mode in _SKIP

    def is_runnable(self, gate_name: str) -> bool:
        """Return True if this gate should be executed (FULL, ADAPTED, CONDITIONAL)."""
        mode = self.gate_mode(gate_name)
        return mode in _RUNNABLE

    def is_replaced(self, gate_name: str) -> bool:
        """Return True if this gate is replaced by a type-specific gate (planned v1.1+)."""
        mode = self.gate_mode(gate_name)
        return mode in _REPLACED

    def applicable_gates(self) -> List[str]:
        """Return list of gate names that should run for this prediction type."""
        matrix = self._load_matrix()
        return [
            gate for gate in matrix
            if self.is_runnable(gate)
        ]

    def skipped_gates(self) -> List[str]:
        """Return gate names that are N/A for this prediction type."""
        matrix = self._load_matrix()
        return [gate for gate in matrix if self.skip_gate(gate)]

    def replaced_gates(self) -> List[str]:
        """Return gate names replaced by type-specific alternatives (planned)."""
        matrix = self._load_matrix()
        return [gate for gate in matrix if self.is_replaced(gate)]

    def adaptation_notes(self, gate_name: str) -> str:
        """Return human-readable adaptation note for a gate (if any)."""
        matrix = self._load_matrix()
        gate_info = matrix.get(gate_name, {})
        notes: Any = gate_info.get("adaptation_notes", {})
        if isinstance(notes, dict):
            return str(notes.get(self.prediction_type, ""))
        return ""

    def summary(self) -> Dict[str, object]:
        """Return applicability summary dict for reporting."""
        matrix = self._load_matrix()
        by_mode: Dict[str, List[str]] = {}
        for gate in matrix:
            mode = self.gate_mode(gate)
            by_mode.setdefault(mode, []).append(gate)

        return {
            "prediction_type": self.prediction_type,
            "total_gates": len(matrix),
            "applicable": len(self.applicable_gates()),
            "skipped_na": len(self.skipped_gates()),
            "replaced_planned": len(self.replaced_gates()),
            "by_mode": by_mode,
        }
