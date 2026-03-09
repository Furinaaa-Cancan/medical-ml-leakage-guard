# GateBase Framework & DAG Pipeline — Developer Guide

> **Version**: 2.0.0 | **Last updated**: 2026-03-05

---

## 1. Report Envelope v2.0.0

Every gate now produces a standardized JSON report envelope. The format is defined in `_gate_framework.py::build_report_envelope()`.

### Required fields

| Field | Type | Description |
|---|---|---|
| `envelope_version` | `"2.0.0"` | Format version tag |
| `gate_name` | `str` | Gate identifier matching `_gate_registry.py` key |
| `status` | `"pass"` \| `"fail"` | Binary outcome |
| `strict_mode` | `bool` | Whether `--strict` was active |
| `failure_count` | `int` | Number of failure issues |
| `warning_count` | `int` | Number of warning issues |
| `failures` | `list[dict]` | Failure issues sorted by severity |
| `warnings` | `list[dict]` | Warning issues sorted by severity |
| `execution_timestamp_utc` | `str` | ISO-8601 UTC timestamp |
| `execution_time_seconds` | `float` | Wall-clock gate duration |

### Optional fields

| Field | Type | Description |
|---|---|---|
| `summary` | `dict` | Gate-specific summary data |
| `input_files` | `dict` | Map of input file roles to paths |
| `gate_version` | `str` | Gate implementation version |
| `remediations` | `dict` | Issue code → fix hint mapping |

### `extra` parameter

`build_report_envelope(..., extra={"key": value})` merges extra fields to the **top level** of the envelope via `envelope.update(extra)`. Used by `request_contract_gate` to expose `normalized_request`.

### Issue dict format

Each item in `failures` / `warnings`:

```json
{
  "code": "leak_row_overlap",
  "severity": "error",
  "message": "Human-readable description",
  "details": {"train_rows": 5, "test_rows": 3},
  "remediation": "Remove overlapping rows from test split.",
  "source_file": "train.csv"
}
```

`remediation` and `source_file` are omitted when `None`.

---

## 2. GateIssue & Severity

```python
from _gate_framework import GateIssue, Severity

issue = GateIssue(
    code="leak_row_overlap",
    severity=Severity.ERROR,
    message="Train/test row overlap detected.",
    details={"overlap_count": 5},
    remediation="Remove overlapping rows.",
)
```

**Severity levels** (ranked):

| Severity | Rank | Terminal label |
|---|---|---|
| `CRITICAL` | 0 | `[CRIT]` |
| `ERROR` | 1 | `[FAIL]` |
| `WARNING` | 2 | `[WARN]` |
| `INFO` | 3 | `[INFO]` |

---

## 3. Remediation Registry

Each gate registers remediation hints at module load time:

```python
from _gate_framework import register_remediations

register_remediations({
    "leak_row_overlap": "Remove overlapping patient IDs between train and test.",
    "leak_temporal": "Ensure all test event_time > max(train event_time).",
})
```

These hints are automatically injected into `GateIssue` objects by `build_report_envelope()` and `wrap_legacy_report()`.

---

## 4. Gate Script Lifecycle

Every gate script follows this pattern:

```python
#!/usr/bin/env python3
"""Gate description."""
from _gate_framework import (
    GateIssue, Severity,
    build_report_envelope, get_remediation,
    print_gate_summary, register_remediations,
)
from _gate_utils import get_gate_elapsed, start_gate_timer, write_json

register_remediations({...})

def main():
    args = parse_args()
    failures, warnings = [], []

    # ... gate logic, appending GateIssue objects ...

    status = "fail" if failures else "pass"
    report = build_report_envelope(
        gate_name="my_gate",
        status=status,
        strict_mode=bool(args.strict),
        failures=failures,
        warnings=warnings,
        summary={...},
        input_files={...},
    )
    write_json(Path(args.report), report)
    print_gate_summary(
        gate_name="my_gate",
        status=status,
        failures=failures,
        warnings=warnings,
        strict=bool(args.strict),
        elapsed=get_gate_elapsed(),
    )
    return 2 if (status == "fail" and args.strict) else 0

if __name__ == "__main__":
    start_gate_timer()          # MUST be first in __main__
    raise SystemExit(main())
```

**Key rules**:
- `start_gate_timer()` must be called in `__main__`, not inside `main()`.
- Exit code: `0` = pass, `2` = fail (in strict mode).
- `print_gate_summary()` produces rich terminal output with color-coded issues.

---

## 5. Gate Registry (`_gate_registry.py`)

### GateSpec

Each gate is declared as a `GateSpec`:

```python
GateSpec(
    name="leakage_gate",
    script="leakage_gate.py",
    layer=GateLayer.DATA_VALIDATION,
    description="Detect data leakage between train/test splits.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={},
    report_inputs={},
    report_output="leakage_report.json",
    parallelizable=True,
    category="data_integrity",
)
```

### GateLayer (execution layers)

| Layer | Value | Description |
|---|---|---|
| `REQUEST_CONTRACT` | 0 | Request validation |
| `DATA_FINGERPRINT` | 1 | Data integrity hashing |
| `EXECUTION_ATTESTATION` | 2 | Runtime environment proof |
| `DATA_VALIDATION` | 3 | Split/leakage checks |
| `PROTOCOL_VALIDATION` | 4 | Protocol compliance |
| `MODEL_AUDIT` | 5 | Model selection/tuning audit |
| `METRIC_VALIDATION` | 6 | Performance metric checks |
| `EXTERNAL_VALIDATION` | 7 | External cohort validation |
| `AGGREGATION` | 8 | Publication gate aggregation |
| `SELF_CRITIQUE` | 9 | Meta-review scoring |

### Key functions

| Function | Returns | Description |
|---|---|---|
| `topological_sort()` | `List[str]` | All gates in dependency order |
| `get_execution_layers()` | `List[Tuple[int, List[str]]]` | `(layer_value, gate_names)` tuples |
| `validate_dag()` | `List[str]` | Error messages (empty = valid) |
| `get_runnable_subset(gates, include_dependencies)` | `List[str]` | Subset with optional dependency closure |

---

## 6. DAG Pipeline (`run_dag_pipeline.py`)

### CLI usage

```bash
# Full strict pipeline
python3 run_dag_pipeline.py --request request.json --strict

# Parallel execution (4 workers)
python3 run_dag_pipeline.py --request request.json --strict --parallel --max-workers 4

# Resume from last checkpoint
python3 run_dag_pipeline.py --request request.json --strict --resume

# Re-run only failed gates
python3 run_dag_pipeline.py --request request.json --strict --rerun-failed

# Run a single gate with dependencies
python3 run_dag_pipeline.py --request request.json --strict --only calibration_dca_gate

# Run a single gate without dependencies
python3 run_dag_pipeline.py --request request.json --strict --only calibration_dca_gate --no-deps

# Dry-run (validate inputs, don't execute)
python3 run_dag_pipeline.py --request request.json --strict --dry-run

# Show DAG structure
python3 run_dag_pipeline.py --show-dag
```

### Execution flow

1. **Parse args** and validate DAG structure
2. **Run `request_contract_gate`** (always, even in `--resume`)
3. **Load checkpoint** (if `--resume` or `--rerun-failed`)
4. **Validate checkpoint integrity** — remove passed gates whose report files are missing
5. **Compute runnable subset** based on `--only`, `--from`, `--skip` flags
6. **Execute layer by layer** — gates within a layer run in parallel if `--parallel`
7. **Save checkpoint** after each layer
8. **Write pipeline report** to `dag_pipeline_report.json`

### Pipeline report format

```json
{
  "contract_version": "dag_pipeline_report.v1",
  "status": "pass|fail",
  "strict_mode": true,
  "failure_count": 0,
  "pass_count": 28,
  "skip_count": 0,
  "total_execution_time_seconds": 45.2,
  "evidence_dir": "/path/to/evidence",
  "steps": [
    {
      "name": "leakage_gate",
      "command": "python3 leakage_gate.py ...",
      "exit_code": 0,
      "status": "pass",
      "execution_time_seconds": 1.2,
      "stdout_tail": "...",
      "stderr_tail": ""
    }
  ]
}
```

### Checkpoint/resume

- Checkpoint file: `evidence_dir/dag_checkpoint.json`
- Contains: `passed_gates`, `failed_gates`, `last_run_utc`
- On resume, gates whose report files are missing are automatically invalidated and re-run

---

## 7. Legacy Report Compatibility

`wrap_legacy_report(gate_name, report_dict)` converts old-format reports to v2.0.0 envelope:

- Checks for `envelope_version` key — if present, returns as-is
- Maps `failures[]` items to `GateIssue` with `severity="error"`
- Maps `warnings[]` items to `GateIssue` with `severity="warning"`
- Injects remediation hints from the global registry
- Preserves `summary`, `normalized_request`, and other top-level keys

---

## 8. Testing

### Unit tests

```bash
# Core framework tests (34 tests)
python3 -m pytest tests/test_gate_framework.py -v

# DAG registry tests (24 tests)
python3 -m pytest tests/test_gate_registry.py -v

# Pipeline command-building tests (26 tests)
python3 -m pytest tests/test_run_dag_pipeline.py -v

# E2E smoke tests (14 tests)
python3 -m pytest tests/test_dag_pipeline_e2e.py -v

# All new tests
python3 -m pytest tests/test_gate_framework.py tests/test_gate_registry.py tests/test_run_dag_pipeline.py tests/test_dag_pipeline_e2e.py -v
```

### What the tests cover

| Test file | Coverage |
|---|---|
| `test_gate_framework.py` | Severity, GateIssue, remediation registry, build_report_envelope, wrap_legacy_report, validate_input_files, format_issue_line, load_gate_report |
| `test_gate_registry.py` | Registry integrity (29 gates), DAG validation, topological sort, execution layers, dependencies, runnable subset |
| `test_run_dag_pipeline.py` | Gate-specific CLI extras, standard/aggregation command building, checkpoint save/load/integrity, run_gate_subprocess |
| `test_dag_pipeline_e2e.py` | --show-dag, dry-run behavior, request_contract_gate envelope format, --only flag, pipeline report format, CLI validation |
