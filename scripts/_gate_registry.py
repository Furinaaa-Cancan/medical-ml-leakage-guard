"""
Gate registry and DAG dependency graph for ml-leakage-guard.

Declares every gate's metadata, input/output contracts, and inter-gate
dependencies. The DAG executor uses this to:
  - resolve execution order
  - maximize parallelism
  - enable incremental / single-step re-runs
  - validate wiring before execution

Each GateSpec is a declarative description; it does NOT import or run gate
code. Gate scripts remain independently executable.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Gate execution layer (for parallelism grouping)
# ---------------------------------------------------------------------------

class GateLayer(enum.IntEnum):
    """Logical execution layers. Lower numbers run first."""

    CONTRACT = 0
    MANIFEST = 1
    ATTESTATION = 2
    DATA_VALIDATION = 3
    POLICY_AUDIT = 4
    MODEL_AUDIT = 5
    METRIC_VALIDATION = 6
    AGGREGATION = 7
    FINAL = 8


# ---------------------------------------------------------------------------
# GateSpec: declarative gate metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateSpec:
    """Declarative specification for a single gate."""

    name: str
    script: str
    layer: GateLayer
    description: str

    # Gates whose report output THIS gate reads as input.
    depends_on: FrozenSet[str] = frozenset()

    # CLI argument names that accept file paths from the request's normalized fields.
    # Maps: request_field_name -> CLI_flag
    request_inputs: Dict[str, str] = field(default_factory=dict)

    # CLI argument names that accept other gate report paths.
    # Maps: dependency_gate_name -> CLI_flag
    report_inputs: Dict[str, str] = field(default_factory=dict)

    # Output report file basename (inside evidence_dir).
    report_output: str = ""

    # Whether this gate can be skipped in non-publication mode.
    publication_only: bool = False

    # Whether this gate is parallelizable with peers in the same layer.
    parallelizable: bool = True

    # Human-readable category for summary grouping.
    category: str = "general"


# ---------------------------------------------------------------------------
# The full gate registry
# ---------------------------------------------------------------------------

GATE_REGISTRY: Dict[str, GateSpec] = {}


def _register(spec: GateSpec) -> GateSpec:
    GATE_REGISTRY[spec.name] = spec
    return spec


# -- Layer 0: Contract validation --

_register(GateSpec(
    name="request_contract_gate",
    script="request_contract_gate.py",
    layer=GateLayer.CONTRACT,
    description="Validate the structured request contract and normalize all input paths.",
    request_inputs={"request": "--request"},
    report_output="request_contract_report.json",
    category="contract",
))

# -- Layer 1: Manifest lock --

_register(GateSpec(
    name="manifest_lock",
    script="manifest_lock.py",
    layer=GateLayer.MANIFEST,
    description="Compute and verify SHA-256 manifest of all input artifacts.",
    depends_on=frozenset({"request_contract_gate"}),
    report_output="manifest.json",
    category="integrity",
))

# -- Layer 2: Execution attestation --

_register(GateSpec(
    name="execution_attestation_gate",
    script="execution_attestation_gate.py",
    layer=GateLayer.ATTESTATION,
    description="Verify cryptographic execution attestation: signatures, timestamps, transparency log.",
    depends_on=frozenset({"manifest_lock"}),
    request_inputs={
        "execution_attestation_spec": "--attestation-spec",
        "evaluation_report_file": "--evaluation-report",
    },
    report_output="execution_attestation_report.json",
    category="integrity",
))

# -- Layer 3: Data validation (parallelizable) --

_register(GateSpec(
    name="leakage_gate",
    script="leakage_gate.py",
    layer=GateLayer.DATA_VALIDATION,
    description="Detect patient ID leakage, temporal leakage, and row-hash overlap across splits.",
    depends_on=frozenset({"request_contract_gate"}),
    report_output="leakage_report.json",
    category="data_integrity",
))

_register(GateSpec(
    name="split_protocol_gate",
    script="split_protocol_gate.py",
    layer=GateLayer.DATA_VALIDATION,
    description="Verify split protocol compliance: stratification, temporal ordering, size requirements.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={"split_protocol_spec": "--protocol-spec"},
    report_output="split_protocol_report.json",
    category="data_integrity",
))

_register(GateSpec(
    name="covariate_shift_gate",
    script="covariate_shift_gate.py",
    layer=GateLayer.DATA_VALIDATION,
    description="Detect covariate distribution shift between training and evaluation splits.",
    depends_on=frozenset({"request_contract_gate"}),
    report_output="covariate_shift_report.json",
    category="data_integrity",
))

_register(GateSpec(
    name="reporting_bias_gate",
    script="reporting_bias_gate.py",
    layer=GateLayer.DATA_VALIDATION,
    description="Enforce TRIPOD+AI / PROBAST+AI / STARD-AI reporting and bias checklists.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={"reporting_bias_checklist_spec": "--checklist-spec"},
    report_output="reporting_bias_report.json",
    category="compliance",
))

# -- Layer 4: Policy & lineage audits (parallelizable) --

_register(GateSpec(
    name="definition_variable_guard",
    script="definition_variable_guard.py",
    layer=GateLayer.POLICY_AUDIT,
    description="Verify phenotype definition variables against training data columns.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={"phenotype_definition_spec": "--definition-spec"},
    report_output="definition_guard_report.json",
    category="data_integrity",
))

_register(GateSpec(
    name="feature_lineage_gate",
    script="feature_lineage_gate.py",
    layer=GateLayer.POLICY_AUDIT,
    description="Verify feature lineage spec against phenotype definition and training data.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "phenotype_definition_spec": "--definition-spec",
        "feature_lineage_spec": "--lineage-spec",
    },
    report_output="lineage_report.json",
    category="data_integrity",
))

_register(GateSpec(
    name="imbalance_policy_gate",
    script="imbalance_policy_gate.py",
    layer=GateLayer.POLICY_AUDIT,
    description="Verify class imbalance handling policy against actual split prevalence.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "imbalance_policy_spec": "--policy-spec",
        "evaluation_report_file": "--evaluation-report",
    },
    report_output="imbalance_policy_report.json",
    category="policy",
))

_register(GateSpec(
    name="missingness_policy_gate",
    script="missingness_policy_gate.py",
    layer=GateLayer.POLICY_AUDIT,
    description="Verify missingness handling policy against actual missing data patterns.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "missingness_policy_spec": "--policy-spec",
        "evaluation_report_file": "--evaluation-report",
    },
    report_output="missingness_policy_report.json",
    category="policy",
))

_register(GateSpec(
    name="tuning_leakage_gate",
    script="tuning_leakage_gate.py",
    layer=GateLayer.POLICY_AUDIT,
    description="Verify hyperparameter tuning protocol does not leak test data.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={"tuning_protocol_spec": "--tuning-spec"},
    report_output="tuning_leakage_report.json",
    category="data_integrity",
))

# -- Layer 5: Model & feature audits (parallelizable) --

_register(GateSpec(
    name="model_selection_audit_gate",
    script="model_selection_audit_gate.py",
    layer=GateLayer.MODEL_AUDIT,
    description="Audit model selection process for protocol compliance and data leakage.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "model_selection_report_file": "--model-selection-report",
        "tuning_protocol_spec": "--tuning-spec",
    },
    report_output="model_selection_audit_report.json",
    category="model",
))

_register(GateSpec(
    name="feature_engineering_audit_gate",
    script="feature_engineering_audit_gate.py",
    layer=GateLayer.MODEL_AUDIT,
    description="Audit feature engineering for reproducibility and selection leakage.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "feature_group_spec": "--feature-group-spec",
        "feature_engineering_report_file": "--feature-engineering-report",
        "feature_lineage_spec": "--lineage-spec",
        "tuning_protocol_spec": "--tuning-spec",
    },
    report_output="feature_engineering_audit_report.json",
    category="model",
))

_register(GateSpec(
    name="clinical_metrics_gate",
    script="clinical_metrics_gate.py",
    layer=GateLayer.MODEL_AUDIT,
    description="Verify clinical metric floors and operating point requirements.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "evaluation_report_file": "--evaluation-report",
        "external_validation_report_file": "--external-validation-report",
        "performance_policy_spec": "--performance-policy",
    },
    report_output="clinical_metrics_report.json",
    category="performance",
))

# -- Layer 6: Metric validation (mostly parallelizable) --

_register(GateSpec(
    name="prediction_replay_gate",
    script="prediction_replay_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Replay predictions from trace and verify metric consistency with evaluation report.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "evaluation_report_file": "--evaluation-report",
        "prediction_trace_file": "--prediction-trace",
        "performance_policy_spec": "--performance-policy",
    },
    report_output="prediction_replay_report.json",
    category="performance",
))

_register(GateSpec(
    name="distribution_generalization_gate",
    script="distribution_generalization_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Assess distribution shift and generalization across splits and external cohorts.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "evaluation_report_file": "--evaluation-report",
        "external_validation_report_file": "--external-validation-report",
        "feature_group_spec": "--feature-group-spec",
        "performance_policy_spec": "--performance-policy",
        "distribution_report_file": "--distribution-report",
    },
    report_output="distribution_generalization_report.json",
    category="generalization",
))

_register(GateSpec(
    name="generalization_gap_gate",
    script="generalization_gap_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Compute and threshold directional performance gaps between splits.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "evaluation_report_file": "--evaluation-report",
        "performance_policy_spec": "--performance-policy",
    },
    report_output="generalization_gap_report.json",
    category="generalization",
))

_register(GateSpec(
    name="robustness_gate",
    script="robustness_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Verify model robustness across time slices and patient subgroups.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "robustness_report_file": "--robustness-report",
        "performance_policy_spec": "--performance-policy",
    },
    report_output="robustness_gate_report.json",
    category="generalization",
))

_register(GateSpec(
    name="seed_stability_gate",
    script="seed_stability_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Verify model stability across random seed variations.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "seed_sensitivity_report_file": "--seed-sensitivity-report",
        "performance_policy_spec": "--performance-policy",
    },
    report_output="seed_stability_report.json",
    category="generalization",
))

_register(GateSpec(
    name="external_validation_gate",
    script="external_validation_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Validate transferability on external cohorts with metric replay.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "external_validation_report_file": "--external-validation-report",
        "prediction_trace_file": "--prediction-trace",
        "evaluation_report_file": "--evaluation-report",
        "performance_policy_spec": "--performance-policy",
    },
    report_output="external_validation_gate_report.json",
    category="generalization",
))

_register(GateSpec(
    name="calibration_dca_gate",
    script="calibration_dca_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Validate model calibration and decision curve analysis thresholds.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "prediction_trace_file": "--prediction-trace",
        "evaluation_report_file": "--evaluation-report",
        "external_validation_report_file": "--external-validation-report",
        "performance_policy_spec": "--performance-policy",
    },
    report_output="calibration_dca_report.json",
    category="performance",
))

_register(GateSpec(
    name="ci_matrix_gate",
    script="ci_matrix_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Validate confidence interval matrix via bootstrap resampling.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "evaluation_report_file": "--evaluation-report",
        "prediction_trace_file": "--prediction-trace",
        "external_validation_report_file": "--external-validation-report",
        "performance_policy_spec": "--performance-policy",
        "ci_matrix_report_file": "--ci-matrix-report",
    },
    report_output="ci_matrix_gate_report.json",
    category="performance",
))

_register(GateSpec(
    name="metric_consistency_gate",
    script="metric_consistency_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Verify primary metric value matches between request and evaluation report.",
    depends_on=frozenset({"request_contract_gate"}),
    request_inputs={
        "evaluation_report_file": "--evaluation-report",
    },
    report_output="metric_consistency_report.json",
    category="performance",
))

_register(GateSpec(
    name="evaluation_quality_gate",
    script="evaluation_quality_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Assess overall evaluation quality: CI width, baseline delta, resampling adequacy.",
    depends_on=frozenset({"request_contract_gate", "metric_consistency_gate", "ci_matrix_gate"}),
    request_inputs={
        "evaluation_report_file": "--evaluation-report",
        "ci_matrix_report_file": "--ci-matrix-report",
    },
    report_output="evaluation_quality_report.json",
    category="performance",
))

_register(GateSpec(
    name="permutation_significance_gate",
    script="permutation_significance_gate.py",
    layer=GateLayer.METRIC_VALIDATION,
    description="Verify model performance is statistically significant vs. permutation null distribution.",
    depends_on=frozenset({"request_contract_gate", "metric_consistency_gate"}),
    request_inputs={
        "permutation_null_metrics_file": "--null-metrics-file",
    },
    report_output="permutation_report.json",
    category="performance",
))

# -- Layer 7: Aggregation --

_register(GateSpec(
    name="publication_gate",
    script="publication_gate.py",
    layer=GateLayer.AGGREGATION,
    description="Aggregate all gate results and determine publication eligibility.",
    depends_on=frozenset(
        name for name, spec in GATE_REGISTRY.items()
        if spec.layer < GateLayer.AGGREGATION
    ),
    report_output="publication_gate_report.json",
    parallelizable=False,
    category="aggregation",
))

# -- Layer 8: Final self-critique --

_register(GateSpec(
    name="self_critique_gate",
    script="self_critique_gate.py",
    layer=GateLayer.FINAL,
    description="Self-critique: compute quality score and generate actionable recommendations.",
    depends_on=frozenset(
        name for name in GATE_REGISTRY
    ),
    report_output="self_critique_report.json",
    parallelizable=False,
    category="aggregation",
))


# ---------------------------------------------------------------------------
# DAG resolution utilities
# ---------------------------------------------------------------------------

def get_execution_layers() -> List[Tuple[int, List[str]]]:
    """Return gate names grouped by execution layer, sorted by layer order.

    Each element is (layer_value, [gate_names]) so callers can map back to
    the correct GateLayer enum value even when some layers are empty.
    Within each layer, gates are listed alphabetically for determinism.
    Gates in the same layer can execute in parallel.
    """
    layer_map: Dict[int, List[str]] = {}
    for name, spec in GATE_REGISTRY.items():
        layer_map.setdefault(spec.layer.value, []).append(name)

    layers: List[Tuple[int, List[str]]] = []
    for layer_idx in sorted(layer_map.keys()):
        layers.append((layer_idx, sorted(layer_map[layer_idx])))
    return layers


def topological_sort() -> List[str]:
    """Return all gate names in a valid topological execution order."""
    visited: Set[str] = set()
    order: List[str] = []

    def _visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        spec = GATE_REGISTRY.get(name)
        if spec is None:
            return
        for dep in sorted(spec.depends_on):
            _visit(dep)
        order.append(name)

    for name in sorted(GATE_REGISTRY.keys()):
        _visit(name)

    return order


def get_dependencies(gate_name: str, transitive: bool = False) -> FrozenSet[str]:
    """Return direct (or transitive) dependencies of a gate."""
    spec = GATE_REGISTRY.get(gate_name)
    if spec is None:
        return frozenset()

    if not transitive:
        return spec.depends_on

    visited: Set[str] = set()

    def _collect(name: str) -> None:
        s = GATE_REGISTRY.get(name)
        if s is None:
            return
        for dep in s.depends_on:
            if dep not in visited:
                visited.add(dep)
                _collect(dep)

    _collect(gate_name)
    return frozenset(visited)


def get_dependents(gate_name: str) -> FrozenSet[str]:
    """Return gates that directly depend on the given gate."""
    return frozenset(
        name for name, spec in GATE_REGISTRY.items()
        if gate_name in spec.depends_on
    )


def get_runnable_subset(
    target_gates: Sequence[str],
    include_dependencies: bool = True,
) -> List[str]:
    """Compute the minimal set of gates to run for given targets.

    If include_dependencies is True, all transitive dependencies are included.
    Returns gates in topological order.
    """
    needed: Set[str] = set(target_gates)

    if include_dependencies:
        for gate in list(target_gates):
            needed |= get_dependencies(gate, transitive=True)

    full_order = topological_sort()
    return [g for g in full_order if g in needed]


def validate_dag() -> List[str]:
    """Validate the DAG for cycles and missing references.

    Returns a list of error messages (empty if valid).
    """
    errors: List[str] = []
    for name, spec in GATE_REGISTRY.items():
        for dep in spec.depends_on:
            if dep not in GATE_REGISTRY:
                errors.append(f"{name}: dependency '{dep}' not in registry")

    # Cycle detection via Kahn's algorithm
    adj: Dict[str, List[str]] = {name: [] for name in GATE_REGISTRY}
    for name, spec in GATE_REGISTRY.items():
        for dep in spec.depends_on:
            if dep in adj:
                adj[dep].append(name)

    in_deg: Dict[str, int] = {name: len(spec.depends_on & frozenset(GATE_REGISTRY.keys()))
                               for name, spec in GATE_REGISTRY.items()}
    queue = [n for n, d in in_deg.items() if d == 0]
    order: List[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in adj.get(node, []):
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    if len(order) != len(GATE_REGISTRY):
        missing = set(GATE_REGISTRY.keys()) - set(order)
        errors.append(f"Cycle detected involving: {sorted(missing)}")

    return errors


def get_gate_spec(name: str) -> Optional[GateSpec]:
    """Look up a gate spec by name."""
    return GATE_REGISTRY.get(name)


def list_gates_by_category() -> Dict[str, List[str]]:
    """Group gate names by their category."""
    cats: Dict[str, List[str]] = {}
    for name, spec in GATE_REGISTRY.items():
        cats.setdefault(spec.category, []).append(name)
    return {k: sorted(v) for k, v in sorted(cats.items())}


def print_dag_summary() -> None:
    """Print a human-readable DAG summary to stdout."""
    layers = get_execution_layers()
    print(f"\nGate DAG: {len(GATE_REGISTRY)} gates in {len(layers)} layers\n")
    for layer_val, layer_gates in layers:
        layer_enum = GateLayer(layer_val)
        print(f"  Layer {layer_val} ({layer_enum.name}):")
        for gate_name in layer_gates:
            spec = GATE_REGISTRY[gate_name]
            deps = ", ".join(sorted(spec.depends_on)) if spec.depends_on else "(none)"
            par = "||" if spec.parallelizable else ">>"
            print(f"    {par} {gate_name:40s} <- {deps}")
        print()
