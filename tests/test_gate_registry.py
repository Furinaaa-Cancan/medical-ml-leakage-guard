"""Unit tests for scripts/_gate_registry.py DAG structure and utilities."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Set


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _gate_registry import (
    GATE_REGISTRY,
    GateLayer,
    get_dependencies,
    get_dependents,
    get_execution_layers,
    get_gate_spec,
    get_runnable_subset,
    list_gates_by_category,
    topological_sort,
    validate_dag,
)


# ────────────────────────────────────────────────────────
# Registry integrity
# ────────────────────────────────────────────────────────

class TestRegistryIntegrity:
    def test_registry_not_empty(self):
        assert len(GATE_REGISTRY) > 0

    def test_all_gates_have_script(self):
        for name, spec in GATE_REGISTRY.items():
            assert spec.script.endswith(".py"), f"{name} script doesn't end with .py"

    def test_all_gates_have_report_output(self):
        for name, spec in GATE_REGISTRY.items():
            assert spec.report_output, f"{name} missing report_output"

    def test_all_gate_names_match_keys(self):
        for name, spec in GATE_REGISTRY.items():
            assert spec.name == name, f"Registry key '{name}' != spec.name '{spec.name}'"

    def test_known_gates_present(self):
        expected = {
            "request_contract_gate", "manifest_lock", "execution_attestation_gate",
            "leakage_gate", "publication_gate", "self_critique_gate",
        }
        for gate in expected:
            assert gate in GATE_REGISTRY, f"Expected gate '{gate}' not in registry"

    def test_expected_gate_count(self):
        assert len(GATE_REGISTRY) == 28


# ────────────────────────────────────────────────────────
# DAG validation
# ────────────────────────────────────────────────────────

class TestDAGValidation:
    def test_validate_dag_no_errors(self):
        errors = validate_dag()
        assert errors == [], f"DAG validation errors: {errors}"

    def test_no_self_dependencies(self):
        for name, spec in GATE_REGISTRY.items():
            assert name not in spec.depends_on, f"{name} depends on itself"

    def test_all_dependencies_exist(self):
        for name, spec in GATE_REGISTRY.items():
            for dep in spec.depends_on:
                assert dep in GATE_REGISTRY, f"{name} depends on unknown gate '{dep}'"

    def test_request_contract_has_no_dependencies(self):
        spec = GATE_REGISTRY["request_contract_gate"]
        assert len(spec.depends_on) == 0

    def test_publication_gate_depends_on_all_prior_layers(self):
        pub = GATE_REGISTRY["publication_gate"]
        for name, spec in GATE_REGISTRY.items():
            if spec.layer < GateLayer.AGGREGATION:
                assert name in pub.depends_on, (
                    f"publication_gate missing dependency on {name}"
                )

    def test_self_critique_depends_on_all_gates(self):
        sc = GATE_REGISTRY["self_critique_gate"]
        for name in GATE_REGISTRY:
            if name != "self_critique_gate":
                assert name in sc.depends_on, (
                    f"self_critique_gate missing dependency on {name}"
                )


# ────────────────────────────────────────────────────────
# Topological sort
# ────────────────────────────────────────────────────────

class TestTopologicalSort:
    def test_returns_all_gates(self):
        order = topological_sort()
        assert set(order) == set(GATE_REGISTRY.keys())

    def test_dependencies_before_dependents(self):
        order = topological_sort()
        idx = {name: i for i, name in enumerate(order)}
        for name, spec in GATE_REGISTRY.items():
            for dep in spec.depends_on:
                assert idx[dep] < idx[name], (
                    f"{dep} should come before {name} in topological order"
                )

    def test_request_contract_is_first(self):
        order = topological_sort()
        assert order[0] == "request_contract_gate"

    def test_self_critique_is_last(self):
        order = topological_sort()
        assert order[-1] == "self_critique_gate"


# ────────────────────────────────────────────────────────
# Execution layers
# ────────────────────────────────────────────────────────

class TestExecutionLayers:
    def test_returns_tuples(self):
        layers = get_execution_layers()
        for item in layers:
            assert isinstance(item, tuple)
            assert len(item) == 2
            layer_val, gate_names = item
            assert isinstance(layer_val, int)
            assert isinstance(gate_names, list)

    def test_all_gates_covered(self):
        layers = get_execution_layers()
        all_names: Set[str] = set()
        for _, gate_names in layers:
            all_names.update(gate_names)
        assert all_names == set(GATE_REGISTRY.keys())

    def test_layers_sorted_ascending(self):
        layers = get_execution_layers()
        layer_vals = [lv for lv, _ in layers]
        assert layer_vals == sorted(layer_vals)

    def test_layer_values_match_gate_layer(self):
        layers = get_execution_layers()
        for layer_val, gate_names in layers:
            for name in gate_names:
                spec = GATE_REGISTRY[name]
                assert spec.layer.value == layer_val, (
                    f"{name} in layer {layer_val} but spec says {spec.layer.value}"
                )

    def test_gates_within_layer_sorted_alphabetically(self):
        layers = get_execution_layers()
        for _, gate_names in layers:
            assert gate_names == sorted(gate_names)


# ────────────────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────────────────

class TestDependencies:
    def test_direct_dependencies(self):
        deps = get_dependencies("leakage_gate")
        assert "request_contract_gate" in deps

    def test_transitive_dependencies(self):
        deps = get_dependencies("self_critique_gate", transitive=True)
        assert "request_contract_gate" in deps
        assert "publication_gate" in deps
        assert "leakage_gate" in deps

    def test_unknown_gate_returns_empty(self):
        deps = get_dependencies("nonexistent_gate_xyz")
        assert deps == frozenset()

    def test_get_dependents(self):
        dependents = get_dependents("request_contract_gate")
        assert "leakage_gate" in dependents
        assert "manifest_lock" in dependents

    def test_get_dependents_leaf_gate(self):
        dependents = get_dependents("self_critique_gate")
        assert len(dependents) == 0


# ────────────────────────────────────────────────────────
# Runnable subset
# ────────────────────────────────────────────────────────

class TestRunnableSubset:
    def test_single_gate_with_deps(self):
        subset = get_runnable_subset(["leakage_gate"], include_dependencies=True)
        assert "request_contract_gate" in subset
        assert "leakage_gate" in subset

    def test_single_gate_without_deps(self):
        subset = get_runnable_subset(["leakage_gate"], include_dependencies=False)
        assert subset == ["leakage_gate"]

    def test_topological_order_preserved(self):
        subset = get_runnable_subset(
            ["publication_gate"], include_dependencies=True
        )
        full_order = topological_sort()
        sub_order = [g for g in full_order if g in subset]
        assert subset == sub_order


# ────────────────────────────────────────────────────────
# Misc utilities
# ────────────────────────────────────────────────────────

class TestMiscUtilities:
    def test_get_gate_spec_found(self):
        spec = get_gate_spec("leakage_gate")
        assert spec is not None
        assert spec.name == "leakage_gate"

    def test_get_gate_spec_not_found(self):
        assert get_gate_spec("nonexistent_xyz") is None

    def test_list_gates_by_category(self):
        cats = list_gates_by_category()
        assert isinstance(cats, dict)
        assert len(cats) > 0
        for cat, names in cats.items():
            assert names == sorted(names)
            for name in names:
                assert GATE_REGISTRY[name].category == cat
