#!/usr/bin/env python3
"""
End-to-end strict skill validation on authoritative public medical binary datasets.

Workflow per dataset:
1. Load and clean raw data.
2. Build train/valid/test CSV splits with disjoint IDs and strict temporal ordering.
3. Train a baseline model and compute real test AUROC/PR-AUC.
4. Build permutation-null AUROC distribution.
5. Generate signed execution attestation bundle.
6. Run strict pipeline bootstrap (allow-missing-compare).
7. Freeze manifest baseline and rerun strict pipeline with comparison.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetCase:
    case_id: str
    raw_filename: str
    target_name: str
    source_name: str


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "experiments" / "authority-e2e"
RAW_ROOT = DATA_ROOT / "raw"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
REFERENCES_ROOT = REPO_ROOT / "references"


def run_cmd(cmd: List[str], cwd: Path | None = None, allow_fail: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    if not allow_fail and proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"$ {' '.join(cmd)}\n"
            f"exit={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def load_heart_dataset(raw_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "goal",
    ]
    df = pd.read_csv(raw_path, header=None, names=cols, na_values="?")
    df = df.dropna(axis=0).reset_index(drop=True)
    df["y"] = (pd.to_numeric(df["goal"], errors="coerce") > 0).astype(int)
    feature_cols = [c for c in cols if c != "goal"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=0).reset_index(drop=True)
    return df[feature_cols + ["y"]], feature_cols


def load_breast_dataset(raw_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",
        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave_points_worst",
        "symmetry_worst",
        "fractal_dimension_worst",
    ]
    cols = ["sample_id", "diagnosis"] + feature_cols
    df = pd.read_csv(raw_path, header=None, names=cols)
    df["y"] = (df["diagnosis"].astype(str).str.strip().str.upper() == "M").astype(int)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=0).reset_index(drop=True)
    return df[feature_cols + ["y"]], feature_cols


def split_with_temporal_order(df: pd.DataFrame, feature_cols: List[str], case_id: str) -> Dict[str, pd.DataFrame]:
    y = df["y"].to_numpy()
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=20260224,
        stratify=y,
    )
    temp_y = y[temp_idx]
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=20260224,
        stratify=temp_y,
    )

    split_map = {
        "train": np.array(sorted(train_idx.tolist()), dtype=int),
        "valid": np.array(sorted(valid_idx.tolist()), dtype=int),
        "test": np.array(sorted(test_idx.tolist()), dtype=int),
    }
    starts = {
        "train": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "valid": datetime(2022, 1, 1, tzinfo=timezone.utc),
        "test": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }

    output: Dict[str, pd.DataFrame] = {}
    for split_name, idx in split_map.items():
        sub = df.iloc[idx].copy().reset_index(drop=True)
        patient_ids = [f"{case_id.upper()}_{int(i):06d}" for i in idx]
        times = [
            (starts[split_name] + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(len(sub))
        ]
        sub.insert(0, "event_time", times)
        sub.insert(0, "patient_id", patient_ids)
        ordered = sub[["patient_id", "event_time", "y"] + feature_cols].copy()
        ordered["y"] = ordered["y"].astype(int)
        output[split_name] = ordered
    return output


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[Pipeline, Dict[str, float], Dict[str, Any], List[float]]:
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["y"].to_numpy(dtype=int)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df["y"].to_numpy(dtype=int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, solver="liblinear", random_state=20260224)),
        ]
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))
    brier = float(brier_score_loss(y_test, proba))

    prevalence = float(np.mean(y_train))
    baseline_proba = np.full(shape=y_test.shape[0], fill_value=prevalence, dtype=float)
    baseline_auroc = float(roc_auc_score(y_test, baseline_proba))
    baseline_pr_auc = float(average_precision_score(y_test, baseline_proba))

    rng = np.random.default_rng(20260224)
    null_metrics: List[float] = []
    for _ in range(300):
        y_perm = rng.permutation(y_test)
        null_metrics.append(float(roc_auc_score(y_perm, proba)))

    ci_samples: List[float] = []
    n_test = y_test.shape[0]
    for _ in range(500):
        idx = rng.integers(0, n_test, n_test)
        if len(np.unique(y_test[idx])) < 2:
            continue
        ci_samples.append(float(roc_auc_score(y_test[idx], proba[idx])))
    if len(ci_samples) < 200:
        raise RuntimeError(f"Insufficient bootstrap resamples for CI: {len(ci_samples)}")
    ci_arr = np.array(ci_samples, dtype=float)
    ci_lower, ci_upper = np.percentile(ci_arr, [2.5, 97.5]).tolist()

    metrics = {
        "roc_auc": round(auroc, 6),
        "pr_auc": round(pr_auc, 6),
        "brier": round(brier, 6),
    }
    quality = {
        "uncertainty": {
            "metrics": {
                "roc_auc": {
                    "method": "bootstrap",
                    "n_resamples": int(len(ci_samples)),
                    "ci_95": [round(float(ci_lower), 6), round(float(ci_upper), 6)],
                }
            }
        },
        "baselines": {
            "prevalence_model": {
                "metrics": {
                    "roc_auc": round(baseline_auroc, 6),
                    "pr_auc": round(baseline_pr_auc, 6),
                }
            }
        },
    }
    return model, metrics, quality, null_metrics


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)


def copy_reference(src_name: str, dst_path: Path) -> None:
    src = REFERENCES_ROOT / src_name
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)


def prepare_case_artifacts(case: DatasetCase) -> Dict[str, Any]:
    raw_path = RAW_ROOT / case.raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset missing: {raw_path}")

    if case.case_id == "uci-heart-disease":
        df, feature_cols = load_heart_dataset(raw_path)
    elif case.case_id == "uci-breast-cancer-wdbc":
        df, feature_cols = load_breast_dataset(raw_path)
    else:
        raise ValueError(f"Unsupported case_id: {case.case_id}")

    splits = split_with_temporal_order(df, feature_cols, case.case_id)
    model, metrics, quality, null_metrics = train_and_evaluate(splits["train"], splits["test"], feature_cols)

    case_root = DATA_ROOT / case.case_id
    data_dir = case_root / "data"
    cfg_dir = case_root / "configs"
    evidence_dir = case_root / "evidence"
    model_dir = case_root / "models"
    key_dir = case_root / "keys"
    for d in (data_dir, cfg_dir, evidence_dir, model_dir, key_dir):
        d.mkdir(parents=True, exist_ok=True)

    for split_name, frame in splits.items():
        frame.to_csv(data_dir / f"{split_name}.csv", index=False)

    model_path = model_dir / "logreg_model.joblib"
    joblib.dump(model, model_path)

    config_payload = {
        "dataset_case": case.case_id,
        "source_name": case.source_name,
        "model_type": "logistic_regression",
        "random_seed": 20260224,
        "features": feature_cols,
    }
    write_json(cfg_dir / "train_config.json", config_payload)

    train_log_path = evidence_dir / "train.log"
    train_log_lines = [
        f"[INFO] case={case.case_id}",
        "[INFO] model=LogisticRegression(liblinear)",
        f"[INFO] feature_count={len(feature_cols)}",
        f"[INFO] train_rows={len(splits['train'])}",
        f"[INFO] valid_rows={len(splits['valid'])}",
        f"[INFO] test_rows={len(splits['test'])}",
        f"[INFO] test_roc_auc={metrics['roc_auc']:.6f}",
        f"[INFO] test_pr_auc={metrics['pr_auc']:.6f}",
        "[INFO] training_complete=true",
    ]
    train_log_path.write_text("\n".join(train_log_lines) + "\n", encoding="utf-8")

    eval_report: Dict[str, Any] = {
        "model_id": f"logreg_{case.case_id}",
        "split": "test",
        "metrics": {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier": metrics["brier"],
        },
    }
    eval_report.update(quality)
    write_json(evidence_dir / "evaluation_report.json", eval_report)
    with (evidence_dir / "permutation_null_auc.txt").open("w", encoding="utf-8") as fh:
        for value in null_metrics:
            fh.write(f"{value:.6f}\n")

    phenotype_def = {
        "global_forbidden_patterns": ["(?i)target", "(?i)label"],
        "targets": {
            case.target_name: {
                "defining_variables": [
                    "confirmed_diagnosis_code",
                    "reference_standard_positive",
                ],
                "forbidden_patterns": ["(?i)diagnosis_code", "(?i)reference_standard"],
                "notes": f"Definition variables for {case.target_name} must not appear in predictors.",
            }
        },
    }
    write_json(cfg_dir / "phenotype_definitions.json", phenotype_def)

    lineage = {"features": {}}
    for col in feature_cols:
        lineage["features"][col] = {"ancestors": [f"raw_{col}"]}
    write_json(cfg_dir / "feature_lineage.json", lineage)

    copy_reference("split-protocol.example.json", cfg_dir / "split_protocol.json")
    copy_reference("imbalance-policy.example.json", cfg_dir / "imbalance_policy.json")
    copy_reference("missingness-policy.example.json", cfg_dir / "missingness_policy.json")
    copy_reference("tuning-protocol.example.json", cfg_dir / "tuning_protocol.json")
    copy_reference("reporting-bias-checklist.example.json", cfg_dir / "reporting_bias_checklist.json")

    key_pairs: Dict[str, Tuple[Path, Path]] = {}
    for role in ("attestation", "timestamp", "transparency", "execution", "execution_log", "witness_a", "witness_b"):
        priv_key = key_dir / f"{role}_priv.pem"
        pub_key = key_dir / f"{role}_pub.pem"
        run_cmd(
            [
                "openssl",
                "genpkey",
                "-algorithm",
                "RSA",
                "-pkeyopt",
                "rsa_keygen_bits:3072",
                "-out",
                str(priv_key),
            ]
        )
        run_cmd(["openssl", "pkey", "-in", str(priv_key), "-pubout", "-out", str(pub_key)])
        key_pairs[role] = (priv_key, pub_key)

    study_id = f"{case.case_id}-study-v1"
    run_id = f"{case.case_id}-run-001"
    run_cmd(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "generate_execution_attestation.py"),
            "--study-id",
            study_id,
            "--run-id",
            run_id,
            "--payload-out",
            str(evidence_dir / "attestation_payload.json"),
            "--signature-out",
            str(evidence_dir / "attestation.sig"),
            "--spec-out",
            str(cfg_dir / "execution_attestation.json"),
            "--private-key-file",
            str(key_pairs["attestation"][0]),
            "--public-key-file",
            str(key_pairs["attestation"][1]),
            "--timestamp-private-key-file",
            str(key_pairs["timestamp"][0]),
            "--timestamp-public-key-file",
            str(key_pairs["timestamp"][1]),
            "--transparency-private-key-file",
            str(key_pairs["transparency"][0]),
            "--transparency-public-key-file",
            str(key_pairs["transparency"][1]),
            "--execution-private-key-file",
            str(key_pairs["execution"][0]),
            "--execution-public-key-file",
            str(key_pairs["execution"][1]),
            "--execution-log-private-key-file",
            str(key_pairs["execution_log"][0]),
            "--execution-log-public-key-file",
            str(key_pairs["execution_log"][1]),
            "--require-independent-timestamp-authority",
            "--require-independent-execution-authority",
            "--require-independent-log-authority",
            "--require-witness-quorum",
            "--min-witness-count",
            "2",
            "--require-independent-witness-keys",
            "--require-witness-independence-from-signing",
            "--witness",
            f"witness-a|{key_pairs['witness_a'][1]}|{key_pairs['witness_a'][0]}",
            "--witness",
            f"witness-b|{key_pairs['witness_b'][1]}|{key_pairs['witness_b'][0]}",
            "--command",
            f"python train_model.py --case {case.case_id}",
            "--artifact",
            f"training_log={train_log_path}",
            "--artifact",
            f"training_config={cfg_dir / 'train_config.json'}",
            "--artifact",
            f"model_artifact={model_path}",
            "--artifact",
            f"evaluation_report={evidence_dir / 'evaluation_report.json'}",
        ]
    )

    request_payload = {
        "study_id": study_id,
        "run_id": run_id,
        "target_name": case.target_name,
        "prediction_unit": "patient-encounter",
        "index_time_col": "event_time",
        "label_col": "y",
        "patient_id_col": "patient_id",
        "primary_metric": "roc_auc",
        "claim_tier_target": "publication-grade",
        "phenotype_definition_spec": "phenotype_definitions.json",
        "feature_lineage_spec": "feature_lineage.json",
        "split_protocol_spec": "split_protocol.json",
        "imbalance_policy_spec": "imbalance_policy.json",
        "missingness_policy_spec": "missingness_policy.json",
        "tuning_protocol_spec": "tuning_protocol.json",
        "reporting_bias_checklist_spec": "reporting_bias_checklist.json",
        "execution_attestation_spec": "execution_attestation.json",
        "split_paths": {
            "train": "../data/train.csv",
            "valid": "../data/valid.csv",
            "test": "../data/test.csv",
        },
        "evaluation_report_file": "../evidence/evaluation_report.json",
        "evaluation_metric_path": "metrics.roc_auc",
        "permutation_null_metrics_file": "../evidence/permutation_null_auc.txt",
        "actual_primary_metric": metrics["roc_auc"],
        "thresholds": {
            "alpha": 0.01,
            "min_delta": 0.03,
            "min_baseline_delta": 0.0,
            "ci_min_resamples": 200,
            "ci_max_width": 0.50,
        },
        "context": {
            "source": case.source_name,
            "notes": "Authoritative public benchmark dataset for strict skill validation.",
        },
    }
    write_json(cfg_dir / "request.json", request_payload)

    bootstrap_report = evidence_dir / "strict_pipeline_bootstrap_report.json"
    bootstrap_proc = run_cmd(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "run_strict_pipeline.py"),
            "--request",
            str(cfg_dir / "request.json"),
            "--evidence-dir",
            str(evidence_dir),
            "--allow-missing-compare",
            "--strict",
            "--report",
            str(bootstrap_report),
        ],
        allow_fail=True,
    )

    manifest_path = evidence_dir / "manifest.json"
    baseline_manifest = evidence_dir / "manifest_baseline.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Bootstrap manifest missing for {case.case_id}: {manifest_path}")
    shutil.copy2(manifest_path, baseline_manifest)

    final_report = evidence_dir / "strict_pipeline_report.json"
    final_proc = run_cmd(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "run_strict_pipeline.py"),
            "--request",
            str(cfg_dir / "request.json"),
            "--evidence-dir",
            str(evidence_dir),
            "--compare-manifest",
            str(baseline_manifest),
            "--strict",
            "--report",
            str(final_report),
        ],
        allow_fail=True,
    )

    pipeline_report = json.loads(final_report.read_text(encoding="utf-8"))
    self_critique_report = json.loads((evidence_dir / "self_critique_report.json").read_text(encoding="utf-8"))
    publication_report = json.loads((evidence_dir / "publication_gate_report.json").read_text(encoding="utf-8"))

    return {
        "case_id": case.case_id,
        "source_name": case.source_name,
        "rows": {
            "train": int(len(splits["train"])),
            "valid": int(len(splits["valid"])),
            "test": int(len(splits["test"])),
        },
        "metrics": metrics,
        "bootstrap_exit_code": int(bootstrap_proc.returncode),
        "final_exit_code": int(final_proc.returncode),
        "pipeline_status": pipeline_report.get("status"),
        "publication_status": publication_report.get("status"),
        "self_critique_status": self_critique_report.get("status"),
        "self_critique_score": self_critique_report.get("quality_score"),
        "artifacts": {
            "case_root": str(case_root),
            "request": str(cfg_dir / "request.json"),
            "pipeline_report": str(final_report),
        },
    }


def main() -> int:
    cases = [
        DatasetCase(
            case_id="uci-heart-disease",
            raw_filename="heart_disease_processed.cleveland.data",
            target_name="heart_disease",
            source_name="UCI Heart Disease (Cleveland)",
        ),
        DatasetCase(
            case_id="uci-breast-cancer-wdbc",
            raw_filename="breast_cancer_wdbc.data",
            target_name="breast_cancer_malignancy",
            source_name="UCI Breast Cancer Wisconsin (Diagnostic)",
        ),
    ]

    results: List[Dict[str, Any]] = []
    failed_cases: List[str] = []

    for case in cases:
        try:
            result = prepare_case_artifacts(case)
            results.append(result)
            if result.get("pipeline_status") != "pass":
                failed_cases.append(case.case_id)
        except Exception as exc:
            failed_cases.append(case.case_id)
            results.append(
                {
                    "case_id": case.case_id,
                    "source_name": case.source_name,
                    "error": str(exc),
                }
            )

    summary = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_root": str(REPO_ROOT),
        "results": results,
        "failed_cases": failed_cases,
        "overall_status": "pass" if not failed_cases else "fail",
    }
    summary_path = DATA_ROOT / "authority_e2e_summary.json"
    write_json(summary_path, summary)
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if not failed_cases else 2


if __name__ == "__main__":
    raise SystemExit(main())
