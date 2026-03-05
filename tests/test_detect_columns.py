"""Comprehensive unit tests for _hint_match and detect_columns in mlgg_pixel.py."""
from __future__ import annotations

import sys
from pathlib import Path


# Ensure scripts/ is importable and skip terminal-dependent init
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# Set test mode before import to skip clear/logo
import mlgg_pixel
mlgg_pixel._TEST_MODE = True

from mlgg_pixel import (
    _hint_match,
    detect_columns,
)


# ────────────────────────────────────────────────────────
# _hint_match — PID hints
# ────────────────────────────────────────────────────────

class TestHintMatchPID:
    # --- patient_id (contains underscore → substring match) ---
    def test_patient_id_exact(self):
        assert _hint_match("patient_id", "patient_id") is True

    def test_patient_id_prefix(self):
        assert _hint_match("patient_id", "some_patient_id") is True

    def test_patient_id_no_match(self):
        assert _hint_match("patient_id", "patientid") is False  # no underscore separation

    # --- patientid (no underscore → word-level match) ---
    def test_patientid_exact(self):
        assert _hint_match("patientid", "patientid") is True

    def test_patientid_in_compound(self):
        assert _hint_match("patientid", "my_patientid_col") is True

    def test_patientid_substring_no_match(self):
        # "patientid" as word should match in "some_patientid"
        assert _hint_match("patientid", "some_patientid") is True

    # --- patient (no underscore → word-level match) ---
    def test_patient_exact(self):
        assert _hint_match("patient", "patient") is True

    def test_patient_as_word(self):
        assert _hint_match("patient", "my_patient_data") is True

    def test_patient_not_in_patientid(self):
        # "patient" should NOT match "patientid" because split("_") gives ["patientid"]
        assert _hint_match("patient", "patientid") is False

    # --- subject_id (contains underscore → substring match) ---
    def test_subject_id_exact(self):
        assert _hint_match("subject_id", "subject_id") is True

    def test_subject_id_in_longer(self):
        assert _hint_match("subject_id", "my_subject_id_col") is True

    def test_subject_id_no_underscore(self):
        assert _hint_match("subject_id", "subjectid") is False

    # --- subjectid (no underscore → word-level) ---
    def test_subjectid_exact(self):
        assert _hint_match("subjectid", "subjectid") is True

    def test_subjectid_in_compound(self):
        assert _hint_match("subjectid", "my_subjectid") is True

    # --- sample_id (contains underscore → substring) ---
    def test_sample_id_exact(self):
        assert _hint_match("sample_id", "sample_id") is True

    def test_sample_id_not_sampleid(self):
        assert _hint_match("sample_id", "sampleid") is False

    # --- pid (no underscore → word-level) ---
    def test_pid_exact(self):
        assert _hint_match("pid", "pid") is True

    def test_pid_as_word(self):
        assert _hint_match("pid", "some_pid_col") is True

    def test_pid_not_in_rapid(self):
        # "pid" should NOT match "rapid" because split("_") gives ["rapid"]
        assert _hint_match("pid", "rapid") is False

    def test_pid_not_in_stupid(self):
        assert _hint_match("pid", "stupid") is False

    # --- mrn (no underscore → word-level) ---
    def test_mrn_exact(self):
        assert _hint_match("mrn", "mrn") is True

    def test_mrn_as_word(self):
        assert _hint_match("mrn", "patient_mrn") is True

    def test_mrn_not_substring(self):
        assert _hint_match("mrn", "mrnumber") is False

    # --- record_id (contains underscore → substring) ---
    def test_record_id_exact(self):
        assert _hint_match("record_id", "record_id") is True

    # --- case_id (contains underscore → substring) ---
    def test_case_id_exact(self):
        assert _hint_match("case_id", "case_id") is True

    def test_case_id_not_caseid(self):
        assert _hint_match("case_id", "caseid") is False


# ────────────────────────────────────────────────────────
# _hint_match — TGT hints
# ────────────────────────────────────────────────────────

class TestHintMatchTGT:
    def test_target_exact(self):
        assert _hint_match("target", "target") is True

    def test_target_as_word(self):
        assert _hint_match("target", "my_target") is True

    def test_target_not_in_targetname(self):
        assert _hint_match("target", "targetname") is False

    def test_label_exact(self):
        assert _hint_match("label", "label") is True

    def test_label_as_word(self):
        assert _hint_match("label", "class_label") is True

    def test_label_not_in_relabel(self):
        assert _hint_match("label", "relabel") is False

    def test_y_exact(self):
        assert _hint_match("y", "y") is True

    def test_y_as_word(self):
        assert _hint_match("y", "some_y_col") is True

    def test_y_not_in_year(self):
        # "y" should NOT match "year" (split gives ["year"])
        assert _hint_match("y", "year") is False

    def test_y_not_in_therapy(self):
        assert _hint_match("y", "therapy") is False

    def test_y_not_in_systolic(self):
        assert _hint_match("y", "systolic") is False

    def test_outcome_exact(self):
        assert _hint_match("outcome", "outcome") is True

    def test_outcome_as_word(self):
        assert _hint_match("outcome", "primary_outcome") is True

    def test_diagnosis_exact(self):
        assert _hint_match("diagnosis", "diagnosis") is True

    def test_class_exact(self):
        assert _hint_match("class", "class") is True

    def test_class_not_in_classification(self):
        assert _hint_match("class", "classification") is False

    def test_result_exact(self):
        assert _hint_match("result", "result") is True

    def test_status_exact(self):
        assert _hint_match("status", "status") is True

    def test_status_as_word(self):
        assert _hint_match("status", "patient_status") is True

    def test_status_not_in_statuscode(self):
        assert _hint_match("status", "statuscode") is False

    def test_disease_exact(self):
        assert _hint_match("disease", "disease") is True

    def test_mortality_exact(self):
        assert _hint_match("mortality", "mortality") is True


# ────────────────────────────────────────────────────────
# _hint_match — TIME hints
# ────────────────────────────────────────────────────────

class TestHintMatchTIME:
    def test_time_exact(self):
        assert _hint_match("time", "time") is True

    def test_time_as_word(self):
        assert _hint_match("time", "event_time") is True

    def test_time_not_in_timestamp_word(self):
        # "time" as word should NOT match "timestamp" (single word)
        assert _hint_match("time", "timestamp") is False

    def test_date_exact(self):
        assert _hint_match("date", "date") is True

    def test_date_as_word(self):
        assert _hint_match("date", "admission_date") is True

    def test_date_not_in_update(self):
        assert _hint_match("date", "update") is False

    def test_timestamp_exact(self):
        assert _hint_match("timestamp", "timestamp") is True

    def test_timestamp_as_word(self):
        assert _hint_match("timestamp", "event_timestamp") is True

    def test_event_time_exact(self):
        assert _hint_match("event_time", "event_time") is True

    def test_event_time_substring(self):
        assert _hint_match("event_time", "my_event_time_col") is True

    def test_event_time_no_underscore(self):
        assert _hint_match("event_time", "eventtime") is False

    def test_datetime_exact(self):
        assert _hint_match("datetime", "datetime") is True

    def test_admission_exact(self):
        assert _hint_match("admission", "admission") is True

    def test_admission_as_word(self):
        assert _hint_match("admission", "hospital_admission") is True

    def test_visit_date_exact(self):
        assert _hint_match("visit_date", "visit_date") is True

    def test_visit_date_not_visitdate(self):
        assert _hint_match("visit_date", "visitdate") is False

    def test_created_at_exact(self):
        assert _hint_match("created_at", "created_at") is True

    def test_created_at_not_createdat(self):
        assert _hint_match("created_at", "createdat") is False


# ────────────────────────────────────────────────────────
# _hint_match — Edge cases
# ────────────────────────────────────────────────────────

class TestHintMatchEdge:
    def test_empty_hint(self):
        # Empty hint with no underscore → word match; "" not in ["anything"]
        assert _hint_match("", "anything") is False

    def test_empty_col(self):
        # Empty col → split("_") gives [""], "y" not in [""]
        assert _hint_match("y", "") is False

    def test_both_empty(self):
        assert _hint_match("", "") is True

    def test_only_underscores_col(self):
        # "_" → split("_") gives ["", ""]
        assert _hint_match("y", "_") is False

    def test_underscore_in_hint_empty_parts(self):
        # "__" hint has underscore → substring match
        assert _hint_match("__", "a__b") is True

    def test_case_sensitivity(self):
        # _hint_match itself doesn't lowercase; detect_columns does
        assert _hint_match("pid", "PID") is False  # case sensitive
        assert _hint_match("pid", "pid") is True


# ────────────────────────────────────────────────────────
# detect_columns — Typical medical datasets
# ────────────────────────────────────────────────────────

class TestDetectColumns:
    def test_typical_medical(self):
        cols = ["patient_id", "age", "gender", "target", "event_time", "feature1"]
        result = detect_columns(cols)
        assert result["pid"] == "patient_id"
        assert result["target"] == "target"
        assert result["time"] == "event_time"

    def test_alternative_names(self):
        cols = ["subject_id", "diagnosis", "admission", "bp_systolic"]
        result = detect_columns(cols)
        assert result["pid"] == "subject_id"
        assert result["target"] == "diagnosis"
        assert result["time"] == "admission"

    def test_y_column(self):
        cols = ["mrn", "y", "age", "date"]
        result = detect_columns(cols)
        assert result["pid"] == "mrn"
        assert result["target"] == "y"
        assert result["time"] == "date"

    def test_no_match(self):
        cols = ["col_a", "col_b", "col_c"]
        result = detect_columns(cols)
        assert result["pid"] is None
        assert result["target"] is None
        assert result["time"] is None

    def test_only_pid_match(self):
        cols = ["patient_id", "feature1", "feature2"]
        result = detect_columns(cols)
        assert result["pid"] == "patient_id"
        assert result["target"] is None
        assert result["time"] is None

    def test_space_replaced_with_underscore(self):
        cols = ["Patient ID", "Target Value", "Event Time"]
        result = detect_columns(cols)
        # "Patient ID" → "patient_id" → matches patient_id hint
        assert result["pid"] == "Patient ID"
        # "Target Value" → "target_value" → "target" not in ["target_value"] as word
        # but "target" hint has no underscore → word match → "target" in ["target", "value"] ✓
        assert result["target"] == "Target Value"

    def test_uppercase(self):
        cols = ["PATIENT_ID", "Y", "EVENT_TIME"]
        result = detect_columns(cols)
        assert result["pid"] == "PATIENT_ID"
        assert result["target"] == "Y"
        assert result["time"] == "EVENT_TIME"

    def test_mixed_case(self):
        cols = ["PatientId", "Label", "DateTime"]
        result = detect_columns(cols)
        # "patientid" → patientid hint matches
        assert result["pid"] == "PatientId"
        assert result["target"] == "Label"
        assert result["time"] == "DateTime"

    def test_first_match_wins(self):
        """detect_columns should use first matching column for each category."""
        cols = ["pid", "mrn", "target", "label", "time", "date"]
        result = detect_columns(cols)
        assert result["pid"] == "pid"  # first PID hint match
        assert result["target"] == "target"  # first TGT hint match
        assert result["time"] == "time"  # first TIME hint match

    def test_cross_category_no_conflict(self):
        """Each column should only be assigned to one category."""
        # "status" matches TGT, not PID or TIME
        cols = ["status", "patient_id", "timestamp"]
        result = detect_columns(cols)
        assert result["target"] == "status"
        assert result["pid"] == "patient_id"
        assert result["time"] == "timestamp"

    def test_y_not_match_year(self):
        """'y' hint should NOT match 'year' column."""
        cols = ["patient_id", "year", "outcome"]
        result = detect_columns(cols)
        assert result["target"] == "outcome"  # "year" should not be target

    def test_pid_not_match_rapid(self):
        """'pid' hint should NOT match 'rapid' column."""
        cols = ["rapid", "patient_id", "y"]
        result = detect_columns(cols)
        assert result["pid"] == "patient_id"  # "rapid" should not be pid

    def test_empty_columns(self):
        result = detect_columns([])
        assert result["pid"] is None
        assert result["target"] is None
        assert result["time"] is None

    def test_single_column_pid(self):
        result = detect_columns(["patient_id"])
        assert result["pid"] == "patient_id"
        assert result["target"] is None
        assert result["time"] is None

    def test_whitespace_column_names(self):
        cols = ["  patient_id  ", " y ", " event_time "]
        result = detect_columns(cols)
        assert result["pid"] == "  patient_id  "
        assert result["target"] == " y "
        assert result["time"] == " event_time "

    def test_100_plus_columns(self):
        """Performance test with many columns."""
        cols = [f"feature_{i}" for i in range(100)]
        cols.extend(["patient_id", "y", "event_time"])
        result = detect_columns(cols)
        assert result["pid"] == "patient_id"
        assert result["target"] == "y"
        assert result["time"] == "event_time"

    def test_status_cross_category(self):
        """'status' in TGT hints should match as target, not as time or pid."""
        cols = ["sample_id", "patient_status", "visit_date"]
        result = detect_columns(cols)
        # "patient_status" → "patient_status" → "status" in ["patient", "status"] ✓
        assert result["target"] == "patient_status"
        assert result["pid"] == "sample_id"
        assert result["time"] == "visit_date"

    def test_class_label_column(self):
        cols = ["record_id", "class_label", "created_at"]
        result = detect_columns(cols)
        assert result["pid"] == "record_id"
        # "class_label" → "class" in ["class", "label"] ✓ (matches "class" first in TGT)
        assert result["target"] == "class_label"
        assert result["time"] == "created_at"
