# Contributing Guide

## Development Environment

### Prerequisites

- Python 3.10 or later
- `openssl` available in PATH (required for execution attestation)
- Git

### Setup

```bash
git clone https://github.com/Furinaaa-Cancan/medical-ml-leakage-guard.git
cd medical-ml-leakage-guard
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Optional model backends
python3 -m pip install -r requirements-optional.txt

# Verify
python3 scripts/mlgg.py doctor
```

---

## Code Style

### Python

- **Target version**: Python 3.10+.
- **Type annotations**: All function signatures must include type annotations.
- **Docstrings**: Google style. Include `Args`, `Returns`, and `Raises` sections
  where applicable. Types may be omitted from docstrings when present in the
  function signature.
- **Imports**: Standard library first, then third-party, then local. Always at
  the top of the file.
- **Line length**: No strict enforced limit, but prefer ≤120 characters.
- **Naming**: `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for
  module-level constants.

### Gate Scripts

Every gate script must follow this contract:

- Accept `--report <path>` to specify the output JSON report path.
- Accept `--strict` to enable strict-mode validation (warnings become failures).
- Use exit code `0` for pass, `2` for fail.
- Output a JSON report with at least a `status` field (`"pass"` or `"fail"`).
- Use `_gate_utils.py` shared utilities (`add_issue`, `load_json`, `write_json`,
  `to_float`) for consistency.
- `to_float` must reject `inf` and `nan` with a `math.isfinite` guard.
- `finish()` must use `bool(failures) or (args.strict and bool(warnings))` for
  the fail decision.

### Commit Messages

Follow the conventional format:

```
<type>: <description>

[optional body]
```

Types:
- `feat`: New feature or gate
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code restructure without behavior change
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

Examples:
```
feat: add calibration_dca_gate with Brier score check
fix: generalization_gap_gate finish() ignores --strict for warnings
docs: add complete docstrings to train_select_evaluate.py (#68)
test: add split_data.py smoke tests for all 3 strategies
```

---

## Testing

### Existing Test Suites

| Test | Purpose | Command |
|------|---------|---------|
| `test_gate_smoke.py` | Gate script smoke tests (leakage, publication, interactive) | `python3 scripts/test_gate_smoke.py` |
| `test_onboarding_smoke.py` | Onboarding flow smoke tests | `python3 scripts/test_onboarding_smoke.py` |
| `test_split_smoke.py` | Data splitting smoke tests (single-CSV workflow) | `python3 scripts/test_split_smoke.py` |

### Testing Requirements

- **New gate scripts** must have corresponding smoke tests added to
  `test_gate_smoke.py` covering at least one pass case and one fail case.
- **New features** must include tests that verify the expected behavior.
- **Bug fixes** should include a regression test when feasible.
- Tests use `subprocess` to run scripts as independent processes, matching
  the real CLI execution model.

### Running Tests

```bash
# Pytest unit tests (2800+ tests, ~8 min)
python3 -m pytest tests/ -q --tb=short

# Pytest with coverage report
python3 -m pytest tests/ -q --tb=no --cov=scripts --cov-report=term-missing

# Smoke tests (quick integration checks)
python3 scripts/test_gate_smoke.py
python3 scripts/test_onboarding_smoke.py
python3 scripts/test_split_smoke.py

# Full authority benchmark (longer)
python3 scripts/mlgg.py authority

# Release-grade benchmark suite
python3 scripts/mlgg.py benchmark-suite --profile release
```

---

## Pull Request Process

1. **Branch**: Create a feature branch from `main`.
2. **Implement**: Make your changes following the code style above.
3. **Test**: Run all smoke tests and ensure they pass.
4. **Compile check**: Verify Python files compile without errors:
   ```bash
   python3 -c "import py_compile; py_compile.compile('scripts/<your_file>.py', doraise=True)"
   ```
5. **Commit**: Use conventional commit messages.
6. **PR description**: Describe what changed, why, and how it was tested.
7. **Review**: All PRs require review before merging.

---

## Architecture Overview

See `references/Architecture.md` for the 29-gate pipeline flowchart,
gate reference table, and data flow diagram.

---

## Key Constraints

- **Fail-closed**: The pipeline is fail-closed by design. Any change that
  could silently pass a gate that should fail is a critical bug.
- **Deterministic**: Same inputs must produce the same verdict. Avoid
  non-deterministic behavior in gate scripts.
- **Medical safety**: Never weaken the medical non-negotiable rules listed
  in `SKILL.md`. These are not configurable.
- **No test-data leakage**: Never introduce code paths where test data
  influences model training, selection, calibration, or threshold choice.
- **Backward compatibility**: Gate report JSON schemas should be extended
  additively. Do not remove or rename existing fields without a major
  version bump.
