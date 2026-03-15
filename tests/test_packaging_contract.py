from __future__ import annotations

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_project_metadata() -> dict:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]


def test_core_dependencies_include_scipy() -> None:
    project = _load_project_metadata()
    deps = set(project["dependencies"])
    assert "scipy>=1.9" in deps


def test_public_console_scripts_only_expose_supported_entrypoints() -> None:
    project = _load_project_metadata()
    scripts = project["scripts"]
    assert scripts == {
        "mlgg": "scripts.mlgg:cli_main",
        "mlgg-pixel": "scripts.mlgg_pixel:main",
    }


def test_web_prototype_is_explicit_optional_extra() -> None:
    project = _load_project_metadata()
    optional = project["optional-dependencies"]
    assert optional["web"] == ["flask>=3.0"]
    assert "flask>=3.0" in optional["all"]
