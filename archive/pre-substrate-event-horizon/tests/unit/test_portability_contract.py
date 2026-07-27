import tomllib

from mop.config import REPO_ROOT


def test_freeze_has_no_clone_specific_editable_path():
    freeze = (REPO_ROOT / "scaffolding/requirements.freeze.txt").read_text()
    assert "file:///" not in freeze
    assert "/Users/" not in freeze
    assert "Downloads/brain" not in freeze


def test_active_python_sources_have_no_developer_absolute_path():
    offenders: list[str] = []
    for top in ("src", "scripts"):
        for path in (REPO_ROOT / top).rglob("*.py"):
            text = path.read_text()
            if "/Users/" in text or "Downloads/brain" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_dependency_profiles_declare_readiness_backends():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    dependencies = set(project["dependencies"])
    optional = project["optional-dependencies"]
    assert any(dep.startswith("psutil") for dep in dependencies)
    assert any(dep.startswith("torchvision") for dep in optional["video"])
    assert any(dep.startswith("av") for dep in optional["video"])
    assert any(dep.startswith("transformers") for dep in optional["encoder"])
    assert any(dep.startswith("mlx") for dep in optional["apple"])


def test_install_target_proves_isolated_import():
    makefile = (REPO_ROOT / "Makefile").read_text()
    assert "verify-install:" in makefile
    assert "cd /tmp" in makefile
    assert " -I -c " in makefile
    assert "import importlib.metadata, mop" in makefile
