from __future__ import annotations

from pathlib import Path


EXCLUDED_PARTS = {
    ".git",
    ".idea",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "docs",
    "data",
}


def discover_repo_root(start: str | Path | None = None) -> Path:
    candidate = Path(start).resolve() if start is not None else Path(__file__).resolve().parents[3]
    if (candidate / "manage.py").exists():
        return candidate
    for parent in [candidate, *candidate.parents]:
        if (parent / "manage.py").exists():
            return parent
    return candidate


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_relative_path(path: str | Path, root: str | Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(root).resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def module_name_for_path(root: str | Path, path: str | Path) -> str:
    rel = Path(safe_relative_path(path, root))
    without_suffix = rel.with_suffix("")
    parts = list(without_suffix.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def path_for_module_name(root: str | Path, module_name: str) -> Path:
    root_path = Path(root).resolve()
    file_path = root_path.joinpath(*str(module_name).split(".")).with_suffix(".py")
    if file_path.exists():
        return file_path
    return root_path.joinpath(*str(module_name).split("."), "__init__.py")


def iter_python_files(root: str | Path, *, include_tests: bool = True) -> list[Path]:
    root_path = Path(root).resolve()
    files: list[Path] = []
    for path in root_path.rglob("*.py"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if "migrations" in path.parts:
            continue
        if not include_tests and any(part == "tests" for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        files.append(path)
    return sorted(files)
