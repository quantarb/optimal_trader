from __future__ import annotations

from pathlib import Path


IGNORED_DIR_NAMES = {
    ".git",
    ".idea",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
}

IGNORED_PATH_PARTS = {
    "migrations",
}


def discover_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIR_NAMES for part in relative.parts):
            continue
        if any(part in IGNORED_PATH_PARTS for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def module_name_for_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parent_module(module_name: str) -> str:
    if "." not in module_name:
        return ""
    return module_name.rsplit(".", 1)[0]


def resolve_import_module(current_module: str, target_module: str | None, level: int) -> str:
    if level <= 0:
        return str(target_module or "").strip()

    current_parts = current_module.split(".")
    if current_parts:
        current_parts = current_parts[:-1]
    if level > 1:
        current_parts = current_parts[: max(len(current_parts) - (level - 1), 0)]
    base = ".".join(current_parts)
    if target_module:
        return ".".join(part for part in [base, str(target_module).strip()] if part)
    return base
