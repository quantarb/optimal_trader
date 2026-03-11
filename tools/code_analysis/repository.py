from __future__ import annotations

import ast
import io
import keyword
import tokenize
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .discovery import discover_python_files, module_name_for_path, resolve_import_module


@dataclass
class FunctionRecord:
    module: str
    path: str
    qualname: str
    name: str
    lineno: int
    end_lineno: int
    class_name: str = ""
    is_method: bool = False
    resolved_calls: list[str] = field(default_factory=list)
    unresolved_calls: list[str] = field(default_factory=list)
    source: str = ""
    normalized_tokens: list[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.module}.{self.qualname}" if self.module else self.qualname

    @property
    def line_count(self) -> int:
        return max(0, int(self.end_lineno) - int(self.lineno) + 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["full_name"] = self.full_name
        payload["line_count"] = self.line_count
        return payload


@dataclass
class ClassRecord:
    module: str
    path: str
    qualname: str
    name: str
    lineno: int
    end_lineno: int
    methods: list[str] = field(default_factory=list)
    source: str = ""
    normalized_tokens: list[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.module}.{self.qualname}" if self.module else self.qualname

    @property
    def line_count(self) -> int:
        return max(0, int(self.end_lineno) - int(self.lineno) + 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["full_name"] = self.full_name
        payload["line_count"] = self.line_count
        return payload


@dataclass
class ModuleRecord:
    module: str
    path: str
    imports: list[str]
    import_module_aliases: dict[str, str]
    import_name_aliases: dict[str, str]
    classes: list[str]
    class_records: list[ClassRecord]
    functions: list[FunctionRecord]
    line_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "path": self.path,
            "imports": list(self.imports),
            "import_module_aliases": dict(self.import_module_aliases),
            "import_name_aliases": dict(self.import_name_aliases),
            "classes": list(self.classes),
            "class_records": [item.to_dict() for item in self.class_records],
            "functions": [item.to_dict() for item in self.functions],
            "line_count": self.line_count,
        }


@dataclass
class RepositoryInventory:
    root: str
    modules: dict[str, ModuleRecord]

    @property
    def functions(self) -> dict[str, FunctionRecord]:
        rows: dict[str, FunctionRecord] = {}
        for module_record in self.modules.values():
            for function_record in module_record.functions:
                rows[function_record.full_name] = function_record
        return rows

    @property
    def classes(self) -> dict[str, ClassRecord]:
        rows: dict[str, ClassRecord] = {}
        for module_record in self.modules.values():
            for class_record in module_record.class_records:
                rows[class_record.full_name] = class_record
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "module_count": len(self.modules),
            "function_count": len(self.functions),
            "class_count": len(self.classes),
            "modules": {name: record.to_dict() for name, record in self.modules.items()},
        }


def _normalize_tokens(source: str) -> list[str]:
    tokens: list[str] = []
    try:
        generator = tokenize.generate_tokens(io.StringIO(source).readline)
    except Exception:
        return tokens
    for token_type, token_text, *_ in generator:
        if token_type in {
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.ENCODING,
            tokenize.ENDMARKER,
        }:
            continue
        if token_type == tokenize.NAME:
            if keyword.iskeyword(token_text):
                tokens.append(token_text)
            else:
                tokens.append(token_text.lower())
            continue
        if token_type == tokenize.NUMBER:
            tokens.append("NUM")
            continue
        if token_type == tokenize.STRING:
            tokens.append("STR")
            continue
        tokens.append(token_text)
    return tokens


class _CallCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        module: str,
        class_name: str,
        local_functions: dict[str, str],
        class_methods: dict[str, str],
        import_module_aliases: dict[str, str],
        import_name_aliases: dict[str, str],
    ) -> None:
        self.module = module
        self.class_name = class_name
        self.local_functions = local_functions
        self.class_methods = class_methods
        self.import_module_aliases = import_module_aliases
        self.import_name_aliases = import_name_aliases
        self.resolved_calls: list[str] = []
        self.unresolved_calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> Any:
        resolved = self._resolve_callee(node.func)
        if resolved:
            self.resolved_calls.append(resolved)
        else:
            label = self._expr_label(node.func)
            if label:
                self.unresolved_calls.append(label)
        self.generic_visit(node)

    def _resolve_callee(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            if node.id in self.local_functions:
                return self.local_functions[node.id]
            if node.id in self.class_methods and self.class_name:
                return self.class_methods[node.id]
            if node.id in self.import_name_aliases:
                return self.import_name_aliases[node.id]
            return ""
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                base_name = node.value.id
                if base_name in self.import_module_aliases:
                    return f"{self.import_module_aliases[base_name]}.{node.attr}"
                if base_name in self.import_name_aliases:
                    return f"{self.import_name_aliases[base_name]}.{node.attr}"
                if base_name in {"self", "cls"} and self.class_name:
                    return self.class_methods.get(node.attr, f"{self.module}.{self.class_name}.{node.attr}")
                if base_name == self.class_name and self.class_name:
                    return self.class_methods.get(node.attr, f"{self.module}.{self.class_name}.{node.attr}")
                if base_name == self.module.split(".")[-1]:
                    return f"{self.module}.{node.attr}"
            parent = self._resolve_callee(node.value)
            if parent:
                return f"{parent}.{node.attr}"
        return ""

    def _expr_label(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._expr_label(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Call):
            return self._expr_label(node.func)
        return ""


def _extract_imports(tree: ast.Module, current_module: str, internal_modules: set[str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    imports: list[str] = []
    module_aliases: dict[str, str] = {}
    name_aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = str(alias.name or "").strip()
                if not target:
                    continue
                internal_target = _match_internal_module(target, internal_modules)
                if internal_target:
                    imports.append(internal_target)
                    module_aliases[alias.asname or target.split(".")[-1]] = internal_target
        elif isinstance(node, ast.ImportFrom):
            resolved_module = resolve_import_module(current_module, node.module, node.level)
            internal_target = _match_internal_module(resolved_module, internal_modules)
            if internal_target:
                imports.append(internal_target)
            for alias in node.names:
                if alias.name == "*":
                    continue
                exported = ".".join(part for part in [resolved_module, alias.name] if part)
                if _match_internal_module(resolved_module, internal_modules):
                    name_aliases[alias.asname or alias.name] = exported
    return sorted(set(imports)), module_aliases, name_aliases


def _match_internal_module(target: str, internal_modules: set[str]) -> str:
    target = str(target or "").strip()
    if not target:
        return ""
    if target in internal_modules:
        return target
    parts = target.split(".")
    while len(parts) > 1:
        parts = parts[:-1]
        candidate = ".".join(parts)
        if candidate in internal_modules:
            return candidate
    return ""


def _collect_functions(
    *,
    module: str,
    path: Path,
    tree: ast.Module,
    source_lines: list[str],
    import_module_aliases: dict[str, str],
    import_name_aliases: dict[str, str],
) -> tuple[list[str], list[ClassRecord], list[FunctionRecord]]:
    classes: list[str] = []
    class_records: list[ClassRecord] = []
    top_level_functions: dict[str, str] = {}
    class_methods_by_name: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_functions[node.name] = f"{module}.{node.name}"
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            class_methods_by_name[node.name] = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_methods_by_name[node.name][child.name] = f"{module}.{node.name}.{child.name}"

    functions: list[FunctionRecord] = []

    def build_record(node: ast.FunctionDef | ast.AsyncFunctionDef, *, class_name: str = "") -> FunctionRecord:
        lineno = int(getattr(node, "lineno", 1))
        end_lineno = int(getattr(node, "end_lineno", lineno))
        source = "".join(source_lines[lineno - 1 : end_lineno])
        collector = _CallCollector(
            module=module,
            class_name=class_name,
            local_functions=top_level_functions,
            class_methods=class_methods_by_name.get(class_name, {}),
            import_module_aliases=import_module_aliases,
            import_name_aliases=import_name_aliases,
        )
        collector.visit(node)
        qualname = f"{class_name}.{node.name}" if class_name else node.name
        return FunctionRecord(
            module=module,
            path=str(path),
            qualname=qualname,
            name=node.name,
            lineno=lineno,
            end_lineno=end_lineno,
            class_name=class_name,
            is_method=bool(class_name),
            resolved_calls=sorted(set(filter(None, collector.resolved_calls))),
            unresolved_calls=sorted(set(filter(None, collector.unresolved_calls))),
            source=source,
            normalized_tokens=_normalize_tokens(source),
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(build_record(node))
        elif isinstance(node, ast.ClassDef):
            class_lineno = int(getattr(node, "lineno", 1))
            class_end_lineno = int(getattr(node, "end_lineno", class_lineno))
            class_source = "".join(source_lines[class_lineno - 1 : class_end_lineno])
            class_records.append(
                ClassRecord(
                    module=module,
                    path=str(path),
                    qualname=node.name,
                    name=node.name,
                    lineno=class_lineno,
                    end_lineno=class_end_lineno,
                    methods=sorted(class_methods_by_name.get(node.name, {}).keys()),
                    source=class_source,
                    normalized_tokens=_normalize_tokens(class_source),
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(build_record(child, class_name=node.name))
    return classes, class_records, functions


def build_repository_inventory(root: Path) -> RepositoryInventory:
    python_files = discover_python_files(root)
    module_names = {module_name_for_path(root, path) for path in python_files if module_name_for_path(root, path)}
    modules: dict[str, ModuleRecord] = {}
    for path in python_files:
        module = module_name_for_path(root, path)
        if not module:
            continue
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines(keepends=True)
        tree = ast.parse(source, filename=str(path))
        imports, module_aliases, name_aliases = _extract_imports(tree, module, module_names)
        classes, class_records, functions = _collect_functions(
            module=module,
            path=path,
            tree=tree,
            source_lines=source_lines,
            import_module_aliases=module_aliases,
            import_name_aliases=name_aliases,
        )
        modules[module] = ModuleRecord(
            module=module,
            path=str(path),
            imports=imports,
            import_module_aliases=module_aliases,
            import_name_aliases=name_aliases,
            classes=classes,
            class_records=class_records,
            functions=functions,
            line_count=len(source_lines),
        )
    return RepositoryInventory(root=str(root), modules=modules)
