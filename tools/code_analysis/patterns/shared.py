from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..repository import (
    ClassRecord,
    FunctionRecord,
    ModuleRecord,
    RepositoryInventory,
    build_repository_inventory,
)


CONTROL_FLOW_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
COMPLEXITY_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.ExceptHandler,
    ast.IfExp,
    ast.BoolOp,
    ast.comprehension,
    ast.Match,
)
PURE_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "frozenset",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
QUERY_LIKE_PREFIXES = ("get_", "list_", "find_", "compute_", "calculate_", "score_", "build_", "derive_", "normalize_")
COMMON_MAGIC_NUMBERS = {
    -1,
    0,
    1,
    2,
    3,
    5,
    7,
    10,
    12,
    24,
    30,
    60,
    100,
    365,
    1000,
}
EFFECTFUL_CALL_MARKERS = (
    "commit",
    "delete",
    "execute",
    "log",
    "mkdir",
    "post",
    "print",
    "publish",
    "put",
    "remove",
    "rename",
    "rmdir",
    "save",
    "send",
    "touch",
    "unlink",
    "upload",
    "write",
)
PURE_SERIALIZATION_CALLS = {
    "ast.unparse",
    "dataclasses.asdict",
    "dataclasses.astuple",
    "json.dump",
    "json.dumps",
    "yaml.dump",
    "yaml.safe_dump",
}
LOCAL_MUTATION_METHODS = {
    "append",
    "clear",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
    "update",
}
EXPENSIVE_CALL_MARKERS = (
    "backtest",
    "collect",
    "download",
    "execute",
    "fetch",
    "fit",
    "predict",
    "query",
    "request",
    "scan",
    "select",
    "train",
    ".objects.filter",
    ".objects.get",
    ".objects.create",
    ".objects.bulk",
    ".read_sql",
    ".read_parquet",
    ".read_csv",
    ".read_text",
    ".load_artifact",
    "requests.",
    "httpx.",
    "urllib.",
)
CHEAP_CALL_MARKERS = (
    ".append",
    ".copy",
    ".get",
    ".items",
    ".keys",
    ".lower",
    ".setdefault",
    ".sort",
    ".strip",
    ".update",
    ".values",
    "dict.get",
    "json.loads",
    "len",
    "max",
    "min",
    "range",
    "sum",
)
TOKEN_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])|[^A-Za-z0-9]+")


@dataclass
class FunctionContext:
    module: str
    path: str
    class_name: str
    qualname: str
    full_name: str
    lineno: int
    end_lineno: int
    node: ast.FunctionDef | ast.AsyncFunctionDef
    source: str
    resolved_calls: list[str] = field(default_factory=list)
    unresolved_calls: list[str] = field(default_factory=list)

    @property
    def loc(self) -> int:
        return max(0, self.end_lineno - self.lineno + 1)

    @property
    def is_method(self) -> bool:
        return bool(self.class_name)

    @property
    def is_public(self) -> bool:
        return self.name not in {"__init__", "__call__"} and not self.name.startswith("_")

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def decorator_names(self) -> list[str]:
        return decorator_names(self.node)

    @property
    def positional_parameters(self) -> list[ast.arg]:
        args = list(self.node.args.posonlyargs) + list(self.node.args.args)
        if self.is_method and args and args[0].arg in {"self", "cls"}:
            return args[1:]
        return args

    @property
    def parameter_count(self) -> int:
        count = len(self.positional_parameters) + len(self.node.args.kwonlyargs)
        if self.node.args.vararg:
            count += 1
        if self.node.args.kwarg:
            count += 1
        return count

    @property
    def typed_parameter_count(self) -> int:
        count = sum(1 for arg in self.positional_parameters if arg.annotation is not None)
        count += sum(1 for arg in self.node.args.kwonlyargs if arg.annotation is not None)
        count += 1 if self.node.args.vararg and self.node.args.vararg.annotation is not None else 0
        count += 1 if self.node.args.kwarg and self.node.args.kwarg.annotation is not None else 0
        return count

    @property
    def has_return_annotation(self) -> bool:
        return self.node.returns is not None


@dataclass
class ClassContext:
    module: str
    path: str
    qualname: str
    full_name: str
    lineno: int
    end_lineno: int
    node: ast.ClassDef
    source: str
    methods: list[FunctionContext] = field(default_factory=list)

    @property
    def loc(self) -> int:
        return max(0, self.end_lineno - self.lineno + 1)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def decorator_names(self) -> list[str]:
        return decorator_names(self.node)

    @property
    def base_names(self) -> list[str]:
        return [expression_name(base) for base in self.node.bases if expression_name(base)]

    @property
    def public_method_count(self) -> int:
        return sum(1 for method in self.methods if method.is_public)


@dataclass
class ModuleContext:
    module: str
    path: str
    tree: ast.Module
    source: str
    source_lines: list[str]
    record: ModuleRecord
    functions: list[FunctionContext] = field(default_factory=list)
    classes: list[ClassContext] = field(default_factory=list)
    top_level_constants: set[str] = field(default_factory=set)

    @property
    def line_count(self) -> int:
        return len(self.source_lines)

    @property
    def is_test_module(self) -> bool:
        return is_test_module_name(self.module)


@dataclass
class RepositoryAstContext:
    root: str
    inventory: RepositoryInventory
    modules: dict[str, ModuleContext]

    @property
    def functions(self) -> dict[str, FunctionContext]:
        rows: dict[str, FunctionContext] = {}
        for module_context in self.modules.values():
            for function in module_context.functions:
                rows[function.full_name] = function
        return rows

    @property
    def classes(self) -> dict[str, ClassContext]:
        rows: dict[str, ClassContext] = {}
        for module_context in self.modules.values():
            for class_context in module_context.classes:
                rows[class_context.full_name] = class_context
        return rows


def build_repository_ast_context(root: Path, inventory: RepositoryInventory | None = None) -> RepositoryAstContext:
    inventory = inventory or build_repository_inventory(root)
    modules: dict[str, ModuleContext] = {}
    for module_name, module_record in inventory.modules.items():
        path = Path(module_record.path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        attach_parents(tree)
        source_lines = source.splitlines(keepends=True)
        function_records = {record.full_name: record for record in module_record.functions}
        class_records = {record.full_name: record for record in module_record.class_records}

        class_map: dict[str, ClassContext] = {}
        functions: list[FunctionContext] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_full_name = f"{module_name}.{node.name}"
                class_record = class_records.get(class_full_name)
                class_context = ClassContext(
                    module=module_name,
                    path=str(path),
                    qualname=node.name,
                    full_name=class_full_name,
                    lineno=int(getattr(node, "lineno", class_record.lineno if class_record else 0) or 0),
                    end_lineno=int(getattr(node, "end_lineno", class_record.end_lineno if class_record else 0) or 0),
                    node=node,
                    source=_slice_source(source_lines, int(getattr(node, "lineno", 1) or 1), int(getattr(node, "end_lineno", getattr(node, "lineno", 1)) or 1)),
                )
                class_map[node.name] = class_context
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.append(
                            _build_function_context(
                                module_name=module_name,
                                path=path,
                                source_lines=source_lines,
                                node=child,
                                class_name=node.name,
                                function_record=function_records.get(f"{module_name}.{node.name}.{child.name}"),
                            )
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(
                    _build_function_context(
                        module_name=module_name,
                        path=path,
                        source_lines=source_lines,
                        node=node,
                        class_name="",
                        function_record=function_records.get(f"{module_name}.{node.name}"),
                    )
                )

        for function in functions:
            if function.class_name and function.class_name in class_map:
                class_map[function.class_name].methods.append(function)

        modules[module_name] = ModuleContext(
            module=module_name,
            path=str(path),
            tree=tree,
            source=source,
            source_lines=source_lines,
            record=module_record,
            functions=functions,
            classes=list(class_map.values()),
            top_level_constants=_collect_top_level_constants(tree),
        )
    return RepositoryAstContext(root=str(root), inventory=inventory, modules=modules)


def attach_parents(node: ast.AST) -> None:
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "_parent", parent)


def parent_of(node: ast.AST) -> ast.AST | None:
    value = getattr(node, "_parent", None)
    return value if isinstance(value, ast.AST) else None


def expression_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = expression_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return expression_name(node.func)
    if isinstance(node, ast.Subscript):
        return expression_name(node.value)
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return ""


def annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return expression_name(node)


def decorator_names(node: ast.AST) -> list[str]:
    decorator_list = getattr(node, "decorator_list", [])
    return [expression_name(item) for item in decorator_list if expression_name(item)]


def cyclomatic_complexity(node: ast.AST) -> int:
    return 1 + sum(1 for child in walk_without_nested_defs(node) if isinstance(child, COMPLEXITY_NODES))


def max_nesting_depth(node: ast.AST) -> int:
    def visit(statements: list[ast.stmt], depth: int) -> int:
        best = depth
        for statement in statements:
            if isinstance(statement, CONTROL_FLOW_NODES):
                next_depth = depth + 1
                best = max(best, next_depth)
                for block in child_statement_blocks(statement):
                    best = max(best, visit(block, next_depth))
            elif isinstance(statement, ast.ExceptHandler):
                next_depth = depth + 1
                best = max(best, next_depth, visit(statement.body, next_depth))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            else:
                for block in child_statement_blocks(statement):
                    best = max(best, visit(block, depth))
        return best

    return visit(list(getattr(node, "body", [])), 0)


def child_statement_blocks(node: ast.AST) -> list[list[ast.stmt]]:
    blocks: list[list[ast.stmt]] = []
    for field_name in ("body", "orelse", "finalbody"):
        value = getattr(node, field_name, None)
        if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
            blocks.append(value)
    handlers = getattr(node, "handlers", None)
    if isinstance(handlers, list):
        blocks.extend(handler.body for handler in handlers if isinstance(handler, ast.ExceptHandler) and handler.body)
    cases = getattr(node, "cases", None)
    if isinstance(cases, list):
        blocks.extend(case.body for case in cases if getattr(case, "body", None))
    return blocks


def walk_without_nested_defs(node: ast.AST) -> Iterable[ast.AST]:
    stack = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield current
        stack.extend(ast.iter_child_nodes(current))


def numeric_literals(node: ast.AST, *, module_constants: set[str] | None = None, include_common: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in walk_without_nested_defs(node):
        if not isinstance(child, ast.Constant):
            continue
        if isinstance(child.value, bool) or not isinstance(child.value, (int, float)):
            continue
        if not include_common and child.value in COMMON_MAGIC_NUMBERS:
            continue
        if _is_exempt_numeric_literal(child, module_constants or set()):
            continue
        rows.append(
            {
                "value": child.value,
                "lineno": int(getattr(child, "lineno", 0) or 0),
                "col_offset": int(getattr(child, "col_offset", 0) or 0),
            }
        )
    return rows


def contains_nested_loop(node: ast.AST) -> bool:
    for child in walk_without_nested_defs(node):
        if not isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            continue
        if any(isinstance(grandchild, (ast.For, ast.AsyncFor, ast.While)) for grandchild in walk_without_nested_defs(child)):
            return True
    return False


def iter_loop_nodes(node: ast.AST) -> Iterable[ast.For | ast.AsyncFor | ast.While]:
    for child in walk_without_nested_defs(node):
        if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
            yield child


def iter_calls(node: ast.AST) -> Iterable[ast.Call]:
    for child in walk_without_nested_defs(node):
        if isinstance(child, ast.Call):
            yield child


def has_broad_exception_handler(node: ast.AST) -> list[ast.ExceptHandler]:
    handlers: list[ast.ExceptHandler] = []
    for child in walk_without_nested_defs(node):
        if not isinstance(child, ast.ExceptHandler):
            continue
        handler_type = expression_name(child.type)
        if child.type is None or handler_type in {"Exception", "BaseException", "builtins.Exception", "builtins.BaseException"}:
            handlers.append(child)
    return handlers


def handler_swallows_exception(handler: ast.ExceptHandler) -> bool:
    if not handler.body:
        return True
    if any(isinstance(statement, ast.Raise) for statement in handler.body):
        return False
    allowed_exprs = {"logger.debug", "logger.info", "logger.warning", "logger.error", "logging.debug", "logging.info", "logging.warning", "logging.error"}
    for statement in handler.body:
        if isinstance(statement, (ast.Pass, ast.Continue, ast.Break)):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            if expression_name(statement.value.func) in allowed_exprs:
                continue
        if isinstance(statement, ast.Return):
            if statement.value is None:
                continue
            if isinstance(statement.value, ast.Constant) and statement.value.value in {None, False, "", 0}:
                continue
            if isinstance(statement.value, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
                continue
        return False
    return True


def mutates_nonlocal_state(function: FunctionContext) -> bool:
    local_names = local_assignment_names(function.node)
    for child in walk_without_nested_defs(function.node):
        if isinstance(child, (ast.Global, ast.Nonlocal)):
            return True
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = []
            if isinstance(child, ast.Assign):
                targets = list(child.targets)
            else:
                targets = [child.target]
            for target in targets:
                if _target_is_external_mutation(target, local_names):
                    return True
        if isinstance(child, ast.Delete):
            if any(_target_is_external_mutation(target, local_names) for target in child.targets):
                return True
    return False


def has_hidden_side_effect(function: FunctionContext) -> bool:
    return bool(hidden_side_effect_reasons(function))


def hidden_side_effect_reasons(function: FunctionContext) -> list[str]:
    if not any(function.name.startswith(prefix) for prefix in QUERY_LIKE_PREFIXES):
        return []
    reasons: list[str] = []
    local_names = local_assignment_names(function.node)
    if mutates_nonlocal_state(function):
        reasons.append("mutates non-local state")
    for call in iter_calls(function.node):
        label = expression_name(call.func)
        if _is_local_mutation_call(call, local_names):
            continue
        if is_effectful_call(label):
            reasons.append(f"effectful call `{label}`")
    return reasons


def local_assignment_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in walk_without_nested_defs(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
    return names


def is_effectful_call(label: str) -> bool:
    lowered = str(label or "").strip().lower()
    if not lowered or lowered in PURE_SERIALIZATION_CALLS:
        return False
    if any(lowered == marker or lowered.endswith(f".{marker}") for marker in EFFECTFUL_CALL_MARKERS):
        return True
    tokens = _call_tokens(lowered)
    return any(marker in tokens for marker in EFFECTFUL_CALL_MARKERS)


def is_expensive_call(label: str) -> bool:
    lowered = str(label or "").lower()
    if any(marker in lowered for marker in EXPENSIVE_CALL_MARKERS if "." in marker):
        return True
    if lowered in PURE_BUILTINS or any(lowered == marker or lowered.endswith(marker) for marker in CHEAP_CALL_MARKERS):
        return False
    return any(marker in lowered for marker in EXPENSIVE_CALL_MARKERS)


def reads_like_pure(function: FunctionContext) -> bool:
    if function.is_method and "staticmethod" not in function.decorator_names:
        return False
    if mutates_nonlocal_state(function):
        return False
    for call in iter_calls(function.node):
        label = expression_name(call.func)
        if is_effectful_call(label):
            return False
        if "." in label and label.split(".", 1)[0] not in {"math", "statistics", "numpy", "np"}:
            return False
        if "." not in label and label and label not in PURE_BUILTINS:
            return False
    return any(isinstance(child, ast.Return) for child in walk_without_nested_defs(function.node))


def has_guard_clause_shape(function: FunctionContext) -> bool:
    body = list(function.node.body)
    if len(body) < 2:
        return False
    guard_count = 0
    for statement in body[:3]:
        if not isinstance(statement, ast.If):
            break
        if len(statement.body) != 1:
            break
        if isinstance(statement.body[0], (ast.Return, ast.Raise)):
            guard_count += 1
            continue
        break
    return guard_count >= 1 and max_nesting_depth(function.node) <= 2


def is_test_module_name(module_name: str) -> bool:
    return module_name == "tests" or module_name.startswith("tests.") or ".tests" in module_name or module_name.endswith("_test")


def config_like_name(name: str) -> bool:
    lowered = str(name or "").lower()
    return any(token in lowered for token in ("config", "settings", "params", "options", "spec"))


def artifact_like_name(name: str) -> bool:
    lowered = str(name or "").lower()
    return any(token in lowered for token in ("artifact", "report", "snapshot", "result", "inventory", "payload", "record"))


def boundary_object_like_name(name: str) -> bool:
    lowered = str(name or "").lower()
    return artifact_like_name(name) or any(token in lowered for token in ("request", "response", "schema"))


def _call_tokens(label: str) -> set[str]:
    tokens = {token for token in TOKEN_SPLIT_RE.split(label) if token}
    dotted = [part for part in label.split(".") if part]
    tokens.update(dotted)
    return {token.lower() for token in tokens if token}


def _is_local_mutation_call(call: ast.Call, local_names: set[str]) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    method_name = str(call.func.attr or "").lower()
    if method_name not in LOCAL_MUTATION_METHODS:
        return False
    base_name = expression_name(call.func.value).split(".", 1)[0]
    return base_name in local_names


def _build_function_context(
    *,
    module_name: str,
    path: Path,
    source_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    class_name: str,
    function_record: FunctionRecord | None,
) -> FunctionContext:
    lineno = int(getattr(node, "lineno", function_record.lineno if function_record else 0) or 0)
    end_lineno = int(getattr(node, "end_lineno", function_record.end_lineno if function_record else lineno) or lineno)
    qualname = f"{class_name}.{node.name}" if class_name else node.name
    return FunctionContext(
        module=module_name,
        path=str(path),
        class_name=class_name,
        qualname=qualname,
        full_name=f"{module_name}.{qualname}",
        lineno=lineno,
        end_lineno=end_lineno,
        node=node,
        source=_slice_source(source_lines, lineno, end_lineno),
        resolved_calls=list(function_record.resolved_calls) if function_record else [],
        unresolved_calls=list(function_record.unresolved_calls) if function_record else [],
    )


def _slice_source(source_lines: list[str], lineno: int, end_lineno: int) -> str:
    if lineno <= 0 or end_lineno <= 0:
        return ""
    return "".join(source_lines[lineno - 1 : end_lineno])


def _collect_top_level_constants(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.isupper():
            names.add(node.target.id)
    return names


def _is_exempt_numeric_literal(node: ast.Constant, module_constants: set[str]) -> bool:
    parent = parent_of(node)
    if isinstance(parent, ast.UnaryOp):
        parent = parent_of(parent)
    if isinstance(parent, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id.isupper() for target in parent.targets)
    if isinstance(parent, ast.AnnAssign):
        return isinstance(parent.target, ast.Name) and parent.target.id.isupper()
    if isinstance(parent, ast.keyword):
        return parent.arg in {"timeout", "axis", "ddof"}
    if isinstance(parent, ast.Compare):
        return False
    if isinstance(parent, ast.Subscript):
        return True
    current = parent
    while isinstance(current, ast.AST):
        if isinstance(current, ast.ClassDef) and any("Enum" in base for base in [expression_name(item) for item in current.bases]):
            return True
        current = parent_of(current)
    return False


def _target_is_external_mutation(target: ast.AST, local_names: set[str]) -> bool:
    if isinstance(target, ast.Attribute):
        base = expression_name(target.value)
        return base in {"self", "cls"} or (base and base not in local_names)
    if isinstance(target, ast.Subscript):
        base = expression_name(target.value)
        return base not in local_names
    if isinstance(target, ast.Name):
        return False
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_is_external_mutation(child, local_names) for child in target.elts)
    return False
