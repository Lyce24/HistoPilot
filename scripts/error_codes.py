"""Scan histopilot/ for the service's error codes and write docs/error-codes.md.

python scripts/error_codes.py          rewrite docs/error-codes.md
python scripts/error_codes.py --check  exit 1 when the registry or the page is out of date
python scripts/error_codes.py --sites  list each code with its statuses and raise sites
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The scan needs only the standard library and the registry, not an installed package.
sys.path.insert(0, str(ROOT))

from histopilot.api import error_codes as registry  # noqa: E402
from histopilot.client.errors import NEXT_STEPS  # noqa: E402

PACKAGE = ROOT / "histopilot"
DOC = ROOT / "docs" / "error-codes.md"
CODE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")
MISSING = object()


@dataclass(frozen=True)
class Slot:
    position: int | None
    keyword: str
    default: object = MISSING


@dataclass(frozen=True)
class Carrier:
    message: Slot
    code: Slot
    status: Slot | None


def _filesystem_code(status):
    from histopilot.storage.filesystem import FilesystemError

    return FilesystemError("", status).code


# Every exception class that carries a code. tests/test_error_codes.py fails when a class
# in histopilot/ sets self.code and is missing here.
CARRIERS = {
    "StorageError": Carrier(Slot(0, "message"), Slot(1, "code"), Slot(2, "status_code", 409)),
    "WorkspaceError": Carrier(Slot(0, "message"), Slot(None, "code"), Slot(1, "status_code", 422)),
    "FilesystemError": Carrier(
        Slot(0, "message"), Slot(None, "code", _filesystem_code), Slot(1, "status_code", 400)
    ),
    # Filter failures become preview findings; a blocked commit can raise them as its code.
    "FilterFailure": Carrier(Slot(1, "message"), Slot(0, "code"), None),
}

# The page's meaning for codes whose raise sites carry no readable message, or whose
# first message is too narrow for a code raised in many places.
MEANINGS = {
    "AGREEMENT_UNAVAILABLE": "Agreement cannot be computed for this run's predictions.",
    "API_ENDPOINT_UNKNOWN": "No API route has this path.",
    "API_METHOD_NOT_ALLOWED": "The API route does not accept this HTTP method.",
    "CLINICAL_TARGET_LEAKAGE": "The field cannot be a clinical input, for example the target.",
    "CLINICAL_VALUES_INVALID": "Frozen clinical values cannot be used for the selected fields.",
    "COMPARISON_EVIDENCE_INVALID": "Saved predictions of a compared run cannot be read.",
    "COMPARISON_PATIENT_MISMATCH": "The compared runs do not predict the same patients.",
    "COMPUTE_NOT_FOUND": "The compute record does not exist.",
    "DRAFT_FROZEN": "The draft or record is frozen. Copy it to make changes.",
    "EVALUATION_INPUTS_CHANGED": "The evaluation's frozen inputs changed. Create and review a new one.",
    "INTERPRETATION_DATASET_FOLDER_UNAVAILABLE": (
        "The frozen dataset has no usable slide folder. Link it in Datasets, then freeze "
        "the feature bundle again."
    ),
    "INTERPRETATION_INPUT_INVALID": "Attention inputs fail validation.",
    "INVALID_INPUT": "A value is missing, too long or malformed.",
    "INVALID_TRIDENT_OPTIONS": "The TRIDENT extraction options fail validation.",
    "LIFECYCLE_LIMIT": "Workspace cleanup metadata reached a size or revision limit.",
    "MIL_BAG_PLANNING_INVALID": "MIL bags cannot be planned for these features.",
    "MORPHOLOGY_INVALID": "Features, coordinates or slides cannot be explored as requested.",
    "MORPHOLOGY_SLIDE_CHANGED": "The slide file changed since it was prepared. Reopen the slide.",
    "OPERATION_CONFLICT": "This operation ID was already used for a different request.",
    "OUTPUT_BUSY": "Another job is using this output folder. Retry after it finishes.",
    "PACK_SOURCE_CHANGED": "The pack's files changed since verification. Verify the pack again.",
    "PORTABILITY_INVALID": "An archive, restore or relink request, or the archive itself, is invalid.",
    "PREVIEW_STALE": "Inputs changed since the preview. Preview again, then send its new hash.",
    "PROJECT_BUSY": "Another operation is writing this project. Retry after it finishes.",
    "RECORD_TRASHED": "The record, or one it uses, is in Trash. Restore it first.",
    "REFIT_BAG_PLANNING_INVALID": "MIL bags cannot be planned for this refit.",
    "REQUEST_INVALID": (
        "The body, path or query does not match the route's schema. "
        "`detail` lists each rejected field."
    ),
    "REVISION_CONFLICT": "The record changed since it was read. Reload it, then retry.",
    "SLIDE_VIEWER_UNAVAILABLE": "Slide viewing needs optional imaging libraries that are missing.",
    "STORAGE_CORRUPT": "Stored project data failed validation.",
    "STORAGE_UNSUPPORTED": "Project storage needs POSIX file locking and directory sync.",
    "STORAGE_WRITE_FAILED": "Project storage cannot be written.",
    "TASK_CENTER_UNAVAILABLE": "The Task Center store cannot be reached or updated right now.",
    "TASK_INVALID": "A task specification is invalid.",
    "TRAINING_METRIC_UNAVAILABLE": "The selection metric is unavailable for this target.",
    "TRAINING_RECIPE_UNAVAILABLE": "The training recipe does not fit this target or data.",
}


@dataclass
class Site:
    code: str
    status: int | None
    message: str | None
    path: str
    line: int


@dataclass
class Computed:
    path: str
    line: int
    expression: str
    statuses: tuple = ()


@dataclass
class Scan:
    sites: list = field(default_factory=list)
    computed: list = field(default_factory=list)
    finding_messages: dict = field(default_factory=dict)

    def codes(self) -> dict:
        found = defaultdict(list)
        for site in self.sites:
            found[site.code].append(site)
        return dict(found)


class _Module:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.relative_to(ROOT).as_posix()
        self.tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        self.parents = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.constants = {}
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name):
                    self.constants[node.targets[0].id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    self.constants[node.target.id] = node.value
        self.imports = {}
        package = self.name.removesuffix(".py").split("/")[:-1]
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                parts = package[: len(package) - node.level + 1] if node.level else []
                base = ".".join([*parts, *([node.module] if node.module else [])])
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (base, alias.name)

    def function(self, node):
        while node in self.parents:
            node = self.parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None


def _modules() -> list:
    # The CLI's own codes (client/, commands/) belong to the CLI contract, not the service.
    skipped = {"static", "client", "commands"}
    return [
        _Module(path)
        for path in sorted(PACKAGE.rglob("*.py"))
        if not skipped & set(path.relative_to(PACKAGE).parts[:1])
    ]


def _parameters(function) -> tuple[list, list, dict]:
    """Positional names, all names and defaults of a function's parameters."""
    arguments = function.args
    positional = [item.arg for item in (*arguments.posonlyargs, *arguments.args)]
    defaults = dict(
        zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults)
    )
    for item, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        if default is not None:
            defaults[item.arg] = default
    return positional, [*positional, *(item.arg for item in arguments.kwonlyargs)], defaults


def _argument(call, slot: Slot | None):
    """The expression a call passes for a slot, else the slot's default, else MISSING."""
    if slot is None:
        return MISSING
    for keyword in call.keywords:
        if keyword.arg == slot.keyword:
            return keyword.value
    if slot.position is not None and len(call.args) > slot.position:
        value = call.args[slot.position]
        return MISSING if isinstance(value, ast.Starred) else value
    return slot.default


def _response_parts(call):
    """(detail, code, status) of JSONResponse({"detail": …, "code": …}, status_code=…)."""
    content = call.args[0] if call.args else None
    for keyword in call.keywords:
        if keyword.arg == "content":
            content = keyword.value
    if not isinstance(content, ast.Dict):
        return None
    values = {
        key.value: value
        for key, value in zip(content.keys, content.values)
        if isinstance(key, ast.Constant)
    }
    if "code" not in values or "detail" not in values:
        return None
    status = call.args[1] if len(call.args) > 1 else MISSING
    for keyword in call.keywords:
        if keyword.arg == "status_code":
            status = keyword.value
    return values["detail"], values["code"], status


class _Scanner:
    def __init__(self, modules):
        self.modules = modules
        # (module, function) -> (function node, construction node, parts). A factory is
        # a function that passes a parameter on as an error's message, code or status,
        # such as the modules' _error(message, code=…, status=…) helpers.
        self.factories = {}
        self.called = set()
        self._find_factories()

    def _find_factories(self):
        changed = True
        while changed:
            changed = False
            for module in self.modules:
                for node in ast.walk(module.tree):
                    parts = self._parts(module, node)
                    function = module.function(node) if parts else None
                    if function is None:
                        continue
                    names = _parameters(function)[1]
                    if not any(isinstance(item, ast.Name) and item.id in names for item in parts):
                        continue
                    key = (module.name, function.name)
                    known = self.factories.get(key)
                    if known is None:
                        self.factories[key] = (function, node, parts)
                        changed = True
                    elif known[1] is not node:
                        raise SystemExit(
                            f"{module.name}: {function.name} builds more than one error from its "
                            "parameters, or shares its name with another such helper. Split or "
                            "rename it so the scan can read its raise sites."
                        )

    def _factory(self, module, call):
        """The factory a call reaches: a local or imported function, or a self method."""
        if isinstance(call.func, ast.Attribute):
            owner = call.func.value
            if isinstance(owner, ast.Name) and owner.id in {"self", "cls"}:
                if (module.name, call.func.attr) in self.factories:
                    return module.name, call.func.attr
            return None
        if not isinstance(call.func, ast.Name):
            return None
        name = call.func.id
        if (module.name, name) in self.factories:
            return module.name, name
        if name in module.imports:
            base, original = module.imports[name]
            path = base.replace(".", "/")
            for candidate in (f"{path}.py", f"{path}/__init__.py"):
                if (candidate, original) in self.factories:
                    return candidate, original
        return None

    def _parts(self, module, node):
        """(message, code, status) expressions of an error construction, or None."""
        if not isinstance(node, ast.Call):
            return None
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in CARRIERS:
            carrier = CARRIERS[name]
            return (
                _argument(node, carrier.message),
                _argument(node, carrier.code),
                _argument(node, carrier.status) if carrier.status else None,
            )
        if name == "JSONResponse":
            return _response_parts(node)
        factory = self._factory(module, node)
        if factory is None:
            return None
        self.called.add(factory)
        function, _construction, parts = self.factories[factory]
        positional, names, defaults = _parameters(function)
        if isinstance(node.func, ast.Attribute) and positional[:1] in (["self"], ["cls"]):
            positional = positional[1:]

        def passed(expression):
            if not (isinstance(expression, ast.Name) and expression.id in names):
                return expression
            index = positional.index(expression.id) if expression.id in positional else None
            return _argument(node, Slot(index, expression.id, defaults.get(expression.id, MISSING)))

        return tuple(passed(item) for item in parts)

    def scan(self) -> Scan:
        result = Scan(finding_messages=self._finding_messages())
        # A factory whose code is a parameter names no code itself; its callers do.
        definitions = {
            construction: key
            for key, (function, construction, parts) in self.factories.items()
            if isinstance(parts[1], ast.Name) and parts[1].id in _parameters(function)[1]
        }
        for module in self.modules:
            for node in ast.walk(module.tree):
                if node in definitions:
                    continue
                parts = self._parts(module, node)
                if parts is not None:
                    self._record(result, module, node, *parts)
        for construction, (path, name) in definitions.items():
            if (path, name) not in self.called:
                # Called some way the scan cannot follow, so its codes would go unseen.
                result.computed.append(
                    Computed(path, construction.lineno, f"{name}(): no call the scan can read")
                )
        result.sites.sort(key=lambda site: (site.code, site.path, site.line))
        return result

    def _record(self, result, module, node, message, code, status):
        function = module.function(node)
        text = self._message(module, function, message)
        table = self._status_table(module, code, status)
        if table is not None:
            for value, number in table:
                result.sites.append(Site(value, number, text, module.name, node.lineno))
            return
        numbers = [None] if status is None else self._values(module, function, status)
        numbers = [number for number in numbers if isinstance(number, int)] or [None]
        if all(number is not None and number < 400 for number in numbers):
            return  # A success that carries a code, such as a commit parked for approval.
        if callable(code):
            for number in numbers:
                result.sites.append(Site(code(number), number, text, module.name, node.lineno))
            return
        if isinstance(code, ast.Attribute) and code.attr == "code":
            return  # Passes on the code of an error raised elsewhere.
        values = self._values(module, function, code)
        for value in values:
            if isinstance(value, str) and CODE.match(value):
                for number in numbers:
                    result.sites.append(Site(value, number, text, module.name, node.lineno))
        if any(not (isinstance(value, str) and CODE.match(value)) for value in values):
            expression = ast.unparse(code) if isinstance(code, ast.AST) else repr(code)
            result.computed.append(
                Computed(module.name, node.lineno, expression, tuple(n for n in numbers if n))
            )

    @staticmethod
    def _status_table(module, code, status):
        """Codes looked up in a status table, such as sdpc's _ERRORS[code]."""
        if not (
            isinstance(status, ast.Subscript)
            and isinstance(status.value, ast.Name)
            and isinstance(status.slice, ast.Name)
            and isinstance(code, ast.Name)
            and status.slice.id == code.id
        ):
            return None
        table = module.constants.get(status.value.id)
        if not isinstance(table, ast.Dict):
            return None
        return [
            (key.value, value.value)
            for key, value in zip(table.keys, table.values)
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        ]

    def _values(self, module, function, expression, depth=0) -> list:
        """The literal values an expression can take; None stands for a runtime value."""
        if expression is MISSING or expression is None or depth > 5:
            return [None]
        if not isinstance(expression, ast.AST):
            return [expression]  # A carrier's own default, such as StorageError's 409.
        if isinstance(expression, ast.Constant):
            return [expression.value]
        if isinstance(expression, ast.IfExp):
            return self._values(module, function, expression.body, depth + 1) + self._values(
                module, function, expression.orelse, depth + 1
            )
        if isinstance(expression, ast.BoolOp):
            return [
                value
                for item in expression.values
                for value in self._values(module, function, item, depth + 1)
            ]
        if isinstance(expression, ast.Name):
            assigned = _assigned(function, expression.id)
            if assigned:
                return [
                    value
                    for item in assigned
                    for value in self._values(module, function, item, depth + 1)
                ]
            if expression.id in module.constants:
                return self._values(module, None, module.constants[expression.id], depth + 1)
        return [None]

    def _message(self, module, function, expression, depth=0) -> str | None:
        if depth > 5 or not isinstance(expression, ast.AST):
            return None
        if isinstance(expression, ast.Constant):
            return expression.value if isinstance(expression.value, str) else None
        if isinstance(expression, ast.JoinedStr):
            text = []
            for item in expression.values:
                if isinstance(item, ast.Constant):
                    text.append(str(item.value))
                    continue
                value = item.value if isinstance(item, ast.FormattedValue) else None
                constant = module.constants.get(value.id) if isinstance(value, ast.Name) else None
                if isinstance(constant, ast.Constant) and isinstance(constant.value, str):
                    text.append(constant.value)
                else:
                    text.append("…")
            return "".join(text)
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            left = self._message(module, function, expression.left, depth + 1)
            right = self._message(module, function, expression.right, depth + 1)
            return (left or "…") + (right or "…") if left or right else None
        if isinstance(expression, (ast.IfExp, ast.BoolOp)):
            options = (
                [expression.body, expression.orelse]
                if isinstance(expression, ast.IfExp)
                else expression.values
            )
            for option in options:
                message = self._message(module, function, option, depth + 1)
                if message:
                    return message
        if isinstance(expression, ast.Name):
            for item in _assigned(function, expression.id):
                message = self._message(module, function, item, depth + 1)
                if message:
                    return message
            if expression.id in module.constants:
                return self._message(module, None, module.constants[expression.id], depth + 1)
        return None

    def _finding_messages(self) -> dict:
        """A message for each code that review findings name, for codes only findings carry."""
        found = {}
        for module in self.modules:
            for node in ast.walk(module.tree):
                code = message = None
                if isinstance(node, ast.Dict):
                    values = {
                        key.value: value
                        for key, value in zip(node.keys, node.values)
                        if isinstance(key, ast.Constant)
                    }
                    code, message = values.get("code"), values.get("message")
                elif isinstance(node, ast.Call) and len(node.args) >= 2:
                    code, message = node.args[0], node.args[1]
                elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
                    code, message = node.elts
                options = [code.body, code.orelse] if isinstance(code, ast.IfExp) else [code]
                for option in options:
                    if not (isinstance(option, ast.Constant) and isinstance(option.value, str)):
                        continue
                    if not CODE.match(option.value) or option.value in found:
                        continue
                    text = self._message(module, module.function(node), message)
                    if text and " " in text.strip():
                        found[option.value] = text
        return found


def _assigned(function, name) -> list:
    """Values assigned to a local name, including by tuple unpacking."""
    if function is None:
        return []
    values = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    values.append(node.value)
                elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    values += [
                        value
                        for item, value in zip(target.elts, node.value.elts)
                        if isinstance(item, ast.Name) and item.id == name
                    ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                values.append(node.value)
    return values


def scan() -> Scan:
    return _Scanner(_modules()).scan()


def _meaning(message: str) -> str:
    """One table cell: the message's first sentence when it is long."""
    text = " ".join(message.split())
    if len(text) > 140:
        end = text.find(". ", 20, 140)
        text = text[: end + 1] if end != -1 else text[:139].rsplit(" ", 1)[0] + " …"
    for character, escaped in (("|", "\\|"), ("*", "\\*"), ("<", "&lt;"), (">", "&gt;")):
        text = text.replace(character, escaped)
    return text


def rows(found: Scan) -> list:
    """(code, kind, statuses, meaning) for every registered code, in page order."""
    codes = found.codes()
    computed_statuses = defaultdict(set)
    for item in found.computed:
        computed_statuses[item.path].update(item.statuses)
    table = []
    for code, kind in registry.ERROR_CODES.items():
        sites = codes.get(code, [])
        statuses = {site.status for site in sites if site.status is not None}
        for module, listed in registry.COMPUTED.items():
            if code in listed:
                statuses |= computed_statuses[module]
        message = MEANINGS.get(code) or next(
            (site.message for site in sites if site.message), found.finding_messages.get(code)
        )
        table.append((code, kind, sorted(statuses), _meaning(message) if message else "—"))
    order = {kind: index for index, kind in enumerate(registry.KINDS)}
    return sorted(table, key=lambda row: (order.get(row[1], len(order)), row[0]))


def render(found: Scan) -> str:
    lines = [
        "# Error codes",
        "",
        "Every error the local service returns has a JSON body `{detail, code}`: `detail` says "
        "what went wrong for a person, `code` names it for a program. A commit refused over "
        "review findings also sends `findings`, each `{code, message, severity}` plus `field` "
        "when one field is at fault. For a request that fails schema validation, `detail` is "
        "the list of rejected fields.",
        "",
        "A code's kind groups codes by what the caller should do next, and the CLI turns each "
        "kind into an exit code: see [exit codes](cli-contract.md#exit-codes). A code missing "
        "from this page takes its kind from its HTTP status by the rules there.",
        "",
        "`scripts/error_codes.py` writes this page from the raise sites in `histopilot/` and "
        "the registry in `histopilot/api/error_codes.py`. Edit those, not this page.",
        "",
        "| Kind | Next move |",
        "| --- | --- |",
        *(f"| `{kind}` | {NEXT_STEPS.get(kind, '')} |" for kind in registry.KINDS),
    ]
    table = rows(found)
    aliases = set(registry.ALIASES)
    for kind in registry.KINDS:
        entries = [row for row in table if row[1] == kind and row[0] not in aliases]
        if not entries:
            continue
        lines += ["", f"## {kind}", "", "| Code | Kind | HTTP status | Meaning |"]
        lines.append("| --- | --- | --- | --- |")
        for code, _kind, statuses, meaning in entries:
            status = ", ".join(str(number) for number in statuses) or "—"
            lines.append(f"| `{code}` | {kind} | {status} | {meaning} |")
    if aliases:
        lines += [
            "",
            "## Deprecated codes",
            "",
            "The service still returns these names in some places. Treat each one as the code "
            "that replaces it.",
            "",
            "| Code | Kind | HTTP status | Replaced by |",
            "| --- | --- | --- | --- |",
        ]
        for code, kind, statuses, _meaning in table:
            if code in aliases:
                status = ", ".join(str(number) for number in statuses) or "—"
                lines.append(f"| `{code}` | {kind} | {status} | `{registry.ALIASES[code]}` |")
    return "\n".join(lines) + "\n"


def problems(found: Scan) -> list:
    """Every way the scan, the registry and the page disagree."""
    issues = []
    codes = found.codes()
    computed = {code for listed in registry.COMPUTED.values() for code in listed}
    for code in sorted(set(codes) - set(registry.ERROR_CODES)):
        sites = ", ".join(f"{site.path}:{site.line}" for site in codes[code][:3])
        issues.append(f"{code} is raised but not registered in ERROR_CODES ({sites}).")
    for code in sorted(set(registry.ERROR_CODES) - set(codes) - computed - set(registry.ALIASES)):
        issues.append(f"{code} is registered but no longer raised; remove it from ERROR_CODES.")
    for code in sorted(computed - set(registry.ERROR_CODES)):
        issues.append(f"{code} is listed in COMPUTED but not registered in ERROR_CODES.")
    for code, kind in sorted(registry.ERROR_CODES.items()):
        if kind not in registry.KINDS:
            issues.append(f"{code} has kind {kind!r}, which is not one of KINDS.")
    for kind in registry.KINDS:
        if kind not in NEXT_STEPS:
            issues.append(f"Kind {kind!r} has no next step in histopilot/client/errors.py.")
    for code in sorted(set(MEANINGS) - set(registry.ERROR_CODES)):
        issues.append(f"MEANINGS describes {code}, which is not registered.")
    for alias, target in sorted(registry.ALIASES.items()):
        if registry.ERROR_CODES.get(alias) != registry.ERROR_CODES.get(target):
            issues.append(f"Alias {alias} must be registered with the kind of {target}.")
    sites = {item.path for item in found.computed}
    for item in found.computed:
        if item.path not in registry.COMPUTED:
            issues.append(
                f"{item.path}:{item.line} raises a runtime code ({item.expression}); list the "
                "codes it can raise under its module in COMPUTED."
            )
    for module in sorted(set(registry.COMPUTED) - sites):
        issues.append(f"COMPUTED lists {module}, which no longer raises a runtime code.")
    if not DOC.is_file() or DOC.read_text(encoding="utf-8") != render(found):
        issues.append("docs/error-codes.md is out of date; run python scripts/error_codes.py.")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report drift and exit 1 on any")
    mode.add_argument("--sites", action="store_true", help="list codes and their raise sites")
    args = parser.parse_args()
    found = scan()
    if args.sites:
        for code, sites in sorted(found.codes().items()):
            statuses = sorted({site.status for site in sites if site.status is not None})
            print(f"{code} {registry.ERROR_CODES.get(code, '?')} {statuses}")
            for site in sites:
                print(f"    {site.path}:{site.line}")
        for item in found.computed:
            print(f"(runtime) {item.path}:{item.line} {item.expression}")
        return
    if args.check:
        issues = problems(found)
        for issue in issues:
            print(issue)
        raise SystemExit(1 if issues else 0)
    DOC.write_text(render(found), encoding="utf-8")
    print(f"Wrote {DOC.relative_to(ROOT)}: {len(registry.ERROR_CODES)} codes.")


if __name__ == "__main__":
    main()
