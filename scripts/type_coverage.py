"""Report how much of the service core carries types, as a Markdown table.

    uv run python scripts/type_coverage.py

Two measures for each package that pyright checks (``[tool.pyright]`` in pyproject.toml):

- annotations: the functions, methods and nested functions whose parameters (other than
  ``self`` and ``cls``) and return are all annotated; ``__init__`` needs no return annotation;
- known types: ``pyright --verifytypes``, the public symbols whose types pyright knows. Unknown
  types that come from outside the package are not counted against it.

CI appends the table to the job summary. It reports and never fails on the numbers; the type
check itself is ``uv run pyright``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("histopilot.domain", "histopilot.application", "histopilot.api")


@dataclass
class Counts:
    functions: int = 0
    annotated_functions: int = 0
    parameters: int = 0
    annotated_parameters: int = 0
    returns: int = 0
    annotated_returns: int = 0
    symbols: int = 0
    known_symbols: int = 0

    def __iadd__(self, other: Counts) -> Counts:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))
        return self


def annotation_counts(folder: Path) -> Counts:
    counts = Counts()
    for path in sorted(folder.rglob("*.py")):
        _count(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), counts, False)
    return counts


def _count(node: ast.AST, counts: Counts, in_class: bool) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            parameters = _parameters(child, in_class)
            annotated = sum(parameter.annotation is not None for parameter in parameters)
            returns = child.returns is not None or child.name == "__init__"
            counts.functions += 1
            counts.annotated_functions += annotated == len(parameters) and returns
            counts.parameters += len(parameters)
            counts.annotated_parameters += annotated
            counts.returns += 1
            counts.annotated_returns += returns
        _count(child, counts, isinstance(child, ast.ClassDef))


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef, method: bool) -> list[ast.arg]:
    arguments = function.args
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    parameters += [extra for extra in (arguments.vararg, arguments.kwarg) if extra is not None]
    static = any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
        for decorator in function.decorator_list
    )
    if method and not static and (arguments.posonlyargs or arguments.args):
        return parameters[1:]  # self or cls
    return parameters


def known_types(package: str) -> tuple[int, int]:
    """The package's public symbols whose types pyright knows, and all of them."""
    # The editable install finds the package through an import hook that pyright cannot
    # follow, so pyright reads it from this checkout through the interpreter's search path.
    search_path = os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH"))))
    command = [sys.executable, "-m", "pyright", "--pythonpath", sys.executable]
    command += ["--verifytypes", package, "--ignoreexternal", "--outputjson"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": search_path},
        capture_output=True,
        text=True,
        check=False,  # an incomplete package exits 1, with the report on stdout
    )
    try:
        symbols = json.loads(completed.stdout)["typeCompleteness"]["exportedSymbolCounts"]
    except (ValueError, KeyError) as error:
        raise SystemExit(
            f"pyright --verifytypes {package} failed:\n{completed.stderr or completed.stdout}"
        ) from error
    known = symbols["withKnownType"]
    return known, known + symbols["withAmbiguousType"] + symbols["withUnknownType"]


def _share(part: int, whole: int) -> str:
    return f"{part:,} of {whole:,} ({100 * part / whole:.1f}%)" if whole else "none"


def report() -> str:
    lines = [
        "### Type coverage",
        "",
        (
            "| Package | Fully annotated functions | Annotated parameters | Annotated returns "
            "| Public symbols with known types |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    total = Counts()
    for package in PACKAGES:
        counts = annotation_counts(ROOT.joinpath(*package.split(".")))
        counts.known_symbols, counts.symbols = known_types(package)
        total += counts
        lines.append(_row(f"`{package}`", counts))
    lines.append(_row("**Core**", total))
    return "\n".join(lines) + "\n"


def _row(name: str, counts: Counts) -> str:
    cells = (
        _share(counts.annotated_functions, counts.functions),
        _share(counts.annotated_parameters, counts.parameters),
        _share(counts.annotated_returns, counts.returns),
        _share(counts.known_symbols, counts.symbols),
    )
    return f"| {name} | " + " | ".join(cells) + " |"


if __name__ == "__main__":
    sys.stdout.write(report())
