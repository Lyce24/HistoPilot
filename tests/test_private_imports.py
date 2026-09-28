"""A ratchet on private helpers imported from another HistoPilot module.

A helper that another module needs is part of its module's interface: name it without the
leading underscore where it is defined, and import that name. This count may only go down.
When you make one of these helpers public, lower ``ALLOWED`` to the new count.
"""

import ast
from pathlib import Path

import histopilot

PACKAGE = Path(histopilot.__file__).resolve().parent
ALLOWED = 21


def private_imports():
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith(
                "histopilot"
            ):
                continue
            for alias in node.names:
                if alias.name.startswith("_") and not alias.name.startswith("__"):
                    relative = path.relative_to(PACKAGE.parent)
                    found.append(f"{relative}:{node.lineno}: {node.module}.{alias.name}")
    return found


def test_private_cross_module_imports_do_not_grow():
    found = private_imports()
    assert len(found) <= ALLOWED, "Make these helpers public instead:\n" + "\n".join(found)
