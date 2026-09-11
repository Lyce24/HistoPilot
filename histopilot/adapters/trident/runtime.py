"""Read installed source metadata without importing TRIDENT or initializing CUDA."""

import ast
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import PATCH_ENCODERS, SLIDE_ENCODERS, TridentOptions


def _source_tree(path: Path) -> ast.Module | None:
    try:
        if path.stat().st_size > 2_000_000:
            return None
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, SyntaxError):
        return None


def _parser_options(script: Path) -> dict:
    tree = _source_tree(script)
    options = {}
    if tree is None:
        return options
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        for arg in node.args:
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                continue
            if not arg.value.startswith("--"):
                continue
            info = {}
            for keyword in node.keywords:
                if keyword.arg == "type" and isinstance(keyword.value, ast.Name):
                    info["type"] = keyword.value.id
                if keyword.arg in {"default", "choices", "action", "nargs"}:
                    try:
                        info[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        pass
            options[arg.value[2:]] = info
    return options


def _encoders(root: Path, level: str) -> list[str]:
    tree = _source_tree(root / "trident" / f"{level}_encoder_models" / "load.py")
    if tree is None:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "encoder_registry"
            for target in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                return [
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ]
    return []


def _roots_for_interpreter(python: Path) -> list[Path]:
    roots = []
    for site in python.parent.parent.glob("lib/python*/site-packages"):
        # Editable installations, including the commonly used conda environment.
        for path in site.glob("*trident*.pth"):
            try:
                for line in path.read_text().splitlines():
                    if line and not line.startswith(("import ", "#")):
                        root = Path(line).expanduser()
                        roots.append(root if root.is_absolute() else site / root)
            except OSError:
                pass
        for path in site.glob("trident-*.dist-info/direct_url.json"):
            try:
                url = urlparse(json.loads(path.read_text())["url"])
                if url.scheme == "file":
                    roots.append(Path(unquote(url.path)))
            except (OSError, ValueError, KeyError):
                pass
        roots.extend((site, site / "trident"))
    return roots


def discover_runtime(
    python_path: str | Path | None = None, trident_root: str | Path | None = None
) -> dict:
    """Configuration is server-owned; this function never executes an interpreter.

    Availability means interpreter and batch source are present. Model-specific
    dependencies and checkpoint access are checked by the isolated worker.
    """
    configured_python = python_path or os.environ.get("HISTOPILOT_TRIDENT_PYTHON")
    configured_root = trident_root or os.environ.get("HISTOPILOT_TRIDENT_ROOT")
    if configured_python:
        candidate = str(configured_python)
        python = Path(shutil.which(candidate) or candidate).expanduser().absolute()
    else:
        candidates = [
            Path.home() / "miniconda3/envs/trident/bin/python",
            Path.home() / "anaconda3/envs/trident/bin/python",
            Path(sys.executable),
        ]
        python = next((path for path in candidates if path.is_file()), candidates[-1])
    roots = (
        [Path(configured_root).expanduser().absolute()]
        if configured_root
        else [
            Path(__file__).resolve().parents[3] / ".local/TRIDENT",
            Path(__file__).resolve().parents[3] / ".local/trident",
            *_roots_for_interpreter(python),
            Path.home() / "projects/TRIDENT",
            Path.home() / "projects/trident",
        ]
    )
    root = next((path for path in roots if (path / "run_batch_of_slides.py").is_file()), None)
    script = root / "run_batch_of_slides.py" if root else None
    available = bool(script and python.is_file() and os.access(python, os.X_OK))
    if not python.is_file() or not os.access(python, os.X_OK):
        reason = "TRIDENT Python interpreter does not exist or is not executable. Set HISTOPILOT_TRIDENT_PYTHON."
    elif script is None:
        reason = "TRIDENT batch source was not found. Set HISTOPILOT_TRIDENT_ROOT to a checkout containing run_batch_of_slides.py."
    else:
        reason = "TRIDENT source and interpreter found. Model dependencies and checkpoint access are checked in the worker."
    return {
        "available": available,
        "pythonPath": str(python),
        "tridentRoot": str(root) if root else str(configured_root) if configured_root else None,
        "scriptPath": str(script) if script else None,
        "message": reason,
        "supportedOptions": sorted(_parser_options(script)) if script else [],
        "patchEncoders": _encoders(root, "patch") if root else [],
        "slideEncoders": _encoders(root, "slide") if root else [],
        "validationLevel": "source",
        "scriptSha256": hashlib.sha256(script.read_bytes()).hexdigest() if script else None,
    }


def build_command(
    options: TridentOptions,
    *,
    python_path: str | Path,
    trident_root: str | Path,
    wsi_dir: str | Path,
    job_dir: str | Path,
    custom_list_of_wsis: str | Path | None = None,
) -> list[str]:
    """Construct an argv list; no shell interpolation or process launch occurs."""
    script = Path(trident_root) / "run_batch_of_slides.py"
    parser = _parser_options(script)
    if not script.is_file() or not parser:
        raise ValueError("TRIDENT batch parser could not be inspected")
    if not {"wsi_dir", "job_dir", "task"}.issubset(parser):
        raise ValueError("TRIDENT batch script does not expose the expected CLI")
    root = Path(trident_root)
    for level, name, fallback in (
        ("patch", options.patch_encoder, PATCH_ENCODERS),
        ("slide", options.slide_encoder, SLIDE_ENCODERS),
    ):
        available = _encoders(root, level) or fallback
        if name is not None and name not in available:
            raise ValueError(f"Unknown {level} encoder in this TRIDENT installation: {name}")
    values = options.model_dump(mode="json")
    if custom_list_of_wsis is not None:
        values["custom_list_of_wsis"] = str(custom_list_of_wsis)
    argv = [
        str(python_path),
        "-u",
        str(script),
        "--wsi_dir",
        str(wsi_dir),
        "--job_dir",
        str(job_dir),
    ]
    baseline = TridentOptions().model_dump(mode="json")
    for name, value in values.items():
        if value is None:
            continue
        if name not in parser:
            if value != baseline[name] or name in {"task", "patch_encoder", "mag", "patch_size"}:
                raise ValueError(
                    f"This TRIDENT installation does not support --{name}; update TRIDENT"
                )
            continue
        choices = parser[name].get("choices")
        if choices and value not in choices:
            raise ValueError(f"This TRIDENT installation does not support {value!r} for --{name}")
        if isinstance(value, bool):
            if value:
                argv.append(f"--{name}")
        elif isinstance(value, list):
            argv.extend([f"--{name}", *map(str, value)])
        else:
            # Separate argv values retain paths with whitespace verbatim.
            if parser[name].get("type") == "int" and isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(f"This TRIDENT installation requires an integer for --{name}")
                value = int(value)
            argv.extend([f"--{name}", str(value)])
    return argv
