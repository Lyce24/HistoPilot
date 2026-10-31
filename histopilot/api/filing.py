"""What a commit checks, run when an agent files it and before it is parked for a person.

A scoped token's commit does not run. The service parks it as a request, a person approves
it, and the approval replays it under that person's session, which checks it all again
(api/scopes.py, docs/agents.md#approvals). Filing runs the commit's own checks of the request
and the project's records first, writing nothing, and refuses a request they refuse with the
commit's own answer. So no request is parked that could only fail when approved: a start for
a design nobody froze, a freeze at a revision that has moved on, an Apply models run whose
preview went stale.

Each router registers a check beside the commit route it serves, calling the same service
code the route calls. A commit route without a check is parked unchecked.
"""

from collections.abc import Callable, Collection
from typing import Any

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from histopilot.api.route_classes import COMMIT, ROUTE_CLASSES, expand, route_template

# check(parameters of the route's path, the body as the route's request model reads it)
Check = Callable[[dict[str, str], Any], object]


class FilingChecks:
    """The commit routes whose checks run when an agent files one; see the module doc."""

    def __init__(self) -> None:
        self._checks: dict[tuple[str, str], tuple[type[BaseModel], Check]] = {}

    def add(self, method: str, path: str, model: type[BaseModel], check: Check) -> None:
        """Check commits to ``path``, written as route_classes writes it (`{P}` for a
        project's routes), with ``check``. ``model`` is the route's own request model."""
        template = expand(path)
        if ROUTE_CLASSES.get((method, template)) != COMMIT:
            raise ValueError(f"{method} {template} is not a commit route.")
        self._checks[(method, template)] = (model, check)

    def routes(self) -> set[tuple[str, str]]:
        return set(self._checks)

    def check(self, method: str, path: str, body: Any) -> None:
        """Run the check of the route ``path`` names, if it has one, raising what the commit
        would: the service's coded error, or the validation error of a refused body."""
        template = route_template(method, path)
        found = self._checks.get((method, template)) if template else None
        if found is None or template is None:
            return
        model, check = found
        try:
            payload = model.model_validate(body)
        except ValidationError as error:
            raise RequestValidationError(
                [
                    {**item, "loc": ("body", *item["loc"])}
                    for item in error.errors(include_url=False)
                ]
            ) from error
        check(parameters(template, path), payload)


def parameters(template: str, path: str) -> dict[str, str]:
    """The path parameters a concrete path gives a route template."""
    names, values = template.split("/"), path.rstrip("/").split("/")
    return {
        name[1:-1]: value
        for name, value in zip(names, values, strict=True)
        if name.startswith("{") and name.endswith("}")
    }


def choice(found: dict[str, str], name: str, choices: Collection[str]) -> str:
    """A path parameter the route types as one of ``choices``, refused as the route refuses
    any other value."""
    value = found[name]
    if value not in choices:
        expected = ", ".join(repr(item) for item in choices)
        raise RequestValidationError(
            [
                {
                    "type": "literal_error",
                    "loc": ("path", name),
                    "msg": f"Input should be {expected}",
                    "input": value,
                }
            ]
        )
    return value
