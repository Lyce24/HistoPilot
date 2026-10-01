"""Client errors, one kind per exit code, as docs/cli-contract.md#exit-codes defines."""

import json

from histopilot.api.error_codes import kind_for

EXIT_CODES = {
    "internal": 1,
    "invalid": 2,
    "refused": 3,
    "conflict": 4,
    "not-found": 5,
    "unavailable": 6,
    "unconfirmed": 7,
    "timeout": 8,
    "work-failed": 9,
    "interrupted": 130,
}

# What a caller does next for each kind: the contract's "Next move" column, which the error
# code page and the skill reference print.
NEXT_STEPS = {
    "internal": "Report it with the output.",
    "invalid": "Fix the command or spec.",
    "refused": "Change the request or the state first; resent unchanged, it fails again.",
    "conflict": "Reload or preview again, then retry.",
    "not-found": "Check the ID or tag.",
    "unavailable": "Start the service, fix --url, or sign in.",
    "unconfirmed": "Review data.preview, then rerun with --yes; a person approves a pending "
    "request with `histopilot confirm`.",
    "timeout": "Wait again, or rerun to replay.",
    "work-failed": "`histopilot tasks show`, then resume or retry.",
    "interrupted": "Rerun to replay.",
}

# The boundary's refusals from services that predate error codes, matched by their text.
_BOUNDARY_REFUSALS = {
    "Unrecognized local service Host.",
    "This browser Origin is not permitted.",
    "Cross-site browser requests are not permitted.",
    "Unrecognized browser request context.",
    "Same-site browser requests require an approved Origin.",
    "A valid local session token is required.",
}


class ClientError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        kind: str,
        status: int | None = None,
        findings: list[dict] | None = None,
        data=None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.kind = kind
        self.status = status
        self.findings = list(findings or [])
        # What there is to review despite the error, such as a commit's preview.
        self.data = data

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.kind, 1)

    def to_json(self) -> dict:
        return {
            "code": self.code,
            "kind": self.kind,
            "message": self.message,
            "status": self.status,
            "findings": self.findings,
        }


def from_response(status: int, body: bytes) -> ClientError:
    detail, code, findings = None, None, []
    try:
        payload = json.loads(body) if body else None
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail, code = payload.get("detail"), payload.get("code")
        findings = [item for item in payload.get("findings") or [] if isinstance(item, dict)]
    elif body:
        detail = body.decode("utf-8", "replace").strip()[:500]
    if isinstance(detail, list):
        # Request validation: one finding per rejected field.
        findings = [_validation_finding(item) for item in detail if isinstance(item, dict)]
        shown = "; ".join(
            f"{item['field']}: {item['message']}" if item.get("field") else item["message"]
            for item in findings[:3]
        )
        detail = f"The service rejected the request: {shown}" if shown else None
        code = code or "REQUEST_INVALID"
    message = str(detail) if detail else f"The service answered HTTP {status}."
    if code is None and status in (400, 401, 403) and message in _BOUNDARY_REFUSALS:
        return ClientError(message, code="SESSION_REFUSED", kind="unavailable", status=status)
    code = code if isinstance(code, str) and code else f"HTTP_{status}"
    return ClientError(
        message, code=code, kind=kind_for(code, status), status=status, findings=findings
    )


def _validation_finding(item: dict) -> dict:
    location = list(item.get("loc") or [])
    if location and location[0] in ("body", "query", "path", "header"):
        location = location[1:]
    return {
        "code": str(item.get("type") or "invalid"),
        "message": str(item.get("msg") or "Invalid value."),
        "severity": "error",
        "field": ".".join(str(part) for part in location) or None,
    }


def usage_error(message: str, *, code: str = "USAGE_ERROR") -> ClientError:
    """Bad options, arguments or names: exit 2."""
    return ClientError(message, code=code, kind="invalid")
