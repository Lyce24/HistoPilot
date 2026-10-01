"""The service's coded JSON answers: a refusal, and a coded error as its handlers give it."""

from starlette.responses import JSONResponse

FINDING_FIELDS = ("code", "message", "severity", "field")


def refusal(message: str, code: str, status: int = 403) -> JSONResponse:
    return JSONResponse({"detail": message, "code": code}, status_code=status)


def coded_response(error) -> JSONResponse:
    """A StorageError, WorkspaceError or FilesystemError: its message, code, findings, status."""
    body = {"detail": str(error), "code": error.code}
    findings = getattr(error, "findings", None)
    if findings:
        body["findings"] = [
            {key: item[key] for key in FINDING_FIELDS if key in item} for item in findings
        ]
    return JSONResponse(body, status_code=error.status_code)
