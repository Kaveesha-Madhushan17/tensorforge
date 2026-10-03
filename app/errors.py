from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None, headers=None):
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers or {}


def error_response(status, code, message, details=None, headers=None) -> JSONResponse:
    err = {"code": code, "message": message}
    if details:
        err["details"] = details
    return JSONResponse(status_code=status, content={"error": err}, headers=headers or {})
