from fastapi import Request
from fastapi.responses import JSONResponse

from ..ml.error_handling.ml_exceptions import MLBaseException


async def ml_exception_handler(request: Request, exc: MLBaseException) -> JSONResponse:
    """MLサービスの例外ハンドラー"""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": exc.error_code,
                "message": str(exc),
                "details": exc.details,
            }
        },
    )


def setup_error_handlers(app):
    """エラーハンドラーの設定"""
    app.add_exception_handler(MLBaseException, ml_exception_handler)
