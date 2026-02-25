from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


class PredictError(Exception):
    pass


class NotReadyError(Exception):
    pass


def http_400(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)


def http_503(msg: str) -> HTTPException:
    return HTTPException(status_code=503, detail=msg)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(PredictError)
    async def _predict_error_handler(_: Request, exc: PredictError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(NotReadyError)
    async def _not_ready_handler(_: Request, exc: NotReadyError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})