from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import Request
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from model_trainer.api.v1.api_route import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    lifespan=lifespan,
    title="model_trainer",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


class StatusResponse(BaseModel):
    status: str

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "App healthy"}]}
    )


@app.get("/")
async def root():
    return JSONResponse(
        status_code=HTTPStatus.OK, content=[
            {
                "status": "App healthy"
            }
        ]
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, ex: HTTPException):
    return JSONResponse(
        status_code=ex.status_code,
        content={
            "detail": [
                {
                    "loc": ["string", 0],
                    "msg": ex.detail if ex.detail else "An error occurred.",
                    "type": "string"
                }
            ]
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, ex: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "loc": err["loc"],
                    "msg": err["msg"],
                    "type": err["type"]
                } for err in ex.errors()
            ]
        },
    )


app.include_router(router, prefix="/api/v1/models", tags=["Router 1"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="debug", workers=1, limit_concurrency=100)
