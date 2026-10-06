import hmac
import logging
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.errors import ServiceError
from app.schemas import (
    ErrorResponse,
    ExtractFormRequest,
    ExtractFormResponse,
    LiveHealthResponse,
    ReadyHealthResponse,
)
from app.service import ExtractionService


logger = logging.getLogger("api_form_extraction")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def create_app(
    settings: Settings | None = None,
    extraction_service: ExtractionService | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, resolved_settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(
        title="Form Extraction API",
        version="1.0.0",
        description="Trích xuất nhiều field từ transcript theo schema biểu mẫu.",
    )
    app.state.settings = resolved_settings
    app.state.extraction_service = extraction_service or ExtractionService(resolved_settings)

    allowed_origins = resolved_settings.cors_allowed_origins_list
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request_completed request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": _request_id(request),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request payload is invalid",
                    "request_id": _request_id(request),
                },
                "details": jsonable_encoder(exc.errors()),
            },
        )

    async def verify_internal_api_key(
        request: Request,
        x_internal_api_key: str | None = Header(default=None, alias="X-Internal-API-Key"),
    ) -> None:
        if request.app.state.settings.allow_public_extraction:
            return
        expected = request.app.state.settings.internal_api_key_value
        if expected is None and request.app.state.settings.app_env != "production":
            return
        if x_internal_api_key is None or not hmac.compare_digest(
            x_internal_api_key.encode("utf-8"),
            expected.encode("utf-8"),
        ):
            raise ServiceError(401, "unauthorized", "Invalid internal API key")

    @app.get("/health/live", response_model=LiveHealthResponse)
    async def live() -> LiveHealthResponse:
        return LiveHealthResponse()

    @app.get(
        "/health/ready",
        response_model=ReadyHealthResponse,
        responses={503: {"model": ReadyHealthResponse}},
    )
    async def ready(request: Request):
        service: ExtractionService = request.app.state.extraction_service
        configured = service.is_configured
        payload = ReadyHealthResponse(
            status="ready" if configured else "not_ready",
            openai_configured=configured,
            model=request.app.state.settings.openai_model,
        )
        if not configured:
            return JSONResponse(status_code=503, content=payload.model_dump())
        return payload

    @app.post(
        "/api/v1/openai/extract-form",
        response_model=ExtractFormResponse,
        responses={
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
        dependencies=[Depends(verify_internal_api_key)],
    )
    async def extract_form(payload: ExtractFormRequest, request: Request) -> ExtractFormResponse:
        service: ExtractionService = request.app.state.extraction_service
        reference_date = payload.reference_date or datetime.now(
            ZoneInfo(request.app.state.settings.app_timezone)
        ).date()
        result = await service.extract(payload, _request_id(request), reference_date)
        logger.info(
            "extraction_completed request_id=%s requested_fields=%s extracted_fields=%s ambiguous_fields=%s total_tokens=%s latency_ms=%s",
            result.request_id,
            len(payload.fields),
            sum(item.status == "extracted" for item in result.extractions),
            sum(item.status == "ambiguous" for item in result.extractions),
            result.usage.total_tokens,
            result.usage.latency_ms,
        )
        return result

    return app


app = create_app()
