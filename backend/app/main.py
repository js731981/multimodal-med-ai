from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import api_v1_router
from backend.app.config import get_settings
from backend.app.logging import init_logging


def create_app() -> FastAPI:
    settings = get_settings()
    init_logging(level=settings.log_level, json_logs=settings.log_json)

    app = FastAPI(title=settings.app_name, version="0.1.0")

    # Safe defaults; tighten in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_prod else [],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_v1_router, prefix="/api/v1")
    return app


app = create_app()

