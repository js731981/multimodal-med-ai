from fastapi import FastAPI

from backend.app.api.v1.router import api_v1_router
from backend.app.config import get_settings
from backend.app.logging import init_logging


def create_app() -> FastAPI:
    settings = get_settings()
    init_logging(level=settings.log_level, json_logs=settings.log_json)

    app = FastAPI(title=settings.app_name)
    app.include_router(api_v1_router, prefix="/api/v1")
    return app

