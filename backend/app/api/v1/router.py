from fastapi import APIRouter

from backend.app.api.v1.routes.health import router as health_router
from backend.app.api.v1.routes.infer import router as infer_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router, tags=["health"])
api_v1_router.include_router(infer_router, tags=["inference"])

