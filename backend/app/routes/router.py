from fastapi import APIRouter

from backend.app.routes.health import router as health_router
from backend.app.routes.predict import router as predict_router
from backend.app.routes.explain import router as explain_router

router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(predict_router, tags=["inference"])
router.include_router(explain_router, tags=["explainability"])

