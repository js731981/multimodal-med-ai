from fastapi import APIRouter, Depends

from backend.app.schemas.explain import ExplainRequest, ExplainResponse
from backend.app.services.deps import get_explanation_service
from backend.app.services.explain import ExplanationService

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    payload: ExplainRequest,
    svc: ExplanationService = Depends(get_explanation_service),
) -> ExplainResponse:
    result = await svc.explain(payload)
    return ExplainResponse(**result.model_dump())

