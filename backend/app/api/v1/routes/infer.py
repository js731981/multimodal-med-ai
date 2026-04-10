from fastapi import APIRouter, HTTPException

from backend.app.api.deps import get_pipeline, to_pipeline_input
from backend.app.api.schemas.infer import InferRequest, InferResponse
from backend.app.inference.errors import InvalidImageInputError

router = APIRouter()


@router.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest) -> InferResponse:
    try:
        pipeline_input = to_pipeline_input(payload.text, payload.image_b64)
    except InvalidImageInputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    pipeline = get_pipeline()
    try:
        result = pipeline.run(pipeline_input)
    except InvalidImageInputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return InferResponse(
        disease=result.disease,
        confidence=result.confidence,
        explanation=result.explanation,
    )

