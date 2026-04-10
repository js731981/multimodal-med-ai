from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.app.schemas.predict import PredictResponse
from backend.app.services.deps import get_prediction_service
from backend.app.services.prediction import PredictionService

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    symptoms: str = Form(..., description="Symptoms / clinical text input"),
    image: UploadFile | None = File(default=None, description="Optional image file upload"),
    svc: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    image_bytes: bytes | None = None
    if image is not None:
        image_bytes = await image.read()
    result = await svc.predict(symptoms=symptoms, image_bytes=image_bytes, image_filename=image.filename if image else None)
    return PredictResponse(**result.model_dump())

