from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    text: str | None = Field(default=None, description="Clinical note / query text")
    image_b64: str | None = Field(
        default=None, description="Base64-encoded image bytes (optional)"
    )
    top_k: int = Field(default=5, ge=1, le=20)


class InferResponse(BaseModel):
    disease: str
    confidence: float
    explanation: str = Field(
        default="",
        description="RAG-backed reasoning text (empty if RAG is disabled or failed).",
    )
