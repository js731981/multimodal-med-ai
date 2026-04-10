from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    - Loads from process environment variables
    - Also supports local development via `.env`
    - Keeps `extra="ignore"` so adding env vars won't break startup
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Environment / app
    app_env: Literal["local", "dev", "prod", "test"] = Field(
        default="local",
        validation_alias=AliasChoices("APP_ENV", "ENV", "ENVIRONMENT"),
    )
    app_name: str = Field(default="multimodal-med-ai", validation_alias=AliasChoices("APP_NAME"))

    # HTTP server (useful for uvicorn/gunicorn launchers)
    host: str = Field(default="0.0.0.0", validation_alias=AliasChoices("HOST"))
    port: int = Field(default=8000, validation_alias=AliasChoices("PORT"))

    # Logging
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    log_json: bool = Field(default=True, validation_alias=AliasChoices("LOG_JSON"))

    # External services
    qdrant_url: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_URL"))
    qdrant_collection: str | None = Field(default=None, validation_alias=AliasChoices("QDRANT_COLLECTION"))

    # Inference configuration (backwards compatible with `MMEDAI_` env prefix)
    image_model_path: str = Field(
        default="backend.app.inference.image_model_impl:ImageModelImpl",
        validation_alias=AliasChoices("MMEDAI_IMAGE_MODEL_PATH", "IMAGE_MODEL_PATH"),
        description="Import path for the image embedding model class/factory.",
    )
    text_model_path: str = Field(
        default="backend.app.inference.text_model_impl:TextModelImpl",
        validation_alias=AliasChoices("MMEDAI_TEXT_MODEL_PATH", "TEXT_MODEL_PATH"),
        description="Import path for the text embedding model class/factory.",
    )
    fusion_model_path: str = Field(
        default="backend.app.inference.fusion_model_impl:FusionModelImpl",
        validation_alias=AliasChoices("MMEDAI_FUSION_MODEL_PATH", "FUSION_MODEL_PATH"),
        description="Import path for the fusion classifier class/factory.",
    )

    image_checkpoint_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("MMEDAI_IMAGE_CHECKPOINT_PATH", "IMAGE_CHECKPOINT_PATH"),
        description=(
            "Optional path to the ResNet CXR checkpoint (.pth). Absolute or relative to the "
            "project root. When unset, ImageModelImpl tries models/image_model/resnet_model.pth "
            "then models/image_model/dummy_smoke.pth."
        ),
    )

    image_model_kwargs: dict[str, Any] = Field(
        default_factory=dict, validation_alias=AliasChoices("MMEDAI_IMAGE_MODEL_KWARGS")
    )
    text_model_kwargs: dict[str, Any] = Field(
        default_factory=dict, validation_alias=AliasChoices("MMEDAI_TEXT_MODEL_KWARGS")
    )
    fusion_model_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {
            "image_feature_dim": 2048,
            "text_feature_dim": 256,
        },
        validation_alias=AliasChoices("MMEDAI_FUSION_MODEL_KWARGS"),
        description="Fusion MLP kwargs: align image_feature_dim / text_feature_dim with encoders (e.g. ResNet50 2048, TF-IDF 256).",
    )

    labels: list[str] = Field(
        default_factory=lambda: ["normal", "pneumonia"],
        validation_alias=AliasChoices("MMEDAI_LABELS"),
        description=(
            "Fusion / API class names; order must match logits column order and the image "
            "checkpoint `classes` (index 0 = normal, 1 = pneumonia for binary CXR)."
        ),
    )

    rag_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("MMEDAI_RAG_ENABLED", "RAG_ENABLED"),
        description="When true, attach default medical RAG for post-fusion explanations.",
    )
    rag_top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        validation_alias=AliasChoices("MMEDAI_RAG_TOP_K", "RAG_TOP_K"),
        description="Top-k knowledge passages for RAG retrieval.",
    )

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()

