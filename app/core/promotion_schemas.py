from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


ProductionModelStatus = Literal[
    "ACTIVE",
    "ARCHIVED",
]


class PromotionTestDataConfig(BaseModel):
    """
    Configuration du jeu de test final.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    features_file: Path
    target_file: Path

    target_column: str = Field(
        default="quality_label",
        min_length=1,
    )


class PromotionSourceConfig(BaseModel):
    """
    Emplacements des artefacts sources.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    candidates_directory: Path
    preprocessor_file: Path


class PromotionEvaluationConfig(BaseModel):
    """
    Seuils obligatoires pour la production.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    minimum_f1_macro: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )

    minimum_poor_recall: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
    )


class PromotionOutputConfig(BaseModel):
    """
    Emplacements des artefacts de production.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    production_directory: Path
    evaluation_directory: Path


class PromotionConfig(BaseModel):
    """
    Configuration complète du pipeline de promotion.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    champion_file: Path
    test_data: PromotionTestDataConfig
    source: PromotionSourceConfig
    evaluation: PromotionEvaluationConfig
    output: PromotionOutputConfig


class ChampionDescriptor(BaseModel):
    """
    Informations lues depuis champion.json.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    champion_selected: bool

    model_name: str | None = None
    model_version: str | None = None

    f1_macro: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    poor_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    prediction_latency_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    training_duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    reason: str | None = None


class ProductionModelRecord(BaseModel):
    """
    Enregistrement du modèle actif en production.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )

    status: ProductionModelStatus

    promoted_at: str = Field(
        min_length=1,
    )

    evaluated_split: str = Field(
        default="test",
        min_length=1,
    )

    model_path: str = Field(
        min_length=1,
    )

    preprocessor_path: str = Field(
        min_length=1,
    )

    metadata_path: str = Field(
        min_length=1,
    )

    test_metrics_path: str = Field(
        min_length=1,
    )

    confusion_matrix_path: str = Field(
        min_length=1,
    )

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    precision_macro: float = Field(
        ge=0.0,
        le=1.0,
    )

    recall_macro: float = Field(
        ge=0.0,
        le=1.0,
    )

    f1_macro: float = Field(
        ge=0.0,
        le=1.0,
    )

    poor_recall: float = Field(
        ge=0.0,
        le=1.0,
    )