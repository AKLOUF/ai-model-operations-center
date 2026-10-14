from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.core.quality_schemas import (
    QualityLabel,
)


class DatasetQualityPredictionRequest(BaseModel):
    """
    Métriques de qualité d'un dataset à classifier.

    Ces dix champs correspondent aux features utilisées
    pendant l'entraînement du modèle.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    row_count: int = Field(
        ge=1,
    )

    column_count: int = Field(
        ge=1,
    )

    missing_value_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    duplicate_row_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    invalid_type_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    primary_key_uniqueness: float = Field(
        ge=0.0,
        le=1.0,
    )

    foreign_key_match_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    schema_change_count: int = Field(
        ge=0,
    )

    outlier_ratio: float = Field(
        ge=0.0,
        le=1.0,
    )

    freshness_delay_hours: float = Field(
        ge=0.0,
    )


class DatasetQualityPredictionResponse(BaseModel):
    """
    Résultat retourné par l'API d'inférence.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    prediction: QualityLabel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    probabilities: dict[
        str,
        float,
    ]

    model_name: str = Field(
        min_length=1,
    )

    model_version: str = Field(
        min_length=1,
    )


class ProductionModelInformation(BaseModel):
    """
    Informations publiques sur le modèle de production.
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

    status: str = Field(
        min_length=1,
    )

    promoted_at: str = Field(
        min_length=1,
    )

    evaluated_split: str = Field(
        min_length=1,
    )

    accuracy: float = Field(
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


class InferenceHealthResponse(BaseModel):
    """
    État du service d'inférence.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    status: str = Field(
        min_length=1,
    )

    model_loaded: bool
    preprocessor_loaded: bool