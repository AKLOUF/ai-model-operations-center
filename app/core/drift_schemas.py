from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


DriftStatus = Literal[
    "STABLE",
    "WARNING",
    "DRIFTED",
    "INSUFFICIENT_DATA",
]


class FeatureDriftResult(BaseModel):
    """
    Résultat de dérive calculé pour une feature.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    feature_name: str = Field(
        min_length=1,
    )

    reference_count: int = Field(
        ge=0,
    )

    production_count: int = Field(
        ge=0,
    )

    reference_mean: float
    production_mean: float

    reference_standard_deviation: float = Field(
        ge=0.0,
    )

    normalized_mean_difference: float = Field(
        ge=0.0,
    )

    psi: float = Field(
        ge=0.0,
    )

    status: DriftStatus


class DriftReport(BaseModel):
    """
    Rapport global de dérive des données.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    generated_at: str = Field(
        min_length=1,
    )

    reference_dataset_path: str = Field(
        min_length=1,
    )

    prediction_log_path: str = Field(
        min_length=1,
    )

    minimum_production_records: int = Field(
        ge=1,
    )

    production_record_count: int = Field(
        ge=0,
    )

    analyzed_feature_count: int = Field(
        ge=0,
    )

    drifted_feature_count: int = Field(
        ge=0,
    )

    warning_feature_count: int = Field(
        ge=0,
    )

    stable_feature_count: int = Field(
        ge=0,
    )

    overall_status: DriftStatus

    features: list[
        FeatureDriftResult
    ]