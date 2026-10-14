from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.core.quality_schemas import (
    QualityLabel,
)
from pathlib import Path

class PredictionLogRecord(BaseModel):
    """
    Événement enregistré après une tentative
    de prédiction.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    prediction_id: str = Field(
        min_length=1,
    )

    timestamp: str = Field(
        min_length=1,
    )

    prediction: QualityLabel | None = None

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    model_name: str | None = None
    model_version: str | None = None

    input_features: dict[
        str,
        float | int,
    ] | None = None

    latency_seconds: float = Field(
        ge=0.0,
    )

    success: bool

    error_message: str | None = None


class PredictionMonitoringSummary(BaseModel):
    """
    Résumé des prédictions enregistrées.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    total_prediction_count: int = Field(
        ge=0,
    )

    successful_prediction_count: int = Field(
        ge=0,
    )

    failed_prediction_count: int = Field(
        ge=0,
    )

    success_rate: float = Field(
        ge=0.0,
        le=1.0,
    )

    average_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    average_latency_seconds: float | None = Field(
        default=None,
        ge=0.0,
    )

    prediction_distribution: dict[str, int]

    model_versions: dict[str, int]

class DriftMonitoringConfig(BaseModel):
    """
    Configuration du détecteur de dérive.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    reference_dataset_path: Path

    minimum_production_records: int = Field(
        default=30,
        ge=1,
    )

    warning_threshold: float = Field(
        default=0.10,
        ge=0.0,
    )

    drift_threshold: float = Field(
        default=0.25,
        gt=0.0,
    )

    bin_count: int = Field(
        default=10,
        ge=2,
    )

    report_directory: Path


class AlertMonitoringConfig(BaseModel):
    """
    Configuration des seuils d'alertes.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    warning_failure_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )

    critical_failure_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
    )

    warning_minimum_confidence: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )

    critical_minimum_confidence: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
    )

    warning_latency_seconds: float = Field(
        default=0.20,
        ge=0.0,
    )

    critical_latency_seconds: float = Field(
        default=1.00,
        ge=0.0,
    )

    alert_directory: Path


class MonitoringConfiguration(BaseModel):
    """
    Configuration complète du monitoring.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    prediction_log_directory: Path

    drift: DriftMonitoringConfig
    alerts: AlertMonitoringConfig