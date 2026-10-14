from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


AlertSeverity = Literal[
    "INFO",
    "WARNING",
    "CRITICAL",
]

AlertCategory = Literal[
    "PREDICTION_FAILURE_RATE",
    "LOW_CONFIDENCE",
    "HIGH_LATENCY",
    "DATA_DRIFT",
]

AlertStatus = Literal[
    "OPEN",
    "RESOLVED",
]


class AlertThresholdConfig(BaseModel):
    """
    Seuils utilisés par le moteur d'alertes.
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


class MonitoringAlert(BaseModel):
    """
    Alerte produite par le moteur de monitoring.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    alert_id: str = Field(
        min_length=1,
    )

    created_at: str = Field(
        min_length=1,
    )

    severity: AlertSeverity
    category: AlertCategory
    status: AlertStatus = "OPEN"

    title: str = Field(
        min_length=1,
    )

    message: str = Field(
        min_length=1,
    )

    observed_value: float | str

    threshold_value: float | str | None = None

    model_name: str | None = None
    model_version: str | None = None


class AlertEvaluationReport(BaseModel):
    """
    Rapport global produit par le moteur d'alertes.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    generated_at: str = Field(
        min_length=1,
    )

    overall_severity: AlertSeverity

    alert_count: int = Field(
        ge=0,
    )

    critical_alert_count: int = Field(
        ge=0,
    )

    warning_alert_count: int = Field(
        ge=0,
    )

    info_alert_count: int = Field(
        ge=0,
    )

    alerts: list[MonitoringAlert]