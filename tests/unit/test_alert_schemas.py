import pytest
from pydantic import ValidationError

from app.core.alert_schemas import (
    AlertEvaluationReport,
    AlertThresholdConfig,
    MonitoringAlert,
)


def create_alert(
    severity: str = "WARNING",
) -> MonitoringAlert:
    """
    Crée une alerte valide.
    """

    return MonitoringAlert(
        alert_id="alert_test_001",
        created_at=(
            "2026-10-08T12:00:00+00:00"
        ),
        severity=severity,
        category="LOW_CONFIDENCE",
        status="OPEN",
        title=(
            "Confiance moyenne insuffisante"
        ),
        message=(
            "La confiance est sous le seuil."
        ),
        observed_value=0.65,
        threshold_value=0.70,
        model_name="random_forest",
        model_version="1.0.0",
    )


def test_default_threshold_config() -> None:
    config = AlertThresholdConfig()

    assert config.warning_failure_rate == (
        pytest.approx(0.05)
    )

    assert config.critical_failure_rate == (
        pytest.approx(0.15)
    )

    assert config.warning_minimum_confidence == (
        pytest.approx(0.70)
    )

    assert config.critical_latency_seconds == (
        pytest.approx(1.00)
    )


def test_invalid_failure_rate_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AlertThresholdConfig(
            warning_failure_rate=1.20
        )


def test_valid_alert() -> None:
    alert = create_alert()

    assert alert.severity == "WARNING"
    assert alert.status == "OPEN"

    assert alert.category == (
        "LOW_CONFIDENCE"
    )


def test_invalid_alert_severity_is_rejected() -> None:
    data = create_alert().model_dump()

    data["severity"] = "UNKNOWN"

    with pytest.raises(ValidationError):
        MonitoringAlert.model_validate(
            data
        )


def test_valid_alert_report() -> None:
    alert = create_alert()

    report = AlertEvaluationReport(
        generated_at=(
            "2026-10-08T12:00:00+00:00"
        ),
        overall_severity="WARNING",
        alert_count=1,
        critical_alert_count=0,
        warning_alert_count=1,
        info_alert_count=0,
        alerts=[
            alert
        ],
    )

    assert report.alert_count == 1

    assert report.overall_severity == (
        "WARNING"
    )