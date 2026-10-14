from pathlib import Path

import pytest

from app.core.alert_schemas import (
    AlertThresholdConfig,
)
from app.core.drift_schemas import (
    DriftReport,
)
from app.core.monitoring_schemas import (
    PredictionMonitoringSummary,
)
from app.services.alert_service import (
    AlertService,
    determine_overall_severity,
)


def create_monitoring_summary(
    total_count: int = 100,
    failed_count: int = 0,
    average_confidence: float | None = 0.90,
    average_latency: float | None = 0.05,
) -> PredictionMonitoringSummary:
    """
    Crée un résumé de monitoring.
    """

    successful_count = (
        total_count
        - failed_count
    )

    if total_count == 0:
        success_rate = 0.0
    else:
        success_rate = (
            successful_count
            / total_count
        )

    return PredictionMonitoringSummary(
        total_prediction_count=total_count,
        successful_prediction_count=(
            successful_count
        ),
        failed_prediction_count=(
            failed_count
        ),
        success_rate=success_rate,
        average_confidence=(
            average_confidence
        ),
        average_latency_seconds=(
            average_latency
        ),
        prediction_distribution={
            "GOOD": successful_count
        },
        model_versions={
            "random_forest:1.0.0": (
                successful_count
            )
        },
    )


def create_drift_report(
    overall_status: str = "STABLE",
) -> DriftReport:
    """
    Crée un rapport de dérive minimal.
    """

    return DriftReport(
        generated_at=(
            "2026-10-08T12:00:00+00:00"
        ),
        reference_dataset_path=(
            "data/raw/dataset_quality.csv"
        ),
        prediction_log_path=(
            "data/monitoring/predictions.jsonl"
        ),
        minimum_production_records=30,
        production_record_count=100,
        analyzed_feature_count=0,
        drifted_feature_count=0,
        warning_feature_count=0,
        stable_feature_count=0,
        overall_status=overall_status,
        features=[],
    )


def create_alert_service(
    tmp_path: Path,
) -> AlertService:
    """
    Crée un service avec les seuils par défaut.
    """

    return AlertService(
        thresholds=AlertThresholdConfig(),
        alert_directory=tmp_path,
    )


def test_no_alert_for_healthy_service(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    report = service.evaluate(
        create_monitoring_summary(),
        create_drift_report(),
    )

    assert report.alert_count == 0
    assert report.alerts == []

    assert report.overall_severity == (
        "INFO"
    )


def test_warning_failure_rate(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_failure_rate(
        create_monitoring_summary(
            total_count=100,
            failed_count=10,
        )
    )

    assert alert is not None
    assert alert.severity == "WARNING"

    assert alert.category == (
        "PREDICTION_FAILURE_RATE"
    )


def test_critical_failure_rate(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_failure_rate(
        create_monitoring_summary(
            total_count=100,
            failed_count=20,
        )
    )

    assert alert is not None
    assert alert.severity == "CRITICAL"


def test_warning_low_confidence(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_confidence(
        create_monitoring_summary(
            average_confidence=0.65
        )
    )

    assert alert is not None
    assert alert.severity == "WARNING"

    assert alert.category == (
        "LOW_CONFIDENCE"
    )


def test_critical_low_confidence(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_confidence(
        create_monitoring_summary(
            average_confidence=0.40
        )
    )

    assert alert is not None
    assert alert.severity == "CRITICAL"


def test_warning_latency(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_latency(
        create_monitoring_summary(
            average_latency=0.50
        )
    )

    assert alert is not None
    assert alert.severity == "WARNING"

    assert alert.category == (
        "HIGH_LATENCY"
    )


def test_critical_latency(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_latency(
        create_monitoring_summary(
            average_latency=1.50
        )
    )

    assert alert is not None
    assert alert.severity == "CRITICAL"


def test_warning_drift(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_drift(
        create_drift_report(
            overall_status="WARNING"
        )
    )

    assert alert is not None
    assert alert.severity == "WARNING"


def test_critical_drift(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_drift(
        create_drift_report(
            overall_status="DRIFTED"
        )
    )

    assert alert is not None
    assert alert.severity == "CRITICAL"

    assert alert.category == "DATA_DRIFT"


def test_insufficient_data_does_not_create_alert(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    alert = service.evaluate_drift(
        create_drift_report(
            overall_status=(
                "INSUFFICIENT_DATA"
            )
        )
    )

    assert alert is None


def test_multiple_alerts(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    report = service.evaluate(
        create_monitoring_summary(
            total_count=100,
            failed_count=20,
            average_confidence=0.40,
            average_latency=1.50,
        ),
        create_drift_report(
            overall_status="DRIFTED"
        ),
    )

    assert report.alert_count == 4

    assert report.critical_alert_count == 4

    assert report.overall_severity == (
        "CRITICAL"
    )


def test_save_and_load_report(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    report = service.evaluate(
        create_monitoring_summary(
            average_confidence=0.65
        ),
        create_drift_report(),
    )

    report_path = service.save_report(
        report
    )

    assert report_path.exists()

    loaded_report = service.load_report()

    assert loaded_report == report


def test_missing_report_returns_none(
    tmp_path: Path,
) -> None:
    service = create_alert_service(
        tmp_path
    )

    assert service.load_report() is None


def test_determine_overall_severity() -> None:
    service = create_alert_service(
        Path(".")
    )

    warning_alert = service.evaluate_confidence(
        create_monitoring_summary(
            average_confidence=0.65
        )
    )

    critical_alert = service.evaluate_latency(
        create_monitoring_summary(
            average_latency=1.50
        )
    )

    assert warning_alert is not None
    assert critical_alert is not None

    assert determine_overall_severity(
        [
            warning_alert,
            critical_alert,
        ]
    ) == "CRITICAL"