from pathlib import Path

from fastapi.testclient import TestClient

from app.core.alert_schemas import (
    AlertThresholdConfig,
)
from app.core.drift_schemas import (
    DriftReport,
)
from app.core.monitoring_schemas import (
    PredictionMonitoringSummary,
)
from app.main import create_application
from app.services.alert_service import (
    AlertService,
)


class FakePredictionLogger:
    """
    Journal factice retournant un résumé
    contenant plusieurs problèmes.
    """

    def build_summary(
        self,
    ) -> PredictionMonitoringSummary:
        return PredictionMonitoringSummary(
            total_prediction_count=100,
            successful_prediction_count=80,
            failed_prediction_count=20,
            success_rate=0.80,
            average_confidence=0.40,
            average_latency_seconds=1.50,
            prediction_distribution={
                "GOOD": 30,
                "ACCEPTABLE": 30,
                "POOR": 20,
            },
            model_versions={
                "random_forest:1.0.0": 80,
            },
        )


class HealthyPredictionLogger:
    """
    Journal factice retournant un résumé sain.
    """

    def build_summary(
        self,
    ) -> PredictionMonitoringSummary:
        return PredictionMonitoringSummary(
            total_prediction_count=100,
            successful_prediction_count=100,
            failed_prediction_count=0,
            success_rate=1.0,
            average_confidence=0.90,
            average_latency_seconds=0.05,
            prediction_distribution={
                "GOOD": 40,
                "ACCEPTABLE": 35,
                "POOR": 25,
            },
            model_versions={
                "random_forest:1.0.0": 100,
            },
        )


class FakeDriftDetector:
    """
    Détecteur factice retournant une dérive critique.
    """

    def analyze(self) -> DriftReport:
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
            analyzed_feature_count=10,
            drifted_feature_count=2,
            warning_feature_count=0,
            stable_feature_count=8,
            overall_status="DRIFTED",
            features=[],
        )


class StableDriftDetector:
    """
    Détecteur factice retournant un statut stable.
    """

    def analyze(self) -> DriftReport:
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
            analyzed_feature_count=10,
            drifted_feature_count=0,
            warning_feature_count=0,
            stable_feature_count=10,
            overall_status="STABLE",
            features=[],
        )


def create_alert_client(
    tmp_path: Path,
    prediction_logger,
    drift_detector,
) -> tuple[TestClient, object]:
    """
    Crée l'application avec des services factices.
    """

    alert_directory = (
        tmp_path
        / "data"
        / "monitoring"
        / "alerts"
    )

    application = create_application(
        production_directory=(
            tmp_path
            / "models"
            / "production"
        ),
        monitoring_directory=(
            tmp_path
            / "data"
            / "monitoring"
        ),
        reference_dataset_path=(
            tmp_path
            / "data"
            / "raw"
            / "dataset_quality.csv"
        ),
        drift_report_directory=(
            tmp_path
            / "reports"
            / "drift"
        ),
        alert_directory=alert_directory,
    )

    client = TestClient(
        application
    )

    client.__enter__()

    application.state.prediction_logger = (
        prediction_logger
    )

    application.state.drift_detector = (
        drift_detector
    )

    application.state.alert_service = (
        AlertService(
            AlertThresholdConfig(),
            alert_directory,
        )
    )

    return client, application


def close_alert_client(
    client: TestClient,
) -> None:
    """
    Ferme proprement le client.
    """

    client.__exit__(
        None,
        None,
        None,
    )


def test_get_alerts_returns_null_before_evaluation(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'absence de rapport initial.
    """

    client, _ = create_alert_client(
        tmp_path,
        HealthyPredictionLogger(),
        StableDriftDetector(),
    )

    try:
        response = client.get(
            "/api/v1/monitoring/alerts"
        )

        assert response.status_code == 200
        assert response.json() is None

    finally:
        close_alert_client(
            client
        )


def test_evaluate_critical_alerts(
    tmp_path: Path,
) -> None:
    """
    Vérifie la génération d'alertes critiques.
    """

    client, application = create_alert_client(
        tmp_path,
        FakePredictionLogger(),
        FakeDriftDetector(),
    )

    try:
        response = client.post(
            "/api/v1/monitoring/"
            "alerts/evaluate"
        )

        assert response.status_code == 200

        report = response.json()

        assert report[
            "overall_severity"
        ] == "CRITICAL"

        assert report["alert_count"] == 4

        assert (
            report["critical_alert_count"]
            == 4
        )

        alert_categories = {
            alert["category"]
            for alert in report["alerts"]
        }

        assert alert_categories == {
            "PREDICTION_FAILURE_RATE",
            "LOW_CONFIDENCE",
            "HIGH_LATENCY",
            "DATA_DRIFT",
        }

        alert_service = (
            application.state.alert_service
        )

        assert (
            alert_service
            .alert_report_path
            .exists()
        )

    finally:
        close_alert_client(
            client
        )


def test_get_alerts_returns_saved_report(
    tmp_path: Path,
) -> None:
    """
    Vérifie la lecture du rapport sauvegardé.
    """

    client, _ = create_alert_client(
        tmp_path,
        FakePredictionLogger(),
        FakeDriftDetector(),
    )

    try:
        evaluate_response = client.post(
            "/api/v1/monitoring/"
            "alerts/evaluate"
        )

        assert (
            evaluate_response.status_code
            == 200
        )

        get_response = client.get(
            "/api/v1/monitoring/alerts"
        )

        assert get_response.status_code == 200

        report = get_response.json()

        assert report[
            "overall_severity"
        ] == "CRITICAL"

        assert report["alert_count"] == 4

    finally:
        close_alert_client(
            client
        )


def test_healthy_system_creates_empty_alert_report(
    tmp_path: Path,
) -> None:
    """
    Vérifie un système sans alerte active.
    """

    client, _ = create_alert_client(
        tmp_path,
        HealthyPredictionLogger(),
        StableDriftDetector(),
    )

    try:
        response = client.post(
            "/api/v1/monitoring/"
            "alerts/evaluate"
        )

        assert response.status_code == 200

        report = response.json()

        assert report["alert_count"] == 0
        assert report["alerts"] == []

        assert report[
            "overall_severity"
        ] == "INFO"

    finally:
        close_alert_client(
            client
        )