from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.drift_schemas import (
    DriftReport,
    FeatureDriftResult,
)
from app.main import create_application


class FakeDriftDetector:
    """
    Détecteur factice retournant un rapport stable.
    """

    def analyze(self) -> DriftReport:
        """
        Retourne un rapport de dérive stable.
        """

        feature_result = FeatureDriftResult(
            feature_name="missing_value_ratio",
            reference_count=3000,
            production_count=100,
            reference_mean=0.15,
            production_mean=0.16,
            reference_standard_deviation=0.10,
            normalized_mean_difference=0.10,
            psi=0.05,
            status="STABLE",
        )

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
            analyzed_feature_count=1,
            drifted_feature_count=0,
            warning_feature_count=0,
            stable_feature_count=1,
            overall_status="STABLE",
            features=[
                feature_result
            ],
        )


class InsufficientDataDriftDetector:
    """
    Détecteur factice retournant un rapport
    avec un nombre insuffisant de prédictions.
    """

    def analyze(self) -> DriftReport:
        """
        Retourne un rapport INSUFFICIENT_DATA.
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
            production_record_count=5,
            analyzed_feature_count=0,
            drifted_feature_count=0,
            warning_feature_count=0,
            stable_feature_count=0,
            overall_status="INSUFFICIENT_DATA",
            features=[],
        )


class FailingDriftDetector:
    """
    Détecteur factice simulant une erreur
    pendant l'analyse.
    """

    def analyze(self) -> DriftReport:
        """
        Déclenche une erreur de données.
        """

        raise ValueError(
            "Journal de prédictions invalide."
        )


class MissingReferenceDriftDetector:
    """
    Détecteur factice simulant l'absence
    du dataset de référence.
    """

    def analyze(self) -> DriftReport:
        """
        Déclenche une erreur de fichier absent.
        """

        raise FileNotFoundError(
            "Dataset de référence introuvable."
        )


def create_drift_application(
    tmp_path: Path,
    drift_detector: Any,
) -> FastAPI:
    """
    Crée une application configurée pour les tests.
    """

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
    )

    application.state.test_drift_detector = (
        drift_detector
    )

    return application


def create_drift_client(
    tmp_path: Path,
    drift_detector: Any,
) -> tuple[TestClient, FastAPI]:
    """
    Démarre l'application de test puis remplace
    le détecteur réel par le détecteur factice.
    """

    application = create_drift_application(
        tmp_path,
        drift_detector,
    )

    client = TestClient(
        application
    )

    client.__enter__()

    application.state.drift_detector = (
        drift_detector
    )

    return client, application


def close_drift_client(
    client: TestClient,
) -> None:
    """
    Ferme proprement le client de test.
    """

    client.__exit__(
        None,
        None,
        None,
    )


def test_get_stable_drift_report(
    tmp_path: Path,
) -> None:
    """
    Vérifie GET /api/v1/monitoring/drift
    avec un rapport stable.
    """

    client, _ = create_drift_client(
        tmp_path,
        FakeDriftDetector(),
    )

    try:
        response = client.get(
            "/api/v1/monitoring/drift"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert (
            response_data["overall_status"]
            == "STABLE"
        )

        assert (
            response_data[
                "production_record_count"
            ]
            == 100
        )

        assert (
            response_data[
                "analyzed_feature_count"
            ]
            == 1
        )

        assert (
            response_data[
                "drifted_feature_count"
            ]
            == 0
        )

        assert (
            response_data[
                "warning_feature_count"
            ]
            == 0
        )

        assert (
            response_data[
                "stable_feature_count"
            ]
            == 1
        )

        assert len(
            response_data["features"]
        ) == 1

        feature_result = (
            response_data["features"][0]
        )

        assert (
            feature_result["feature_name"]
            == "missing_value_ratio"
        )

        assert (
            feature_result["status"]
            == "STABLE"
        )

        assert feature_result["psi"] == 0.05

    finally:
        close_drift_client(
            client
        )


def test_get_insufficient_data_report(
    tmp_path: Path,
) -> None:
    """
    Vérifie le rapport lorsque le nombre
    de prédictions est insuffisant.
    """

    client, _ = create_drift_client(
        tmp_path,
        InsufficientDataDriftDetector(),
    )

    try:
        response = client.get(
            "/api/v1/monitoring/drift"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert (
            response_data["overall_status"]
            == "INSUFFICIENT_DATA"
        )

        assert (
            response_data[
                "production_record_count"
            ]
            == 5
        )

        assert (
            response_data[
                "minimum_production_records"
            ]
            == 30
        )

        assert (
            response_data[
                "analyzed_feature_count"
            ]
            == 0
        )

        assert response_data["features"] == []

    finally:
        close_drift_client(
            client
        )


def test_generate_and_save_drift_report(
    tmp_path: Path,
) -> None:
    """
    Vérifie POST /api/v1/monitoring/drift/report
    et la sauvegarde du rapport JSON.
    """

    client, application = create_drift_client(
        tmp_path,
        FakeDriftDetector(),
    )

    try:
        response = client.post(
            "/api/v1/monitoring/drift/report"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert (
            response_data["overall_status"]
            == "STABLE"
        )

        report_directory = Path(
            application.state
            .drift_report_directory
        )

        report_path = (
            report_directory
            / "drift_report.json"
        )

        assert report_path.exists()
        assert report_path.is_file()

        saved_report = DriftReport.model_validate_json(
            report_path.read_text(
                encoding="utf-8"
            )
        )

        assert saved_report.overall_status == (
            "STABLE"
        )

        assert (
            saved_report.production_record_count
            == 100
        )

        assert len(
            saved_report.features
        ) == 1

        assert (
            saved_report
            .features[0]
            .feature_name
            == "missing_value_ratio"
        )

    finally:
        close_drift_client(
            client
        )


def test_generate_report_creates_directory(
    tmp_path: Path,
) -> None:
    """
    Vérifie que le dossier du rapport
    est créé automatiquement.
    """

    client, application = create_drift_client(
        tmp_path,
        FakeDriftDetector(),
    )

    report_directory = Path(
        application.state
        .drift_report_directory
    )

    assert not report_directory.exists()

    try:
        response = client.post(
            "/api/v1/monitoring/drift/report"
        )

        assert response.status_code == 200

        assert report_directory.exists()

        assert (
            report_directory
            / "drift_report.json"
        ).exists()

    finally:
        close_drift_client(
            client
        )


def test_get_drift_returns_500_for_invalid_data(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'une erreur de données retourne
    une réponse HTTP 500.
    """

    client, _ = create_drift_client(
        tmp_path,
        FailingDriftDetector(),
    )

    try:
        response = client.get(
            "/api/v1/monitoring/drift"
        )

        assert response.status_code == 500

        assert response.json() == {
            "detail": (
                "Journal de prédictions invalide."
            )
        }

    finally:
        close_drift_client(
            client
        )


def test_generate_report_returns_500_for_invalid_data(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'aucun rapport n'est sauvegardé
    lorsque l'analyse échoue.
    """

    client, application = create_drift_client(
        tmp_path,
        FailingDriftDetector(),
    )

    try:
        response = client.post(
            "/api/v1/monitoring/drift/report"
        )

        assert response.status_code == 500

        assert response.json() == {
            "detail": (
                "Journal de prédictions invalide."
            )
        }

        report_path = (
            Path(
                application.state
                .drift_report_directory
            )
            / "drift_report.json"
        )

        assert not report_path.exists()

    finally:
        close_drift_client(
            client
        )


def test_get_drift_returns_503_when_reference_is_missing(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un dataset de référence absent
    retourne une réponse HTTP 503.
    """

    client, _ = create_drift_client(
        tmp_path,
        MissingReferenceDriftDetector(),
    )

    try:
        response = client.get(
            "/api/v1/monitoring/drift"
        )

        assert response.status_code == 503

        assert response.json() == {
            "detail": (
                "Dataset de référence introuvable."
            )
        }

    finally:
        close_drift_client(
            client
        )


def test_generate_report_returns_503_when_reference_is_missing(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un rapport ne peut pas être généré
    sans dataset de référence.
    """

    client, application = create_drift_client(
        tmp_path,
        MissingReferenceDriftDetector(),
    )

    try:
        response = client.post(
            "/api/v1/monitoring/drift/report"
        )

        assert response.status_code == 503

        report_path = (
            Path(
                application.state
                .drift_report_directory
            )
            / "drift_report.json"
        )

        assert not report_path.exists()

    finally:
        close_drift_client(
            client
        )