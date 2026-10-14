from pathlib import Path

from fastapi.testclient import TestClient

from app.core.inference_schemas import (
    DatasetQualityPredictionResponse,
    InferenceHealthResponse,
    ProductionModelInformation,
)
from app.core.quality_schemas import (
    QualityLabel,
)
from app.services.prediction_logger import (
    PredictionLogger,
)
from app.main import create_application


VALID_REQUEST_BODY = {
    "row_count": 20000,
    "column_count": 18,
    "missing_value_ratio": 0.08,
    "duplicate_row_ratio": 0.03,
    "invalid_type_ratio": 0.02,
    "primary_key_uniqueness": 0.97,
    "foreign_key_match_ratio": 0.94,
    "schema_change_count": 1,
    "outlier_ratio": 0.05,
    "freshness_delay_hours": 10.0,
}


class FakeInferenceService:
    """
    Service factice utilisé pour tester
    uniquement les routes HTTP.
    """

    def artifacts_are_loaded(
        self,
    ) -> bool:
        return True

    def get_health(
        self,
    ) -> InferenceHealthResponse:
        return InferenceHealthResponse(
            status="healthy",
            model_loaded=True,
            preprocessor_loaded=True,
        )

    def get_model_information(
        self,
    ) -> ProductionModelInformation:
        return ProductionModelInformation(
            model_name="random_forest",
            model_version="1.0.0",
            status="ACTIVE",
            promoted_at=(
                "2026-10-08T12:00:00+00:00"
            ),
            evaluated_split="test",
            accuracy=0.95,
            f1_macro=0.94,
            poor_recall=0.96,
        )

    def predict(
        self,
        request_body,
    ) -> DatasetQualityPredictionResponse:
        assert request_body.row_count == 20000

        return DatasetQualityPredictionResponse(
            prediction=QualityLabel.ACCEPTABLE,
            confidence=0.87,
            probabilities={
                "ACCEPTABLE": 0.87,
                "GOOD": 0.10,
                "POOR": 0.03,
            },
            model_name="random_forest",
            model_version="1.0.0",
        )


class UnavailableInferenceService:
    """
    Service factice représentant des artefacts absents.
    """

    def artifacts_are_loaded(
        self,
    ) -> bool:
        return False

    def get_health(
        self,
    ) -> InferenceHealthResponse:
        return InferenceHealthResponse(
            status="unhealthy",
            model_loaded=False,
            preprocessor_loaded=False,
        )


def create_test_client(
    service,
    tmp_path: Path,
) -> TestClient:
    """
    Crée l'application puis remplace
    le service chargé au démarrage.
    """

    application = create_application(
        tmp_path
        / "production"
    )

    client = TestClient(
        application
    )

    client.__enter__()

    application.state.inference_service = (
        service
    )

    application.state.prediction_logger = (
        PredictionLogger(
            tmp_path / "monitoring"
        )
    )

    return client


def close_test_client(
    client: TestClient,
) -> None:
    """
    Ferme proprement le TestClient.
    """

    client.__exit__(
        None,
        None,
        None,
    )


def test_health_endpoint(
    tmp_path: Path,
) -> None:
    """
    Vérifie GET /health.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    try:
        response = client.get(
            "/api/v1/health"
        )

        assert response.status_code == 200

        assert response.json() == {
            "status": "healthy",
            "model_loaded": True,
            "preprocessor_loaded": True,
        }

    finally:
        close_test_client(
            client
        )


def test_model_endpoint(
    tmp_path: Path,
) -> None:
    """
    Vérifie GET /model.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    try:
        response = client.get(
            "/api/v1/model"
        )

        assert response.status_code == 200

        response_data = response.json()

        assert response_data[
            "model_name"
        ] == "random_forest"

        assert response_data[
            "model_version"
        ] == "1.0.0"

        assert response_data[
            "status"
        ] == "ACTIVE"

        assert response_data[
            "evaluated_split"
        ] == "test"

    finally:
        close_test_client(
            client
        )


def test_predict_endpoint(
    tmp_path: Path,
) -> None:
    """
    Vérifie POST /predict.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    try:
        response = client.post(
            "/api/v1/predict",
            json=VALID_REQUEST_BODY,
        )

        assert response.status_code == 200

        response_data = response.json()

        assert response_data[
            "prediction"
        ] == "ACCEPTABLE"

        assert response_data[
            "confidence"
        ] == 0.87

        assert response_data[
            "model_name"
        ] == "random_forest"

        assert set(
            response_data["probabilities"]
        ) == {
            "GOOD",
            "ACCEPTABLE",
            "POOR",
        }

    finally:
        close_test_client(
            client
        )


def test_invalid_request_returns_422(
    tmp_path: Path,
) -> None:
    """
    Vérifie la validation automatique FastAPI.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    invalid_request = dict(
        VALID_REQUEST_BODY
    )

    invalid_request[
        "missing_value_ratio"
    ] = 1.50

    try:
        response = client.post(
            "/api/v1/predict",
            json=invalid_request,
        )

        assert response.status_code == 422

    finally:
        close_test_client(
            client
        )


def test_unknown_request_field_returns_422(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un champ inconnu est refusé.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    invalid_request = dict(
        VALID_REQUEST_BODY
    )

    invalid_request[
        "unknown_field"
    ] = "unexpected"

    try:
        response = client.post(
            "/api/v1/predict",
            json=invalid_request,
        )

        assert response.status_code == 422

    finally:
        close_test_client(
            client
        )


def test_model_endpoint_returns_503_when_unavailable(
    tmp_path: Path,
) -> None:
    """
    Vérifie le comportement sans artefacts.
    """

    client = create_test_client(
        UnavailableInferenceService(),
        tmp_path,
    )

    try:
        response = client.get(
            "/api/v1/model"
        )

        assert response.status_code == 503

    finally:
        close_test_client(
            client
        )


def test_predict_returns_503_when_unavailable(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'aucune prédiction n'est effectuée
    sans artefacts chargés.
    """

    client = create_test_client(
        UnavailableInferenceService(),
        tmp_path,
    )

    try:
        response = client.post(
            "/api/v1/predict",
            json=VALID_REQUEST_BODY,
        )

        assert response.status_code == 503

    finally:
        close_test_client(
            client
        )

def test_successful_prediction_is_logged(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'une prédiction réussie
    est enregistrée.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    try:
        prediction_response = client.post(
            "/api/v1/predict",
            json=VALID_REQUEST_BODY,
        )

        assert prediction_response.status_code == 200

        monitoring_response = client.get(
            "/api/v1/monitoring/predictions"
        )

        assert monitoring_response.status_code == 200

        records = monitoring_response.json()

        assert len(records) == 1

        assert records[0]["success"] is True

        assert records[0]["prediction"] == (
            "ACCEPTABLE"
        )

        assert records[0]["model_name"] == (
            "random_forest"
        )

        assert records[0][
            "input_features"
        ] is not None

        assert records[0][
            "input_features"
        ][
            "row_count"
        ] == 20000

        assert records[0][
            "input_features"
        ][
            "missing_value_ratio"
        ] == 0.08

    finally:
        close_test_client(
            client
        )


def test_monitoring_summary_endpoint(
    tmp_path: Path,
) -> None:
    """
    Vérifie les statistiques de monitoring.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    try:
        first_response = client.post(
            "/api/v1/predict",
            json=VALID_REQUEST_BODY,
        )

        second_response = client.post(
            "/api/v1/predict",
            json=VALID_REQUEST_BODY,
        )

        assert first_response.status_code == 200
        assert second_response.status_code == 200

        response = client.get(
            "/api/v1/monitoring/summary"
        )

        assert response.status_code == 200

        summary = response.json()

        assert (
            summary["total_prediction_count"]
            == 2
        )

        assert (
            summary[
                "successful_prediction_count"
            ]
            == 2
        )

        assert (
            summary["failed_prediction_count"]
            == 0
        )

        assert summary["success_rate"] == 1.0

        assert summary[
            "prediction_distribution"
        ] == {
            "ACCEPTABLE": 2
        }

        assert summary["model_versions"] == {
            "random_forest:1.0.0": 2
        }

    finally:
        close_test_client(
            client
        )


def test_failed_prediction_is_logged(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un appel sans artefacts
    est enregistré comme échec.
    """

    client = create_test_client(
        UnavailableInferenceService(),
        tmp_path,
    )

    try:
        prediction_response = client.post(
            "/api/v1/predict",
            json=VALID_REQUEST_BODY,
        )

        assert prediction_response.status_code == 503

        monitoring_response = client.get(
            "/api/v1/monitoring/summary"
        )

        assert monitoring_response.status_code == 200

        summary = monitoring_response.json()

        assert (
            summary["total_prediction_count"]
            == 1
        )

        assert (
            summary["successful_prediction_count"]
            == 0
        )

        assert (
            summary["failed_prediction_count"]
            == 1
        )

        assert summary["success_rate"] == 0.0

    finally:
        close_test_client(
            client
        )


def test_predictions_endpoint_limit(
    tmp_path: Path,
) -> None:
    """
    Vérifie le paramètre limit.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    try:
        for _ in range(3):
            response = client.post(
                "/api/v1/predict",
                json=VALID_REQUEST_BODY,
            )

            assert response.status_code == 200

        monitoring_response = client.get(
            "/api/v1/monitoring/predictions",
            params={
                "limit": 2,
            },
        )

        assert monitoring_response.status_code == 200

        records = monitoring_response.json()

        assert len(records) == 2

    finally:
        close_test_client(
            client
        )


def test_invalid_monitoring_limit_returns_422(
    tmp_path: Path,
) -> None:
    """
    Vérifie la validation du paramètre limit.
    """

    client = create_test_client(
        FakeInferenceService(),
        tmp_path,
    )

    try:
        response = client.get(
            "/api/v1/monitoring/predictions",
            params={
                "limit": 0,
            },
        )

        assert response.status_code == 422

    finally:
        close_test_client(
            client
        )