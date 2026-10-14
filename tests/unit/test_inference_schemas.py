import pytest
from pydantic import ValidationError

from app.core.inference_schemas import (
    DatasetQualityPredictionRequest,
    DatasetQualityPredictionResponse,
    InferenceHealthResponse,
    ProductionModelInformation,
)
from app.core.quality_schemas import (
    QualityLabel,
)


def create_valid_request(
) -> DatasetQualityPredictionRequest:
    """
    Crée une requête d'inférence valide.
    """

    return DatasetQualityPredictionRequest(
        row_count=20_000,
        column_count=18,
        missing_value_ratio=0.08,
        duplicate_row_ratio=0.03,
        invalid_type_ratio=0.02,
        primary_key_uniqueness=0.97,
        foreign_key_match_ratio=0.94,
        schema_change_count=1,
        outlier_ratio=0.05,
        freshness_delay_hours=10.0,
    )


def test_valid_prediction_request() -> None:
    """
    Vérifie une requête d'inférence valide.
    """

    request = create_valid_request()

    assert request.row_count == 20_000
    assert request.column_count == 18

    assert request.missing_value_ratio == (
        pytest.approx(0.08)
    )

    assert request.primary_key_uniqueness == (
        pytest.approx(0.97)
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "missing_value_ratio",
        "duplicate_row_ratio",
        "invalid_type_ratio",
        "primary_key_uniqueness",
        "foreign_key_match_ratio",
        "outlier_ratio",
    ],
)
def test_ratio_above_one_is_rejected(
    field_name: str,
) -> None:
    """
    Vérifie qu'un ratio supérieur à un est refusé.
    """

    request_data = (
        create_valid_request()
        .model_dump()
    )

    request_data[field_name] = 1.10

    with pytest.raises(ValidationError):
        DatasetQualityPredictionRequest.model_validate(
            request_data
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "missing_value_ratio",
        "duplicate_row_ratio",
        "invalid_type_ratio",
        "primary_key_uniqueness",
        "foreign_key_match_ratio",
        "outlier_ratio",
    ],
)
def test_negative_ratio_is_rejected(
    field_name: str,
) -> None:
    """
    Vérifie qu'un ratio négatif est refusé.
    """

    request_data = (
        create_valid_request()
        .model_dump()
    )

    request_data[field_name] = -0.10

    with pytest.raises(ValidationError):
        DatasetQualityPredictionRequest.model_validate(
            request_data
        )


def test_zero_row_count_is_rejected() -> None:
    """
    Vérifie qu'un dataset doit contenir
    au moins une ligne.
    """

    request_data = (
        create_valid_request()
        .model_dump()
    )

    request_data["row_count"] = 0

    with pytest.raises(ValidationError):
        DatasetQualityPredictionRequest.model_validate(
            request_data
        )


def test_zero_column_count_is_rejected() -> None:
    """
    Vérifie qu'un dataset doit contenir
    au moins une colonne.
    """

    request_data = (
        create_valid_request()
        .model_dump()
    )

    request_data["column_count"] = 0

    with pytest.raises(ValidationError):
        DatasetQualityPredictionRequest.model_validate(
            request_data
        )


def test_negative_schema_change_count_is_rejected(
) -> None:
    """
    Vérifie que le nombre de changements
    de schéma ne peut pas être négatif.
    """

    request_data = (
        create_valid_request()
        .model_dump()
    )

    request_data[
        "schema_change_count"
    ] = -1

    with pytest.raises(ValidationError):
        DatasetQualityPredictionRequest.model_validate(
            request_data
        )


def test_negative_freshness_delay_is_rejected(
) -> None:
    """
    Vérifie que le retard ne peut pas être négatif.
    """

    request_data = (
        create_valid_request()
        .model_dump()
    )

    request_data[
        "freshness_delay_hours"
    ] = -1.0

    with pytest.raises(ValidationError):
        DatasetQualityPredictionRequest.model_validate(
            request_data
        )


def test_unknown_request_field_is_rejected() -> None:
    """
    Vérifie que les champs inattendus sont refusés.
    """

    request_data = (
        create_valid_request()
        .model_dump()
    )

    request_data[
        "unknown_field"
    ] = "unexpected"

    with pytest.raises(ValidationError):
        DatasetQualityPredictionRequest.model_validate(
            request_data
        )


def test_valid_prediction_response() -> None:
    """
    Vérifie une réponse d'inférence valide.
    """

    response = DatasetQualityPredictionResponse(
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

    assert response.prediction == (
        QualityLabel.ACCEPTABLE
    )

    assert response.confidence == pytest.approx(
        0.87
    )

    assert response.probabilities[
        "POOR"
    ] == pytest.approx(0.03)


def test_invalid_response_confidence_is_rejected(
) -> None:
    """
    Vérifie que la confiance reste comprise
    entre zéro et un.
    """

    with pytest.raises(ValidationError):
        DatasetQualityPredictionResponse(
            prediction=QualityLabel.GOOD,
            confidence=1.20,
            probabilities={
                "GOOD": 1.0,
            },
            model_name="random_forest",
            model_version="1.0.0",
        )


def test_invalid_prediction_label_is_rejected(
) -> None:
    """
    Vérifie qu'un label inconnu est refusé.
    """

    response_data = {
        "prediction": "EXCELLENT",
        "confidence": 0.90,
        "probabilities": {
            "EXCELLENT": 0.90,
        },
        "model_name": "random_forest",
        "model_version": "1.0.0",
    }

    with pytest.raises(ValidationError):
        DatasetQualityPredictionResponse.model_validate(
            response_data
        )


def test_valid_model_information() -> None:
    """
    Vérifie les informations du modèle actif.
    """

    information = ProductionModelInformation(
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

    assert information.status == "ACTIVE"
    assert information.model_name == (
        "random_forest"
    )

    assert information.f1_macro == pytest.approx(
        0.94
    )


def test_valid_health_response() -> None:
    """
    Vérifie la réponse de santé du service.
    """

    health = InferenceHealthResponse(
        status="healthy",
        model_loaded=True,
        preprocessor_loaded=True,
    )

    assert health.status == "healthy"
    assert health.model_loaded is True
    assert health.preprocessor_loaded is True


def test_health_response_rejects_unknown_field(
) -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    health_data = {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True,
        "unknown_field": "unexpected",
    }

    with pytest.raises(ValidationError):
        InferenceHealthResponse.model_validate(
            health_data
        )