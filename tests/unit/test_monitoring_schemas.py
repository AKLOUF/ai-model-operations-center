import pytest
from pydantic import ValidationError

from app.core.monitoring_schemas import (
    PredictionLogRecord,
    PredictionMonitoringSummary,
)
from app.core.quality_schemas import (
    QualityLabel,
)


def create_success_record() -> PredictionLogRecord:
    """
    Crée un événement réussi valide.
    """

    return PredictionLogRecord(
        prediction_id=(
            "pred_20261008_120500_ab12cd34"
        ),
        timestamp=(
            "2026-10-08T12:05:00+00:00"
        ),
        prediction=QualityLabel.GOOD,
        confidence=0.95,
        model_name="random_forest",
        model_version="1.0.0",
        input_features={
            "row_count": 20000,
            "column_count": 18,
            "missing_value_ratio": 0.02,
            "duplicate_row_ratio": 0.01,
            "invalid_type_ratio": 0.005,
            "primary_key_uniqueness": 0.99,
            "foreign_key_match_ratio": 0.97,
            "schema_change_count": 0,
            "outlier_ratio": 0.01,
            "freshness_delay_hours": 2.0,
        },
        latency_seconds=0.01,
        success=True,
        error_message=None,
    )


def test_valid_success_record() -> None:
    record = create_success_record()

    assert record.success is True

    assert record.prediction == (
        QualityLabel.GOOD
    )

    assert record.confidence == pytest.approx(
        0.95
    )


def test_valid_failure_record() -> None:
    record = PredictionLogRecord(
        prediction_id="pred_failed",
        timestamp=(
            "2026-10-08T12:05:00+00:00"
        ),
        prediction=None,
        confidence=None,
        model_name=None,
        model_version=None,
        latency_seconds=0.02,
        success=False,
        error_message="Prediction failed.",
    )

    assert record.success is False
    assert record.prediction is None

    assert record.error_message == (
        "Prediction failed."
    )


def test_invalid_confidence_is_rejected() -> None:
    record_data = (
        create_success_record()
        .model_dump()
    )

    record_data["confidence"] = 1.20

    with pytest.raises(ValidationError):
        PredictionLogRecord.model_validate(
            record_data
        )


def test_negative_latency_is_rejected() -> None:
    record_data = (
        create_success_record()
        .model_dump()
    )

    record_data[
        "latency_seconds"
    ] = -0.01

    with pytest.raises(ValidationError):
        PredictionLogRecord.model_validate(
            record_data
        )


def test_unknown_record_field_is_rejected() -> None:
    record_data = (
        create_success_record()
        .model_dump()
    )

    record_data[
        "unknown_field"
    ] = "unexpected"

    with pytest.raises(ValidationError):
        PredictionLogRecord.model_validate(
            record_data
        )


def test_valid_monitoring_summary() -> None:
    summary = PredictionMonitoringSummary(
        total_prediction_count=10,
        successful_prediction_count=9,
        failed_prediction_count=1,
        success_rate=0.90,
        average_confidence=0.85,
        average_latency_seconds=0.01,
        prediction_distribution={
            "GOOD": 4,
            "ACCEPTABLE": 3,
            "POOR": 2,
        },
        model_versions={
            "random_forest:1.0.0": 9,
        },
    )

    assert summary.total_prediction_count == 10

    assert summary.success_rate == pytest.approx(
        0.90
    )

    assert summary.prediction_distribution[
        "GOOD"
    ] == 4


def test_invalid_success_rate_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionMonitoringSummary(
            total_prediction_count=10,
            successful_prediction_count=9,
            failed_prediction_count=1,
            success_rate=1.20,
            average_confidence=0.85,
            average_latency_seconds=0.01,
            prediction_distribution={},
            model_versions={},
        )

def test_success_record_contains_input_features() -> None:
    """
    Vérifie que les features utilisées pour
    la prédiction peuvent être enregistrées.
    """

    record = create_success_record()

    assert record.input_features is not None

    assert record.input_features[
        "row_count"
    ] == 20000

    assert record.input_features[
        "missing_value_ratio"
    ] == pytest.approx(0.02)

def test_input_features_are_optional() -> None:
    """
    Vérifie que les anciens journaux sans features
    restent compatibles.
    """

    record = PredictionLogRecord(
        prediction_id="pred_legacy",
        timestamp=(
            "2026-10-08T12:05:00+00:00"
        ),
        prediction=QualityLabel.GOOD,
        confidence=0.95,
        model_name="random_forest",
        model_version="1.0.0",
        latency_seconds=0.01,
        success=True,
        error_message=None,
    )

    assert record.input_features is None