from pathlib import Path

import pytest

from app.core.quality_schemas import (
    QualityLabel,
)
from app.services.prediction_logger import (
    PREDICTION_LOG_FILENAME,
    PredictionLogger,
    generate_prediction_id,
    utc_now_iso,
)


def create_input_features() -> dict[
    str,
    float | int,
]:
    """
    Crée les features utilisées par les tests.
    """

    return {
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
    }

def test_generate_prediction_id() -> None:
    first_identifier = generate_prediction_id()
    second_identifier = generate_prediction_id()

    assert first_identifier.startswith(
        "pred_"
    )

    assert first_identifier != (
        second_identifier
    )


def test_utc_now_iso() -> None:
    timestamp = utc_now_iso()

    assert isinstance(
        timestamp,
        str,
    )

    assert "+00:00" in timestamp


def test_log_success(
    tmp_path: Path,
) -> None:
    logger = PredictionLogger(
        tmp_path
    )

    record = logger.log_success(
        prediction=QualityLabel.GOOD,
        confidence=0.95,
        model_name="random_forest",
        model_version="1.0.0",
        latency_seconds=0.01,
    )

    assert record.success is True

    assert logger.prediction_log_path.exists()

    assert logger.prediction_log_path.name == (
        PREDICTION_LOG_FILENAME
    )


def test_log_failure(
    tmp_path: Path,
) -> None:
    logger = PredictionLogger(
        tmp_path
    )

    record = logger.log_failure(
        latency_seconds=0.02,
        error_message="Prediction failed.",
    )

    assert record.success is False

    assert record.error_message == (
        "Prediction failed."
    )


def test_empty_failure_message_is_rejected(
    tmp_path: Path,
) -> None:
    logger = PredictionLogger(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="ne peut pas être vide",
    ):
        logger.log_failure(
            latency_seconds=0.01,
            error_message="",
        )


def test_read_empty_log(
    tmp_path: Path,
) -> None:
    logger = PredictionLogger(
        tmp_path
    )

    records = logger.read_records()

    assert records == []


def test_read_logged_records(
    tmp_path: Path,
) -> None:
    logger = PredictionLogger(
        tmp_path
    )

    logger.log_success(
        prediction=QualityLabel.GOOD,
        confidence=0.95,
        model_name="random_forest",
        model_version="1.0.0",
        latency_seconds=0.01,
    )

    logger.log_failure(
        latency_seconds=0.02,
        error_message="Prediction failed.",
    )

    records = logger.read_records()

    assert len(records) == 2

    assert records[0].success is True
    assert records[1].success is False


def test_build_empty_summary(
    tmp_path: Path,
) -> None:
    logger = PredictionLogger(
        tmp_path
    )

    summary = logger.build_summary()

    assert summary.total_prediction_count == 0

    assert (
        summary.successful_prediction_count
        == 0
    )

    assert summary.failed_prediction_count == 0

    assert summary.success_rate == pytest.approx(
        0.0
    )

    assert summary.average_confidence is None

    assert (
        summary.average_latency_seconds
        is None
    )


def test_build_monitoring_summary(
    tmp_path: Path,
) -> None:
    logger = PredictionLogger(
        tmp_path
    )

    logger.log_success(
        prediction=QualityLabel.GOOD,
        confidence=0.90,
        model_name="random_forest",
        model_version="1.0.0",
        latency_seconds=0.01,
    )

    logger.log_success(
        prediction=QualityLabel.POOR,
        confidence=0.80,
        model_name="random_forest",
        model_version="1.0.0",
        latency_seconds=0.03,
    )

    logger.log_failure(
        latency_seconds=0.02,
        error_message="Prediction failed.",
    )

    summary = logger.build_summary()

    assert summary.total_prediction_count == 3

    assert (
        summary.successful_prediction_count
        == 2
    )

    assert summary.failed_prediction_count == 1

    assert summary.success_rate == pytest.approx(
        2 / 3
    )

    assert summary.average_confidence == (
        pytest.approx(0.85)
    )

    assert summary.average_latency_seconds == (
        pytest.approx(0.02)
    )

    assert summary.prediction_distribution == {
        "GOOD": 1,
        "POOR": 1,
    }

    assert summary.model_versions == {
        "random_forest:1.0.0": 2,
    }

def test_log_success_with_input_features(
    tmp_path: Path,
) -> None:
    """
    Vérifie que les features d'entrée sont
    enregistrées et rechargées.
    """

    logger = PredictionLogger(
        tmp_path
    )

    logger.log_success(
        prediction=QualityLabel.GOOD,
        confidence=0.95,
        model_name="random_forest",
        model_version="1.0.0",
        latency_seconds=0.01,
        input_features=(
            create_input_features()
        ),
    )

    records = logger.read_records()

    assert len(records) == 1

    record = records[0]

    assert record.input_features is not None

    assert record.input_features[
        "row_count"
    ] == 20000

    assert record.input_features[
        "freshness_delay_hours"
    ] == pytest.approx(2.0)