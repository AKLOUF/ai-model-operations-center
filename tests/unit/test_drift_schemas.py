import pytest
from pydantic import ValidationError

from app.core.drift_schemas import (
    DriftReport,
    FeatureDriftResult,
)


def create_feature_result(
    status: str = "STABLE",
) -> FeatureDriftResult:
    """
    Crée un résultat valide.
    """

    return FeatureDriftResult(
        feature_name="missing_value_ratio",
        reference_count=3000,
        production_count=100,
        reference_mean=0.15,
        production_mean=0.16,
        reference_standard_deviation=0.10,
        normalized_mean_difference=0.10,
        psi=0.05,
        status=status,
    )


def test_valid_feature_drift_result() -> None:
    result = create_feature_result()

    assert result.status == "STABLE"

    assert result.psi == pytest.approx(
        0.05
    )


def test_negative_psi_is_rejected() -> None:
    data = (
        create_feature_result()
        .model_dump()
    )

    data["psi"] = -0.01

    with pytest.raises(ValidationError):
        FeatureDriftResult.model_validate(
            data
        )


def test_invalid_drift_status_is_rejected() -> None:
    data = (
        create_feature_result()
        .model_dump()
    )

    data["status"] = "UNKNOWN"

    with pytest.raises(ValidationError):
        FeatureDriftResult.model_validate(
            data
        )


def test_valid_drift_report() -> None:
    feature = create_feature_result()

    report = DriftReport(
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
            feature
        ],
    )

    assert report.overall_status == "STABLE"
    assert report.analyzed_feature_count == 1


def test_insufficient_data_report() -> None:
    report = DriftReport(
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

    assert report.overall_status == (
        "INSUFFICIENT_DATA"
    )

    assert report.features == []