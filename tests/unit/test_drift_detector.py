from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.inference_schemas import (
    DatasetQualityPredictionRequest,
)
from app.core.quality_schemas import (
    QualityLabel,
)
from app.services.drift_detector import (
    DriftDetector,
    calculate_normalized_mean_difference,
    calculate_psi,
    classify_drift_status,
    load_reference_features,
)
from app.services.inference_service import (
    FEATURE_NAMES,
)
from app.services.prediction_logger import (
    PredictionLogger,
)


def create_reference_dataset(
    output_path: Path,
    row_count: int = 200,
) -> pd.DataFrame:
    """
    Crée un dataset de référence numérique.
    """

    random_generator = (
        np.random.default_rng(
            42
        )
    )

    dataframe = pd.DataFrame(
        {
            "row_count": (
                random_generator.integers(
                    1000,
                    100000,
                    size=row_count,
                )
            ),
            "column_count": (
                random_generator.integers(
                    5,
                    50,
                    size=row_count,
                )
            ),
            "missing_value_ratio": (
                random_generator.uniform(
                    0.0,
                    0.30,
                    size=row_count,
                )
            ),
            "duplicate_row_ratio": (
                random_generator.uniform(
                    0.0,
                    0.20,
                    size=row_count,
                )
            ),
            "invalid_type_ratio": (
                random_generator.uniform(
                    0.0,
                    0.15,
                    size=row_count,
                )
            ),
            "primary_key_uniqueness": (
                random_generator.uniform(
                    0.75,
                    1.0,
                    size=row_count,
                )
            ),
            "foreign_key_match_ratio": (
                random_generator.uniform(
                    0.70,
                    1.0,
                    size=row_count,
                )
            ),
            "schema_change_count": (
                random_generator.integers(
                    0,
                    5,
                    size=row_count,
                )
            ),
            "outlier_ratio": (
                random_generator.uniform(
                    0.0,
                    0.20,
                    size=row_count,
                )
            ),
            "freshness_delay_hours": (
                random_generator.uniform(
                    0.0,
                    48.0,
                    size=row_count,
                )
            ),
            "quality_label": [
                "GOOD"
                for _ in range(
                    row_count
                )
            ],
        }
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return dataframe


def log_production_rows(
    logger: PredictionLogger,
    dataframe: pd.DataFrame,
) -> None:
    """
    Enregistre les lignes d'un DataFrame
    comme prédictions de production.
    """

    integer_features = {
        "row_count",
        "column_count",
        "schema_change_count",
    }

    for _, row in dataframe.iterrows():
        input_features = {}

        for feature_name in FEATURE_NAMES:
            value = row[feature_name]

            if feature_name in integer_features:
                input_features[feature_name] = int(
                    value
                )
            else:
                input_features[feature_name] = float(
                    value
                )

        logger.log_success(
            prediction=QualityLabel.GOOD,
            confidence=0.90,
            model_name="random_forest",
            model_version="1.0.0",
            latency_seconds=0.01,
            input_features=input_features,
        )

def test_load_reference_features(
    tmp_path: Path,
) -> None:
    dataset_path = (
        tmp_path
        / "reference.csv"
    )

    create_reference_dataset(
        dataset_path
    )

    features = load_reference_features(
        dataset_path
    )

    assert list(
        features.columns
    ) == FEATURE_NAMES

    assert features.shape == (200, 10)


def test_identical_distributions_have_low_psi() -> None:
    reference_values = np.linspace(
        0.0,
        1.0,
        1000,
    )

    production_values = (
        reference_values.copy()
    )

    psi = calculate_psi(
        reference_values,
        production_values,
    )

    assert psi == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_shifted_distribution_has_high_psi() -> None:
    random_generator = (
        np.random.default_rng(
            42
        )
    )

    reference_values = (
        random_generator.normal(
            loc=0.0,
            scale=1.0,
            size=2000,
        )
    )

    production_values = (
        random_generator.normal(
            loc=3.0,
            scale=1.0,
            size=2000,
        )
    )

    psi = calculate_psi(
        reference_values,
        production_values,
    )

    assert psi >= 0.25


def test_normalized_mean_difference() -> None:
    reference_values = np.asarray(
        [
            0.0,
            1.0,
            2.0,
            3.0,
        ]
    )

    production_values = np.asarray(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    difference = (
        calculate_normalized_mean_difference(
            reference_values,
            production_values,
        )
    )

    assert difference > 0.0


@pytest.mark.parametrize(
    "psi,status",
    [
        (0.05, "STABLE"),
        (0.10, "WARNING"),
        (0.24, "WARNING"),
        (0.25, "DRIFTED"),
        (0.50, "DRIFTED"),
    ],
)
def test_classify_drift_status(
    psi: float,
    status: str,
) -> None:
    result = classify_drift_status(
        psi,
        warning_threshold=0.10,
        drift_threshold=0.25,
    )

    assert result == status


def test_insufficient_production_data(
    tmp_path: Path,
) -> None:
    dataset_path = (
        tmp_path
        / "reference.csv"
    )

    reference_dataframe = (
        create_reference_dataset(
            dataset_path
        )
    )

    logger = PredictionLogger(
        tmp_path / "monitoring"
    )

    log_production_rows(
        logger,
        reference_dataframe.head(
            5
        ),
    )

    detector = DriftDetector(
        reference_dataset_path=(
            dataset_path
        ),
        prediction_logger=logger,
        minimum_production_records=30,
    )

    report = detector.analyze()

    assert report.overall_status == (
        "INSUFFICIENT_DATA"
    )

    assert report.production_record_count == 5
    assert report.features == []


def test_stable_production_data(
    tmp_path: Path,
) -> None:
    dataset_path = (
        tmp_path
        / "reference.csv"
    )

    reference_dataframe = (
        create_reference_dataset(
            dataset_path
        )
    )

    logger = PredictionLogger(
        tmp_path / "monitoring"
    )

    log_production_rows(
        logger,
        reference_dataframe.copy(),
    )

    detector = DriftDetector(
        reference_dataset_path=(
            dataset_path
        ),
        prediction_logger=logger,
        minimum_production_records=30,
    )

    report = detector.analyze()

    assert report.overall_status == "STABLE"

    assert report.analyzed_feature_count == 10

    assert report.stable_feature_count == 10

    assert report.drifted_feature_count == 0


def test_drifted_production_data(
    tmp_path: Path,
) -> None:
    dataset_path = (
        tmp_path
        / "reference.csv"
    )

    reference_dataframe = (
        create_reference_dataset(
            dataset_path
        )
    )

    production_dataframe = (
        reference_dataframe.copy()
    )

    production_dataframe[
        "missing_value_ratio"
    ] = 0.95

    production_dataframe[
        "freshness_delay_hours"
    ] = 500.0

    logger = PredictionLogger(
        tmp_path / "monitoring"
    )

    log_production_rows(
        logger,
        production_dataframe,
    )

    detector = DriftDetector(
        reference_dataset_path=(
            dataset_path
        ),
        prediction_logger=logger,
        minimum_production_records=30,
    )

    report = detector.analyze()

    assert report.overall_status == "DRIFTED"

    assert report.drifted_feature_count >= 2

    result_by_feature = {
        result.feature_name: result
        for result in report.features
    }

    assert (
        result_by_feature[
            "missing_value_ratio"
        ].status
        == "DRIFTED"
    )

    assert (
        result_by_feature[
            "freshness_delay_hours"
        ].status
        == "DRIFTED"
    )