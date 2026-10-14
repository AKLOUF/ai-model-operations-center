import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.core.promotion_schemas import (
    PromotionConfig,
    PromotionEvaluationConfig,
    PromotionOutputConfig,
    PromotionSourceConfig,
    PromotionTestDataConfig,
)
from pipelines.promote_champion import (
    CONFUSION_MATRIX_FILENAME,
    PRODUCTION_METADATA_FILENAME,
    PRODUCTION_MODEL_FILENAME,
    PRODUCTION_PREPROCESSOR_FILENAME,
    PRODUCTION_REGISTRY_FILENAME,
    TEST_METRICS_FILENAME,
    promote_champion,
)


def create_promotion_test_files(
    tmp_path: Path,
) -> PromotionConfig:
    """
    Crée un champion et des données de test
    parfaitement séparables.
    """

    candidates_directory = (
        tmp_path
        / "models"
        / "candidates"
    )

    candidate_directory = (
        candidates_directory
        / "logistic_regression"
    )

    candidate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train = pd.DataFrame(
        {
            "feature_a": [
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "feature_b": [
                -3.0,
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        }
    )

    y_train = pd.Series(
        [
            "POOR",
            "POOR",
            "POOR",
            "ACCEPTABLE",
            "ACCEPTABLE",
            "ACCEPTABLE",
            "GOOD",
            "GOOD",
            "GOOD",
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        model,
        candidate_directory
        / "model.joblib",
    )

    processed_directory = (
        tmp_path
        / "data"
        / "processed"
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train.to_csv(
        processed_directory
        / "X_test.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "quality_label": y_train
        }
    ).to_csv(
        processed_directory
        / "y_test.csv",
        index=False,
    )

    comparison_directory = (
        tmp_path
        / "reports"
        / "evaluation"
        / "candidates"
    )

    comparison_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    champion_file = (
        comparison_directory
        / "champion.json"
    )

    champion_file.write_text(
        json.dumps(
            {
                "champion_selected": True,
                "model_name": (
                    "logistic_regression"
                ),
                "model_version": "1.0.0",
                "f1_macro": 1.0,
                "poor_recall": 1.0,
                "accuracy": 1.0,
                "prediction_latency_seconds": (
                    0.01
                ),
                "training_duration_seconds": (
                    0.10
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    preprocessor_file = (
        tmp_path
        / "models"
        / "preprocessing"
        / "preprocessor.joblib"
    )

    preprocessor_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "test_preprocessor": True
        },
        preprocessor_file,
    )

    return PromotionConfig(
        champion_file=champion_file,
        test_data=PromotionTestDataConfig(
            features_file=(
                processed_directory
                / "X_test.csv"
            ),
            target_file=(
                processed_directory
                / "y_test.csv"
            ),
            target_column="quality_label",
        ),
        source=PromotionSourceConfig(
            candidates_directory=(
                candidates_directory
            ),
            preprocessor_file=(
                preprocessor_file
            ),
        ),
        evaluation=PromotionEvaluationConfig(
            minimum_f1_macro=0.50,
            minimum_poor_recall=0.50,
        ),
        output=PromotionOutputConfig(
            production_directory=(
                tmp_path
                / "models"
                / "production"
            ),
            evaluation_directory=(
                tmp_path
                / "reports"
                / "evaluation"
                / "production"
            ),
        ),
    )


def test_champion_is_promoted(
    tmp_path: Path,
) -> None:
    config = create_promotion_test_files(
        tmp_path
    )

    record = promote_champion(
        config
    )

    production_directory = (
        config.output.production_directory
    )

    evaluation_directory = (
        config.output.evaluation_directory
    )

    assert record.status == "ACTIVE"

    assert (
        record.model_name
        == "logistic_regression"
    )

    assert (
        production_directory
        / PRODUCTION_MODEL_FILENAME
    ).exists()

    assert (
        production_directory
        / PRODUCTION_PREPROCESSOR_FILENAME
    ).exists()

    assert (
        production_directory
        / PRODUCTION_METADATA_FILENAME
    ).exists()

    assert (
        production_directory
        / PRODUCTION_REGISTRY_FILENAME
    ).exists()

    assert (
        evaluation_directory
        / TEST_METRICS_FILENAME
    ).exists()

    assert (
        evaluation_directory
        / CONFUSION_MATRIX_FILENAME
    ).exists()

    production_model = joblib.load(
        production_directory
        / PRODUCTION_MODEL_FILENAME
    )

    assert hasattr(
        production_model,
        "classes_",
    )

    registry_data = json.loads(
        (
            production_directory
            / PRODUCTION_REGISTRY_FILENAME
        ).read_text(
            encoding="utf-8"
        )
    )

    assert registry_data["status"] == "ACTIVE"

    assert (
        registry_data["evaluated_split"]
        == "test"
    )