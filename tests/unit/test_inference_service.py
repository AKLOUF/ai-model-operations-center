import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.inference_schemas import (
    DatasetQualityPredictionRequest,
)
from app.services.inference_service import (
    FEATURE_NAMES,
    InferenceService,
    PRODUCTION_MODEL_FILENAME,
    PRODUCTION_PREPROCESSOR_FILENAME,
    PRODUCTION_REGISTRY_FILENAME,
)


def create_raw_training_features() -> pd.DataFrame:
    """
    Crée des données brutes contenant les dix features.
    """

    rows = []

    for index in range(30):
        class_index = index % 3

        if class_index == 0:
            base_value = 0.02
            uniqueness = 0.99
            label_values = {
                "freshness": 2.0,
                "schema_changes": 0,
            }

        elif class_index == 1:
            base_value = 0.15
            uniqueness = 0.92
            label_values = {
                "freshness": 20.0,
                "schema_changes": 2,
            }

        else:
            base_value = 0.40
            uniqueness = 0.65
            label_values = {
                "freshness": 90.0,
                "schema_changes": 6,
            }

        rows.append(
            {
                "row_count": (
                    10000 + index
                ),
                "column_count": 15,
                "missing_value_ratio": (
                    base_value
                ),
                "duplicate_row_ratio": (
                    base_value / 2
                ),
                "invalid_type_ratio": (
                    base_value / 3
                ),
                "primary_key_uniqueness": (
                    uniqueness
                ),
                "foreign_key_match_ratio": (
                    uniqueness - 0.02
                ),
                "schema_change_count": (
                    label_values[
                        "schema_changes"
                    ]
                ),
                "outlier_ratio": (
                    base_value / 2
                ),
                "freshness_delay_hours": (
                    label_values[
                        "freshness"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=FEATURE_NAMES,
    )


def create_training_target() -> pd.Series:
    """
    Crée les labels associés aux données.
    """

    labels = []

    for index in range(30):
        class_index = index % 3

        if class_index == 0:
            labels.append("GOOD")

        elif class_index == 1:
            labels.append("ACCEPTABLE")

        else:
            labels.append("POOR")

    return pd.Series(
        labels,
        name="quality_label",
    )


def create_valid_request(
) -> DatasetQualityPredictionRequest:
    """
    Crée une requête correspondant
    à un dataset de bonne qualité.
    """

    return DatasetQualityPredictionRequest(
        row_count=20000,
        column_count=15,
        missing_value_ratio=0.02,
        duplicate_row_ratio=0.01,
        invalid_type_ratio=0.005,
        primary_key_uniqueness=0.99,
        foreign_key_match_ratio=0.97,
        schema_change_count=0,
        outlier_ratio=0.01,
        freshness_delay_hours=2.0,
    )


def create_production_artifacts(
    production_directory: Path,
) -> None:
    """
    Crée un modèle, un préprocesseur
    et un registre de production.
    """

    production_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_features = (
        create_raw_training_features()
    )

    target = create_training_target()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                FEATURE_NAMES,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    transformed_features = (
        preprocessor.fit_transform(
            raw_features
        )
    )

    transformed_dataframe = pd.DataFrame(
        np.asarray(
            transformed_features
        ),
        columns=(
            preprocessor
            .get_feature_names_out()
        ),
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(
        transformed_dataframe,
        target,
    )

    joblib.dump(
        model,
        production_directory
        / PRODUCTION_MODEL_FILENAME,
    )

    joblib.dump(
        preprocessor,
        production_directory
        / PRODUCTION_PREPROCESSOR_FILENAME,
    )

    registry_data = {
        "model_name": "logistic_regression",
        "model_version": "1.0.0",
        "status": "ACTIVE",
        "promoted_at": (
            "2026-10-08T12:00:00+00:00"
        ),
        "evaluated_split": "test",
        "model_path": str(
            production_directory
            / PRODUCTION_MODEL_FILENAME
        ),
        "preprocessor_path": str(
            production_directory
            / PRODUCTION_PREPROCESSOR_FILENAME
        ),
        "metadata_path": str(
            production_directory
            / "metadata.json"
        ),
        "test_metrics_path": str(
            production_directory
            / "test_metrics.json"
        ),
        "confusion_matrix_path": str(
            production_directory
            / "confusion_matrix.csv"
        ),
        "accuracy": 0.95,
        "precision_macro": 0.94,
        "recall_macro": 0.93,
        "f1_macro": 0.93,
        "poor_recall": 0.96,
    }

    (
        production_directory
        / PRODUCTION_REGISTRY_FILENAME
    ).write_text(
        json.dumps(
            registry_data,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_service_is_initially_unhealthy(
    tmp_path: Path,
) -> None:
    """
    Vérifie l'état initial du service.
    """

    service = InferenceService(
        tmp_path
    )

    health = service.get_health()

    assert health.status == "unhealthy"
    assert health.model_loaded is False

    assert (
        health.preprocessor_loaded
        is False
    )


def test_load_all_production_artifacts(
    tmp_path: Path,
) -> None:
    """
    Vérifie le chargement des trois artefacts.
    """

    create_production_artifacts(
        tmp_path
    )

    service = InferenceService(
        tmp_path
    )

    service.load_artifacts()

    assert service.artifacts_are_loaded() is True

    health = service.get_health()

    assert health.status == "healthy"
    assert health.model_loaded is True

    assert (
        health.preprocessor_loaded
        is True
    )


def test_missing_model_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie le comportement si le modèle est absent.
    """

    service = InferenceService(
        tmp_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="Modèle de production",
    ):
        service.load_model()


def test_build_input_dataframe(
    tmp_path: Path,
) -> None:
    """
    Vérifie la conversion de la requête en DataFrame.
    """

    service = InferenceService(
        tmp_path
    )

    dataframe = (
        service.build_input_dataframe(
            create_valid_request()
        )
    )

    assert dataframe.shape == (1, 10)

    assert list(
        dataframe.columns
    ) == FEATURE_NAMES


def test_transform_request(
    tmp_path: Path,
) -> None:
    """
    Vérifie la transformation des features.
    """

    create_production_artifacts(
        tmp_path
    )

    service = InferenceService(
        tmp_path
    )

    service.load_artifacts()

    transformed = service.transform_request(
        create_valid_request()
    )

    assert transformed.shape == (1, 10)

    assert list(
        transformed.columns
    ) == FEATURE_NAMES

    assert not transformed.isna().any().any()


def test_prediction(
    tmp_path: Path,
) -> None:
    """
    Vérifie une prédiction complète.
    """

    create_production_artifacts(
        tmp_path
    )

    service = InferenceService(
        tmp_path
    )

    service.load_artifacts()

    response = service.predict(
        create_valid_request()
    )

    assert response.prediction.value in {
        "GOOD",
        "ACCEPTABLE",
        "POOR",
    }

    assert 0.0 <= response.confidence <= 1.0

    assert set(
        response.probabilities
    ) == {
        "GOOD",
        "ACCEPTABLE",
        "POOR",
    }

    assert response.model_name == (
        "logistic_regression"
    )

    assert response.model_version == "1.0.0"


def test_probabilities_sum_to_one(
    tmp_path: Path,
) -> None:
    """
    Vérifie la cohérence des probabilités.
    """

    create_production_artifacts(
        tmp_path
    )

    service = InferenceService(
        tmp_path
    )

    service.load_artifacts()

    response = service.predict(
        create_valid_request()
    )

    probability_sum = sum(
        response.probabilities.values()
    )

    assert probability_sum == pytest.approx(
        1.0
    )


def test_model_information(
    tmp_path: Path,
) -> None:
    """
    Vérifie les informations du modèle actif.
    """

    create_production_artifacts(
        tmp_path
    )

    service = InferenceService(
        tmp_path
    )

    service.load_artifacts()

    information = (
        service.get_model_information()
    )

    assert information.model_name == (
        "logistic_regression"
    )

    assert information.model_version == (
        "1.0.0"
    )

    assert information.status == "ACTIVE"

    assert information.evaluated_split == (
        "test"
    )


def test_prediction_before_loading_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'une prédiction nécessite
    le chargement des artefacts.
    """

    service = InferenceService(
        tmp_path
    )

    with pytest.raises(
        RuntimeError,
        match="pas tous chargés",
    ):
        service.predict(
            create_valid_request()
        )