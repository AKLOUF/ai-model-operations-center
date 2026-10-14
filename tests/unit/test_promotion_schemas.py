from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.promotion_schemas import (
    ChampionDescriptor,
    ProductionModelRecord,
    PromotionConfig,
    PromotionEvaluationConfig,
    PromotionOutputConfig,
    PromotionSourceConfig,
    PromotionTestDataConfig,
)


def create_test_data_config() -> PromotionTestDataConfig:
    """
    Crée une configuration valide du jeu de test.
    """

    return PromotionTestDataConfig(
        features_file=Path(
            "data/processed/X_test.csv"
        ),
        target_file=Path(
            "data/processed/y_test.csv"
        ),
        target_column="quality_label",
    )


def create_source_config() -> PromotionSourceConfig:
    """
    Crée une configuration valide des artefacts sources.
    """

    return PromotionSourceConfig(
        candidates_directory=Path(
            "models/candidates"
        ),
        preprocessor_file=Path(
            "models/preprocessing/preprocessor.joblib"
        ),
    )


def create_evaluation_config() -> PromotionEvaluationConfig:
    """
    Crée des seuils de promotion valides.
    """

    return PromotionEvaluationConfig(
        minimum_f1_macro=0.80,
        minimum_poor_recall=0.85,
    )


def create_output_config() -> PromotionOutputConfig:
    """
    Crée une configuration valide des sorties.
    """

    return PromotionOutputConfig(
        production_directory=Path(
            "models/production"
        ),
        evaluation_directory=Path(
            "reports/evaluation/production"
        ),
    )


def create_promotion_config() -> PromotionConfig:
    """
    Crée une configuration complète et valide.
    """

    return PromotionConfig(
        champion_file=Path(
            "reports/evaluation/candidates/champion.json"
        ),
        test_data=create_test_data_config(),
        source=create_source_config(),
        evaluation=create_evaluation_config(),
        output=create_output_config(),
    )


def create_production_record() -> ProductionModelRecord:
    """
    Crée un enregistrement valide du modèle de production.
    """

    return ProductionModelRecord(
        model_name="random_forest",
        model_version="1.0.0",
        status="ACTIVE",
        promoted_at=(
            "2026-10-08T12:00:00+00:00"
        ),
        evaluated_split="test",
        model_path=(
            "models/production/model.joblib"
        ),
        preprocessor_path=(
            "models/production/preprocessor.joblib"
        ),
        metadata_path=(
            "models/production/metadata.json"
        ),
        test_metrics_path=(
            "reports/evaluation/production/"
            "test_metrics.json"
        ),
        confusion_matrix_path=(
            "reports/evaluation/production/"
            "confusion_matrix.csv"
        ),
        accuracy=0.95,
        precision_macro=0.94,
        recall_macro=0.93,
        f1_macro=0.93,
        poor_recall=0.96,
    )


def test_valid_test_data_config() -> None:
    """
    Vérifie la configuration du jeu de test.
    """

    config = create_test_data_config()

    assert config.features_file == Path(
        "data/processed/X_test.csv"
    )

    assert config.target_file == Path(
        "data/processed/y_test.csv"
    )

    assert config.target_column == (
        "quality_label"
    )


def test_empty_target_column_is_rejected() -> None:
    """
    Vérifie que la colonne cible ne peut pas être vide.
    """

    with pytest.raises(ValidationError):
        PromotionTestDataConfig(
            features_file=Path(
                "data/processed/X_test.csv"
            ),
            target_file=Path(
                "data/processed/y_test.csv"
            ),
            target_column="",
        )


def test_unknown_test_data_field_is_rejected() -> None:
    """
    Vérifie que les champs supplémentaires sont refusés.
    """

    configuration = {
        "features_file": (
            "data/processed/X_test.csv"
        ),
        "target_file": (
            "data/processed/y_test.csv"
        ),
        "target_column": "quality_label",
        "unknown_field": "unexpected",
    }

    with pytest.raises(ValidationError):
        PromotionTestDataConfig.model_validate(
            configuration
        )


def test_valid_source_config() -> None:
    """
    Vérifie la configuration des artefacts sources.
    """

    config = create_source_config()

    assert config.candidates_directory == Path(
        "models/candidates"
    )

    assert config.preprocessor_file == Path(
        "models/preprocessing/preprocessor.joblib"
    )


def test_source_config_requires_preprocessor() -> None:
    """
    Vérifie que le préprocesseur est obligatoire.
    """

    configuration = {
        "candidates_directory": (
            "models/candidates"
        )
    }

    with pytest.raises(ValidationError):
        PromotionSourceConfig.model_validate(
            configuration
        )


def test_valid_evaluation_config() -> None:
    """
    Vérifie les seuils de promotion.
    """

    config = create_evaluation_config()

    assert config.minimum_f1_macro == pytest.approx(
        0.80
    )

    assert (
        config.minimum_poor_recall
        == pytest.approx(0.85)
    )


def test_f1_above_one_is_rejected() -> None:
    """
    Vérifie que le F1 minimum est limité à 1.
    """

    with pytest.raises(ValidationError):
        PromotionEvaluationConfig(
            minimum_f1_macro=1.10,
            minimum_poor_recall=0.85,
        )


def test_negative_f1_is_rejected() -> None:
    """
    Vérifie que le F1 minimum ne peut pas être négatif.
    """

    with pytest.raises(ValidationError):
        PromotionEvaluationConfig(
            minimum_f1_macro=-0.10,
            minimum_poor_recall=0.85,
        )


def test_poor_recall_above_one_is_rejected() -> None:
    """
    Vérifie que le recall POOR est limité à 1.
    """

    with pytest.raises(ValidationError):
        PromotionEvaluationConfig(
            minimum_f1_macro=0.80,
            minimum_poor_recall=1.10,
        )


def test_negative_poor_recall_is_rejected() -> None:
    """
    Vérifie que le recall POOR ne peut pas être négatif.
    """

    with pytest.raises(ValidationError):
        PromotionEvaluationConfig(
            minimum_f1_macro=0.80,
            minimum_poor_recall=-0.10,
        )


def test_valid_output_config() -> None:
    """
    Vérifie la configuration des sorties.
    """

    config = create_output_config()

    assert config.production_directory == Path(
        "models/production"
    )

    assert config.evaluation_directory == Path(
        "reports/evaluation/production"
    )


def test_output_config_requires_all_directories() -> None:
    """
    Vérifie que les deux dossiers sont obligatoires.
    """

    configuration = {
        "production_directory": (
            "models/production"
        )
    }

    with pytest.raises(ValidationError):
        PromotionOutputConfig.model_validate(
            configuration
        )


def test_valid_promotion_config() -> None:
    """
    Vérifie la configuration complète de promotion.
    """

    config = create_promotion_config()

    assert config.champion_file == Path(
        "reports/evaluation/candidates/champion.json"
    )

    assert config.test_data.target_column == (
        "quality_label"
    )

    assert config.source.candidates_directory == Path(
        "models/candidates"
    )

    assert config.output.production_directory == Path(
        "models/production"
    )


def test_promotion_config_rejects_unknown_field() -> None:
    """
    Vérifie que les champs supplémentaires sont refusés.
    """

    configuration = (
        create_promotion_config()
        .model_dump()
    )

    configuration["unknown_field"] = (
        "unexpected"
    )

    with pytest.raises(ValidationError):
        PromotionConfig.model_validate(
            configuration
        )


def test_promotion_config_is_immutable() -> None:
    """
    Vérifie que la configuration est immutable.
    """

    config = create_promotion_config()

    with pytest.raises(ValidationError):
        setattr(
            config,
            "champion_file",
            Path("another/champion.json"),
        )


def test_valid_selected_champion() -> None:
    """
    Vérifie un champion sélectionné.
    """

    champion = ChampionDescriptor(
        champion_selected=True,
        model_name="random_forest",
        model_version="1.0.0",
        f1_macro=0.95,
        poor_recall=0.96,
        accuracy=0.95,
        prediction_latency_seconds=0.02,
        training_duration_seconds=2.0,
        reason=None,
    )

    assert champion.champion_selected is True

    assert champion.model_name == (
        "random_forest"
    )

    assert champion.model_version == "1.0.0"

    assert champion.f1_macro == pytest.approx(
        0.95
    )


def test_valid_champion_not_selected() -> None:
    """
    Vérifie le cas où aucun champion n'est sélectionné.
    """

    champion = ChampionDescriptor(
        champion_selected=False,
        model_name=None,
        model_version=None,
        f1_macro=None,
        poor_recall=None,
        accuracy=None,
        prediction_latency_seconds=None,
        training_duration_seconds=None,
        reason=(
            "Aucun modèle ne respecte les seuils."
        ),
    )

    assert champion.champion_selected is False
    assert champion.model_name is None
    assert champion.model_version is None

    assert champion.reason == (
        "Aucun modèle ne respecte les seuils."
    )


def test_invalid_champion_metric_is_rejected() -> None:
    """
    Vérifie que les métriques du champion
    restent comprises entre zéro et un.
    """

    with pytest.raises(ValidationError):
        ChampionDescriptor(
            champion_selected=True,
            model_name="random_forest",
            model_version="1.0.0",
            f1_macro=1.20,
            poor_recall=0.96,
            accuracy=0.95,
            prediction_latency_seconds=0.02,
            training_duration_seconds=2.0,
            reason=None,
        )


def test_unknown_champion_field_is_rejected() -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    configuration = {
        "champion_selected": True,
        "model_name": "random_forest",
        "model_version": "1.0.0",
        "f1_macro": 0.95,
        "poor_recall": 0.96,
        "accuracy": 0.95,
        "prediction_latency_seconds": 0.02,
        "training_duration_seconds": 2.0,
        "reason": None,
        "unknown_field": "unexpected",
    }

    with pytest.raises(ValidationError):
        ChampionDescriptor.model_validate(
            configuration
        )


def test_valid_production_record() -> None:
    """
    Vérifie un modèle de production valide.
    """

    record = create_production_record()

    assert record.model_name == (
        "random_forest"
    )

    assert record.model_version == "1.0.0"
    assert record.status == "ACTIVE"
    assert record.evaluated_split == "test"

    assert record.accuracy == pytest.approx(
        0.95
    )

    assert record.f1_macro == pytest.approx(
        0.93
    )

    assert record.poor_recall == pytest.approx(
        0.96
    )


def test_invalid_production_status_is_rejected() -> None:
    """
    Vérifie qu'un statut inconnu est refusé.
    """

    configuration = (
        create_production_record()
        .model_dump()
    )

    configuration["status"] = "UNKNOWN"

    with pytest.raises(ValidationError):
        ProductionModelRecord.model_validate(
            configuration
        )


def test_invalid_production_metric_is_rejected() -> None:
    """
    Vérifie que les métriques de production
    restent comprises entre zéro et un.
    """

    configuration = (
        create_production_record()
        .model_dump()
    )

    configuration["poor_recall"] = -0.10

    with pytest.raises(ValidationError):
        ProductionModelRecord.model_validate(
            configuration
        )


def test_empty_production_model_name_is_rejected() -> None:
    """
    Vérifie que le nom du modèle est obligatoire.
    """

    configuration = (
        create_production_record()
        .model_dump()
    )

    configuration["model_name"] = ""

    with pytest.raises(ValidationError):
        ProductionModelRecord.model_validate(
            configuration
        )


def test_unknown_production_field_is_rejected() -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    configuration = (
        create_production_record()
        .model_dump()
    )

    configuration["unknown_field"] = (
        "unexpected"
    )

    with pytest.raises(ValidationError):
        ProductionModelRecord.model_validate(
            configuration
        )