import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from app.core.model_schemas import (
    ModelEvaluationConfig,
    ModelEvaluationResult,
)
from app.core.promotion_config import (
    DEFAULT_PROMOTION_CONFIG_PATH,
    load_promotion_config,
)
from app.core.promotion_schemas import (
    ChampionDescriptor,
    ProductionModelRecord,
    PromotionConfig,
)
from training.evaluation import evaluate_classifier


CANDIDATE_MODEL_FILENAME = "model.joblib"

PRODUCTION_MODEL_FILENAME = "model.joblib"
PRODUCTION_PREPROCESSOR_FILENAME = "preprocessor.joblib"
PRODUCTION_METADATA_FILENAME = "metadata.json"
PRODUCTION_REGISTRY_FILENAME = "registry.json"

TEST_METRICS_FILENAME = "test_metrics.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.csv"


def utc_now_iso() -> str:
    """
    Retourne la date UTC actuelle au format ISO 8601.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def load_champion_descriptor(
    champion_file: Path,
) -> ChampionDescriptor:
    """
    Charge et valide le résultat de la sélection
    du champion.
    """

    resolved_path = champion_file.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Fichier du champion introuvable : "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "Le chemin du champion ne correspond "
            "pas à un fichier."
        )

    raw_content = json.loads(
        resolved_path.read_text(
            encoding="utf-8"
        )
    )

    champion = ChampionDescriptor.model_validate(
        raw_content
    )

    if not champion.champion_selected:
        raise ValueError(
            "Aucun modèle champion n'a été sélectionné. "
            f"Raison : {champion.reason}"
        )

    if champion.model_name is None:
        raise ValueError(
            "Le nom du modèle champion est absent."
        )

    if champion.model_version is None:
        raise ValueError(
            "La version du modèle champion est absente."
        )

    return champion


def load_test_features(
    features_file: Path,
) -> pd.DataFrame:
    """
    Charge et valide les features du jeu de test.
    """

    resolved_path = features_file.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Fichier X_test introuvable : "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "Le chemin X_test ne correspond "
            "pas à un fichier."
        )

    features = pd.read_csv(
        resolved_path
    )

    if features.empty:
        raise ValueError(
            "X_test ne peut pas être vide."
        )

    if features.columns.duplicated().any():
        raise ValueError(
            "X_test contient des colonnes dupliquées."
        )

    if features.isna().any().any():
        raise ValueError(
            "X_test contient des valeurs manquantes."
        )

    non_numeric_columns = []

    for column_name in features.columns:
        if not pd.api.types.is_numeric_dtype(
            features[column_name]
        ):
            non_numeric_columns.append(
                str(column_name)
            )

    if non_numeric_columns:
        raise ValueError(
            "X_test contient des features "
            f"non numériques : {non_numeric_columns}"
        )

    return features


def load_test_target(
    target_file: Path,
    target_column: str,
) -> pd.Series:
    """
    Charge et valide la cible du jeu de test.
    """

    resolved_path = target_file.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Fichier y_test introuvable : "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "Le chemin y_test ne correspond "
            "pas à un fichier."
        )

    target_dataframe = pd.read_csv(
        resolved_path
    )

    if target_dataframe.empty:
        raise ValueError(
            "y_test ne peut pas être vide."
        )

    if target_column not in target_dataframe.columns:
        raise ValueError(
            f"La colonne cible '{target_column}' "
            "est absente de y_test."
        )

    if len(target_dataframe.columns) != 1:
        raise ValueError(
            "y_test doit contenir uniquement "
            f"la colonne '{target_column}'."
        )

    target = target_dataframe[
        target_column
    ].copy()

    if target.isna().any():
        raise ValueError(
            "y_test contient des valeurs manquantes."
        )

    return target


def load_test_data(
    config: PromotionConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge les features et la cible du jeu de test.
    """

    X_test = load_test_features(
        config.test_data.features_file
    )

    y_test = load_test_target(
        config.test_data.target_file,
        config.test_data.target_column,
    )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test et y_test doivent contenir "
            "le même nombre d'observations."
        )

    return X_test, y_test


def get_candidate_model_path(
    champion: ChampionDescriptor,
    config: PromotionConfig,
) -> Path:
    """
    Retourne le chemin du modèle candidat champion.
    """

    if champion.model_name is None:
        raise ValueError(
            "Le nom du champion est absent."
        )

    model_path = (
        config.source.candidates_directory
        / champion.model_name
        / CANDIDATE_MODEL_FILENAME
    ).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            "Modèle candidat champion introuvable : "
            f"{model_path}"
        )

    if not model_path.is_file():
        raise ValueError(
            "Le chemin du modèle champion "
            "ne correspond pas à un fichier."
        )

    return model_path


def load_candidate_model(
    champion: ChampionDescriptor,
    config: PromotionConfig,
):
    """
    Charge le modèle candidat champion.
    """

    model_path = get_candidate_model_path(
        champion,
        config,
    )

    model = joblib.load(
        model_path
    )

    if not hasattr(model, "predict"):
        raise TypeError(
            "Le modèle champion ne possède pas "
            "de méthode predict."
        )

    return model


def create_test_evaluation_config(
    config: PromotionConfig,
) -> ModelEvaluationConfig:
    """
    Convertit les seuils de promotion en configuration
    compatible avec l'évaluateur commun.
    """

    return ModelEvaluationConfig(
        primary_metric="f1_macro",
        minimum_f1_macro=(
            config.evaluation.minimum_f1_macro
        ),
        minimum_poor_recall=(
            config.evaluation.minimum_poor_recall
        ),
    )


def evaluate_champion_on_test(
    champion: ChampionDescriptor,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: PromotionConfig,
) -> ModelEvaluationResult:
    """
    Évalue le champion sur le jeu de test final.
    """

    if champion.model_name is None:
        raise ValueError(
            "Le nom du champion est absent."
        )

    if champion.model_version is None:
        raise ValueError(
            "La version du champion est absente."
        )

    evaluation_config = (
        create_test_evaluation_config(
            config
        )
    )

    return evaluate_classifier(
        model=model,
        X_evaluation=X_test,
        y_evaluation=y_test,
        model_name=champion.model_name,
        model_version=champion.model_version,
        evaluation_config=evaluation_config,
        evaluated_split="test",
    )


def save_test_metrics(
    result: ModelEvaluationResult,
    output_path: Path,
) -> Path:
    """
    Sauvegarde le rapport d'évaluation sur le test.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path.write_text(
        result.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return resolved_path


def save_confusion_matrix(
    result: ModelEvaluationResult,
    output_path: Path,
) -> Path:
    """
    Sauvegarde la matrice de confusion finale.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_names = []

    for label in result.labels:
        row_names.append(
            f"actual_{label}"
        )

    column_names = []

    for label in result.labels:
        column_names.append(
            f"predicted_{label}"
        )

    matrix_dataframe = pd.DataFrame(
        result.confusion_matrix,
        index=row_names,
        columns=column_names,
    )

    matrix_dataframe.index.name = (
        "actual_label"
    )

    matrix_dataframe.to_csv(
        resolved_path,
        index=True,
        encoding="utf-8",
    )

    return resolved_path


def copy_file(
    source_path: Path,
    destination_path: Path,
) -> Path:
    """
    Copie un artefact vers son emplacement
    de production.
    """

    resolved_source = source_path.resolve()
    resolved_destination = (
        destination_path.resolve()
    )

    if not resolved_source.exists():
        raise FileNotFoundError(
            f"Artefact source introuvable : "
            f"{resolved_source}"
        )

    if not resolved_source.is_file():
        raise ValueError(
            "L'artefact source ne correspond "
            "pas à un fichier."
        )

    resolved_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        resolved_source,
        resolved_destination,
    )

    return resolved_destination


def save_json_dictionary(
    data: dict,
    output_path: Path,
) -> Path:
    """
    Sauvegarde un dictionnaire au format JSON.
    """

    resolved_path = output_path.resolve()

    resolved_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolved_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return resolved_path


def create_production_record(
    champion: ChampionDescriptor,
    result: ModelEvaluationResult,
    model_path: Path,
    preprocessor_path: Path,
    metadata_path: Path,
    test_metrics_path: Path,
    confusion_matrix_path: Path,
) -> ProductionModelRecord:
    """
    Construit l'enregistrement du modèle actif.
    """

    if champion.model_name is None:
        raise ValueError(
            "Le nom du champion est absent."
        )

    if champion.model_version is None:
        raise ValueError(
            "La version du champion est absente."
        )

    poor_recall = (
        result
        .metrics
        .metrics_by_class["POOR"]
        .recall
    )

    return ProductionModelRecord(
        model_name=champion.model_name,
        model_version=champion.model_version,
        status="ACTIVE",
        promoted_at=utc_now_iso(),
        evaluated_split="test",
        model_path=str(
            model_path.resolve()
        ),
        preprocessor_path=str(
            preprocessor_path.resolve()
        ),
        metadata_path=str(
            metadata_path.resolve()
        ),
        test_metrics_path=str(
            test_metrics_path.resolve()
        ),
        confusion_matrix_path=str(
            confusion_matrix_path.resolve()
        ),
        accuracy=result.metrics.accuracy,
        precision_macro=(
            result.metrics.precision_macro
        ),
        recall_macro=(
            result.metrics.recall_macro
        ),
        f1_macro=result.metrics.f1_macro,
        poor_recall=poor_recall,
    )


def promote_champion(
    config: PromotionConfig,
) -> ProductionModelRecord:
    """
    Évalue le champion sur le test et le promeut
    uniquement si tous les seuils sont respectés.
    """

    champion = load_champion_descriptor(
        config.champion_file
    )

    X_test, y_test = load_test_data(
        config
    )

    model = load_candidate_model(
        champion,
        config,
    )

    if hasattr(model, "n_features_in_"):
        expected_feature_count = int(
            model.n_features_in_
        )

        if len(X_test.columns) != (
            expected_feature_count
        ):
            raise ValueError(
                "Le nombre de features de X_test "
                "ne correspond pas au modèle champion."
            )

    evaluation_result = (
        evaluate_champion_on_test(
            champion,
            model,
            X_test,
            y_test,
            config,
        )
    )

    evaluation_directory = (
        config.output.evaluation_directory
    )

    test_metrics_path = (
        evaluation_directory
        / TEST_METRICS_FILENAME
    )

    confusion_matrix_path = (
        evaluation_directory
        / CONFUSION_MATRIX_FILENAME
    )

    save_test_metrics(
        evaluation_result,
        test_metrics_path,
    )

    save_confusion_matrix(
        evaluation_result,
        confusion_matrix_path,
    )

    if not evaluation_result.passed:
        reasons = "; ".join(
            evaluation_result.rejection_reasons
        )

        raise ValueError(
            "Le champion ne respecte pas "
            "les seuils de production : "
            f"{reasons}"
        )

    production_directory = (
        config.output.production_directory
    )

    production_model_path = (
        production_directory
        / PRODUCTION_MODEL_FILENAME
    )

    production_preprocessor_path = (
        production_directory
        / PRODUCTION_PREPROCESSOR_FILENAME
    )

    production_metadata_path = (
        production_directory
        / PRODUCTION_METADATA_FILENAME
    )

    production_registry_path = (
        production_directory
        / PRODUCTION_REGISTRY_FILENAME
    )

    candidate_model_path = (
        get_candidate_model_path(
            champion,
            config,
        )
    )

    copy_file(
        candidate_model_path,
        production_model_path,
    )

    copy_file(
        config.source.preprocessor_file,
        production_preprocessor_path,
    )

    metadata = {
        "model_name": champion.model_name,
        "model_version": (
            champion.model_version
        ),
        "status": "ACTIVE",
        "promoted_at": utc_now_iso(),
        "evaluated_split": "test",
        "source_model_path": str(
            candidate_model_path
        ),
        "source_champion_file": str(
            config.champion_file
        ),
        "test_row_count": len(
            X_test
        ),
        "feature_count": len(
            X_test.columns
        ),
        "feature_names": [
            str(column_name)
            for column_name in X_test.columns
        ],
        "test_metrics": {
            "accuracy": (
                evaluation_result
                .metrics
                .accuracy
            ),
            "precision_macro": (
                evaluation_result
                .metrics
                .precision_macro
            ),
            "recall_macro": (
                evaluation_result
                .metrics
                .recall_macro
            ),
            "f1_macro": (
                evaluation_result
                .metrics
                .f1_macro
            ),
            "poor_recall": (
                evaluation_result
                .metrics
                .metrics_by_class["POOR"]
                .recall
            ),
        },
    }

    save_json_dictionary(
        metadata,
        production_metadata_path,
    )

    production_record = (
        create_production_record(
            champion,
            evaluation_result,
            production_model_path,
            production_preprocessor_path,
            production_metadata_path,
            test_metrics_path,
            confusion_matrix_path,
        )
    )

    production_registry_path.write_text(
        production_record.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return production_record


def print_promotion_summary(
    record: ProductionModelRecord,
) -> None:
    """
    Affiche le résumé de la promotion.
    """

    print()
    print("=" * 70)
    print("PRODUCTION MODEL PROMOTION")
    print("=" * 70)

    print(
        f"Modèle : {record.model_name}"
    )

    print(
        f"Version : {record.model_version}"
    )

    print(
        f"Statut : {record.status}"
    )

    print(
        f"Évalué sur : "
        f"{record.evaluated_split}"
    )

    print(
        f"Accuracy : "
        f"{record.accuracy:.4f}"
    )

    print(
        f"Precision macro : "
        f"{record.precision_macro:.4f}"
    )

    print(
        f"Recall macro : "
        f"{record.recall_macro:.4f}"
    )

    print(
        f"F1 macro : "
        f"{record.f1_macro:.4f}"
    )

    print(
        f"Recall POOR : "
        f"{record.poor_recall:.4f}"
    )

    print()
    print(
        f"Modèle de production : "
        f"{record.model_path}"
    )

    print(
        f"Registre de production : "
        f"{record.metadata_path}"
    )

    print("=" * 70)
    print()


def main() -> int:
    """
    Point d'entrée du pipeline de promotion.
    """

    try:
        config = load_promotion_config(
            DEFAULT_PROMOTION_CONFIG_PATH
        )

        record = promote_champion(
            config
        )

        print_promotion_summary(
            record
        )

        print(
            "Le modèle champion a été promu "
            "en production."
        )

        return 0

    except FileNotFoundError as error:
        print(
            f"Fichier introuvable : {error}"
        )

        return 1

    except ValueError as error:
        print(
            f"Promotion refusée : {error}"
        )

        return 1

    except TypeError as error:
        print(
            f"Artefact incompatible : {error}"
        )

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)