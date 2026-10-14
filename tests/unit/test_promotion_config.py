from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.promotion_config import (
    load_promotion_config,
    load_promotion_yaml,
    normalize_promotion_paths,
    resolve_promotion_path,
)


VALID_PROMOTION_YAML = """
promotion:
  champion_file: "reports/evaluation/candidates/champion.json"

  test_data:
    features_file: "data/processed/X_test.csv"
    target_file: "data/processed/y_test.csv"
    target_column: "quality_label"

  source:
    candidates_directory: "models/candidates"
    preprocessor_file: "models/preprocessing/preprocessor.joblib"

  evaluation:
    minimum_f1_macro: 0.80
    minimum_poor_recall: 0.85

  output:
    production_directory: "models/production"
    evaluation_directory: "reports/evaluation/production"
"""


def write_yaml_file(
    file_path: Path,
    content: str,
) -> Path:
    """
    Écrit un fichier YAML temporaire.
    """

    file_path.write_text(
        content.strip(),
        encoding="utf-8",
    )

    return file_path


def create_raw_promotion_config() -> dict:
    """
    Crée la section promotion avant normalisation
    des chemins.
    """

    return {
        "champion_file": (
            "reports/evaluation/"
            "candidates/champion.json"
        ),
        "test_data": {
            "features_file": (
                "data/processed/X_test.csv"
            ),
            "target_file": (
                "data/processed/y_test.csv"
            ),
            "target_column": "quality_label",
        },
        "source": {
            "candidates_directory": (
                "models/candidates"
            ),
            "preprocessor_file": (
                "models/preprocessing/"
                "preprocessor.joblib"
            ),
        },
        "evaluation": {
            "minimum_f1_macro": 0.80,
            "minimum_poor_recall": 0.85,
        },
        "output": {
            "production_directory": (
                "models/production"
            ),
            "evaluation_directory": (
                "reports/evaluation/production"
            ),
        },
    }


def test_resolve_relative_path() -> None:
    """
    Vérifie qu'un chemin relatif devient absolu.
    """

    resolved_path = resolve_promotion_path(
        "models/production"
    )

    assert resolved_path.is_absolute()
    assert resolved_path.name == "production"


def test_resolve_absolute_path(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un chemin absolu est conservé.
    """

    absolute_path = (
        tmp_path
        / "models"
        / "production"
    )

    resolved_path = resolve_promotion_path(
        absolute_path
    )

    assert resolved_path == (
        absolute_path.resolve()
    )


def test_load_valid_yaml(
    tmp_path: Path,
) -> None:
    """
    Vérifie la lecture d'un fichier YAML valide.
    """

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        VALID_PROMOTION_YAML,
    )

    result = load_promotion_yaml(
        configuration_path
    )

    assert isinstance(
        result,
        dict,
    )

    assert "promotion" in result


def test_missing_yaml_file_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie le comportement avec un fichier absent.
    """

    missing_path = (
        tmp_path
        / "missing.yaml"
    )

    with pytest.raises(
        FileNotFoundError,
        match="introuvable",
    ):
        load_promotion_yaml(
            missing_path
        )


def test_yaml_path_must_be_file(
    tmp_path: Path,
) -> None:
    """
    Vérifie que le chemin doit désigner un fichier.
    """

    directory_path = (
        tmp_path
        / "configuration"
    )

    directory_path.mkdir()

    with pytest.raises(
        ValueError,
        match="ne correspond pas à un fichier",
    ):
        load_promotion_yaml(
            directory_path
        )


def test_yaml_root_must_be_dictionary(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la racine YAML doit être un objet.
    """

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        """
- first
- second
""",
    )

    with pytest.raises(
        ValueError,
        match="objet à sa racine",
    ):
        load_promotion_yaml(
            configuration_path
        )


def test_normalize_promotion_paths() -> None:
    """
    Vérifie la normalisation de tous les chemins.
    """

    configuration = (
        create_raw_promotion_config()
    )

    result = normalize_promotion_paths(
        configuration
    )

    assert isinstance(
        result,
        dict,
    )

    assert result[
        "champion_file"
    ].is_absolute()

    assert result[
        "test_data"
    ][
        "features_file"
    ].is_absolute()

    assert result[
        "test_data"
    ][
        "target_file"
    ].is_absolute()

    assert result[
        "source"
    ][
        "candidates_directory"
    ].is_absolute()

    assert result[
        "source"
    ][
        "preprocessor_file"
    ].is_absolute()

    assert result[
        "output"
    ][
        "production_directory"
    ].is_absolute()

    assert result[
        "output"
    ][
        "evaluation_directory"
    ].is_absolute()

    assert result[
        "evaluation"
    ][
        "minimum_f1_macro"
    ] == pytest.approx(0.80)


def test_normalization_does_not_modify_original_config(
) -> None:
    """
    Vérifie que le dictionnaire reçu reste inchangé.
    """

    configuration = (
        create_raw_promotion_config()
    )

    original_champion_path = (
        configuration["champion_file"]
    )

    normalize_promotion_paths(
        configuration
    )

    assert configuration[
        "champion_file"
    ] == original_champion_path

    assert isinstance(
        configuration["champion_file"],
        str,
    )


def test_missing_champion_file_is_rejected() -> None:
    """
    Vérifie que champion_file est obligatoire.
    """

    configuration = (
        create_raw_promotion_config()
    )

    del configuration["champion_file"]

    with pytest.raises(
        ValueError,
        match="champion_file",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_missing_test_data_section_is_rejected(
) -> None:
    """
    Vérifie que test_data est obligatoire.
    """

    configuration = (
        create_raw_promotion_config()
    )

    del configuration["test_data"]

    with pytest.raises(
        ValueError,
        match="test_data",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_missing_source_section_is_rejected() -> None:
    """
    Vérifie que source est obligatoire.
    """

    configuration = (
        create_raw_promotion_config()
    )

    del configuration["source"]

    with pytest.raises(
        ValueError,
        match="source",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_missing_evaluation_section_is_rejected(
) -> None:
    """
    Vérifie que evaluation est obligatoire.
    """

    configuration = (
        create_raw_promotion_config()
    )

    del configuration["evaluation"]

    with pytest.raises(
        ValueError,
        match="evaluation",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_missing_output_section_is_rejected() -> None:
    """
    Vérifie que output est obligatoire.
    """

    configuration = (
        create_raw_promotion_config()
    )

    del configuration["output"]

    with pytest.raises(
        ValueError,
        match="output",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_test_data_section_must_be_dictionary(
) -> None:
    """
    Vérifie que test_data doit être un objet.
    """

    configuration = (
        create_raw_promotion_config()
    )

    configuration["test_data"] = "invalid"

    with pytest.raises(
        ValueError,
        match="test_data",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_source_section_must_be_dictionary() -> None:
    """
    Vérifie que source doit être un objet.
    """

    configuration = (
        create_raw_promotion_config()
    )

    configuration["source"] = "invalid"

    with pytest.raises(
        ValueError,
        match="source",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_output_section_must_be_dictionary() -> None:
    """
    Vérifie que output doit être un objet.
    """

    configuration = (
        create_raw_promotion_config()
    )

    configuration["output"] = "invalid"

    with pytest.raises(
        ValueError,
        match="output",
    ):
        normalize_promotion_paths(
            configuration
        )


def test_load_valid_promotion_config(
    tmp_path: Path,
) -> None:
    """
    Vérifie le chargement complet de la configuration.
    """

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        VALID_PROMOTION_YAML,
    )

    config = load_promotion_config(
        configuration_path
    )

    assert config.champion_file.is_absolute()

    assert (
        config.test_data.features_file
        .is_absolute()
    )

    assert (
        config.test_data.target_file
        .is_absolute()
    )

    assert (
        config.source.candidates_directory
        .is_absolute()
    )

    assert (
        config.source.preprocessor_file
        .is_absolute()
    )

    assert (
        config.output.production_directory
        .is_absolute()
    )

    assert (
        config.output.evaluation_directory
        .is_absolute()
    )

    assert config.test_data.target_column == (
        "quality_label"
    )

    assert (
        config.evaluation.minimum_f1_macro
        == pytest.approx(0.80)
    )

    assert (
        config.evaluation.minimum_poor_recall
        == pytest.approx(0.85)
    )


def test_missing_promotion_section_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la section promotion est obligatoire.
    """

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        """
other_section:
  value: 1
""",
    )

    with pytest.raises(
        ValueError,
        match="section 'promotion'",
    ):
        load_promotion_config(
            configuration_path
        )


def test_promotion_section_must_be_dictionary(
    tmp_path: Path,
) -> None:
    """
    Vérifie que promotion doit être un objet.
    """

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        """
promotion: invalid
""",
    )

    with pytest.raises(
        ValueError,
        match="doit être un objet",
    ):
        load_promotion_config(
            configuration_path
        )


def test_invalid_f1_threshold_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie la validation Pydantic du seuil F1.
    """

    invalid_yaml = (
        VALID_PROMOTION_YAML.replace(
            "minimum_f1_macro: 0.80",
            "minimum_f1_macro: 1.50",
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_promotion_config(
            configuration_path
        )


def test_invalid_poor_recall_threshold_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie la validation du seuil recall POOR.
    """

    invalid_yaml = (
        VALID_PROMOTION_YAML.replace(
            "minimum_poor_recall: 0.85",
            "minimum_poor_recall: -0.20",
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_promotion_config(
            configuration_path
        )


def test_unknown_promotion_field_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    invalid_yaml = (
        VALID_PROMOTION_YAML.replace(
            (
                '  champion_file: '
                '"reports/evaluation/'
                'candidates/champion.json"'
            ),
            (
                '  champion_file: '
                '"reports/evaluation/'
                'candidates/champion.json"\n'
                "  unknown_field: unexpected"
            ),
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "promotion.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_promotion_config(
            configuration_path
        )