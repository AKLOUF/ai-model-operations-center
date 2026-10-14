from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.core.promotion_schemas import (
    PromotionConfig,
)


DEFAULT_PROMOTION_CONFIG_PATH = (
    settings.project_root
    / "config"
    / "promotion.yaml"
)


def resolve_promotion_path(
    configured_path: str | Path,
) -> Path:
    """
    Résout un chemin de configuration.

    Les chemins relatifs sont résolus depuis
    la racine du projet.
    """

    path = Path(
        configured_path
    )

    if not path.is_absolute():
        path = (
            settings.project_root
            / path
        )

    return path.resolve()


def load_promotion_yaml(
    configuration_path: str | Path,
) -> dict[str, Any]:
    """
    Charge le fichier YAML de promotion.
    """

    resolved_path = Path(
        configuration_path
    ).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Configuration de promotion introuvable : "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "Le chemin de configuration ne correspond "
            f"pas à un fichier : {resolved_path}"
        )

    with resolved_path.open(
        mode="r",
        encoding="utf-8",
    ) as configuration_file:
        raw_configuration = yaml.safe_load(
            configuration_file
        )

    if not isinstance(
        raw_configuration,
        dict,
    ):
        raise ValueError(
            "La configuration YAML doit contenir "
            "un objet à sa racine."
        )

    return raw_configuration


def normalize_promotion_paths(
    promotion_configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    Convertit tous les chemins relatifs
    en chemins absolus.

    Le dictionnaire reçu n'est pas modifié.
    """

    required_sections = [
        "champion_file",
        "test_data",
        "source",
        "evaluation",
        "output",
    ]

    missing_sections = []

    for section_name in required_sections:
        if section_name not in promotion_configuration:
            missing_sections.append(
                section_name
            )

    if missing_sections:
        raise ValueError(
            "Élément(s) absent(s) de la configuration : "
            f"{missing_sections}"
        )

    test_data_configuration = (
        promotion_configuration[
            "test_data"
        ]
    )

    source_configuration = (
        promotion_configuration[
            "source"
        ]
    )

    output_configuration = (
        promotion_configuration[
            "output"
        ]
    )

    if not isinstance(
        test_data_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'test_data' doit être un objet."
        )

    if not isinstance(
        source_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'source' doit être un objet."
        )

    if not isinstance(
        output_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'output' doit être un objet."
        )

    normalized_test_data = {
        **test_data_configuration,
        "features_file": resolve_promotion_path(
            test_data_configuration[
                "features_file"
            ]
        ),
        "target_file": resolve_promotion_path(
            test_data_configuration[
                "target_file"
            ]
        ),
    }

    normalized_source = {
        **source_configuration,
        "candidates_directory": (
            resolve_promotion_path(
                source_configuration[
                    "candidates_directory"
                ]
            )
        ),
        "preprocessor_file": (
            resolve_promotion_path(
                source_configuration[
                    "preprocessor_file"
                ]
            )
        ),
    }

    normalized_output = {
        **output_configuration,
        "production_directory": (
            resolve_promotion_path(
                output_configuration[
                    "production_directory"
                ]
            )
        ),
        "evaluation_directory": (
            resolve_promotion_path(
                output_configuration[
                    "evaluation_directory"
                ]
            )
        ),
    }

    normalized_configuration = {
        **promotion_configuration,
        "champion_file": resolve_promotion_path(
            promotion_configuration[
                "champion_file"
            ]
        ),
        "test_data": normalized_test_data,
        "source": normalized_source,
        "output": normalized_output,
    }

    return normalized_configuration


def load_promotion_config(
    configuration_path: str | Path = (
        DEFAULT_PROMOTION_CONFIG_PATH
    ),
) -> PromotionConfig:
    """
    Charge, normalise et valide la configuration
    complète de promotion.
    """

    raw_configuration = load_promotion_yaml(
        configuration_path
    )

    if "promotion" not in raw_configuration:
        raise ValueError(
            "La section 'promotion' est absente "
            "de la configuration."
        )

    promotion_configuration = (
        raw_configuration[
            "promotion"
        ]
    )

    if not isinstance(
        promotion_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'promotion' "
            "doit être un objet."
        )

    normalized_configuration = (
        normalize_promotion_paths(
            promotion_configuration
        )
    )

    return PromotionConfig.model_validate(
        normalized_configuration
    )