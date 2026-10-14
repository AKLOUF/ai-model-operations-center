from pathlib import Path
from typing import Any

import yaml

from app.core.config import settings
from app.core.monitoring_schemas import (
    MonitoringConfiguration,
)


DEFAULT_MONITORING_CONFIG_PATH = (
    settings.project_root
    / "config"
    / "monitoring.yaml"
)


def resolve_monitoring_path(
    configured_path: str | Path,
) -> Path:
    """
    Résout un chemin depuis la racine du projet.
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


def load_monitoring_yaml(
    configuration_path: str | Path,
) -> dict[str, Any]:
    """
    Charge le fichier YAML du monitoring.
    """

    resolved_path = Path(
        configuration_path
    ).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Configuration du monitoring "
            f"introuvable : {resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "Le chemin de configuration "
            "ne correspond pas à un fichier."
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


def normalize_monitoring_paths(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """
    Convertit les chemins relatifs
    en chemins absolus.
    """

    required_sections = [
        "prediction_log_directory",
        "drift",
        "alerts",
    ]

    missing_sections = []

    for section_name in required_sections:
        if section_name not in configuration:
            missing_sections.append(
                section_name
            )

    if missing_sections:
        raise ValueError(
            "Élément(s) absent(s) : "
            f"{missing_sections}"
        )

    drift_configuration = configuration[
        "drift"
    ]

    alert_configuration = configuration[
        "alerts"
    ]

    if not isinstance(
        drift_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'drift' doit être un objet."
        )

    if not isinstance(
        alert_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'alerts' doit être un objet."
        )

    normalized_configuration = {
        **configuration,
        "prediction_log_directory": (
            resolve_monitoring_path(
                configuration[
                    "prediction_log_directory"
                ]
            )
        ),
        "drift": {
            **drift_configuration,
            "reference_dataset_path": (
                resolve_monitoring_path(
                    drift_configuration[
                        "reference_dataset_path"
                    ]
                )
            ),
            "report_directory": (
                resolve_monitoring_path(
                    drift_configuration[
                        "report_directory"
                    ]
                )
            ),
        },
        "alerts": {
            **alert_configuration,
            "alert_directory": (
                resolve_monitoring_path(
                    alert_configuration[
                        "alert_directory"
                    ]
                )
            ),
        },
    }

    return normalized_configuration


def load_monitoring_config(
    configuration_path: str | Path = (
        DEFAULT_MONITORING_CONFIG_PATH
    ),
) -> MonitoringConfiguration:
    """
    Charge et valide la configuration du monitoring.
    """

    raw_configuration = load_monitoring_yaml(
        configuration_path
    )

    if "monitoring" not in raw_configuration:
        raise ValueError(
            "La section 'monitoring' est absente."
        )

    monitoring_configuration = (
        raw_configuration["monitoring"]
    )

    if not isinstance(
        monitoring_configuration,
        dict,
    ):
        raise ValueError(
            "La section 'monitoring' "
            "doit être un objet."
        )

    normalized_configuration = (
        normalize_monitoring_paths(
            monitoring_configuration
        )
    )

    return MonitoringConfiguration.model_validate(
        normalized_configuration
    )