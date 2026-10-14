from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.monitoring_config import (
    load_monitoring_config,
    load_monitoring_yaml,
    normalize_monitoring_paths,
    resolve_monitoring_path,
)


VALID_MONITORING_YAML = """
monitoring:
  prediction_log_directory: "data/monitoring"

  drift:
    reference_dataset_path: "data/raw/dataset_quality.csv"
    minimum_production_records: 30
    warning_threshold: 0.10
    drift_threshold: 0.25
    bin_count: 10
    report_directory: "reports/drift"

  alerts:
    warning_failure_rate: 0.05
    critical_failure_rate: 0.15
    warning_minimum_confidence: 0.70
    critical_minimum_confidence: 0.50
    warning_latency_seconds: 0.20
    critical_latency_seconds: 1.00
    alert_directory: "data/monitoring/alerts"
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


def create_raw_monitoring_config() -> dict:
    """
    Crée une configuration avant normalisation.
    """

    return {
        "prediction_log_directory": (
            "data/monitoring"
        ),
        "drift": {
            "reference_dataset_path": (
                "data/raw/dataset_quality.csv"
            ),
            "minimum_production_records": 30,
            "warning_threshold": 0.10,
            "drift_threshold": 0.25,
            "bin_count": 10,
            "report_directory": (
                "reports/drift"
            ),
        },
        "alerts": {
            "warning_failure_rate": 0.05,
            "critical_failure_rate": 0.15,
            "warning_minimum_confidence": 0.70,
            "critical_minimum_confidence": 0.50,
            "warning_latency_seconds": 0.20,
            "critical_latency_seconds": 1.00,
            "alert_directory": (
                "data/monitoring/alerts"
            ),
        },
    }


def test_resolve_relative_monitoring_path() -> None:
    """
    Vérifie qu'un chemin relatif devient absolu.
    """

    resolved_path = resolve_monitoring_path(
        "data/monitoring"
    )

    assert resolved_path.is_absolute()
    assert resolved_path.name == "monitoring"


def test_resolve_absolute_monitoring_path(
    tmp_path: Path,
) -> None:
    """
    Vérifie qu'un chemin absolu est conservé.
    """

    absolute_path = (
        tmp_path
        / "data"
        / "monitoring"
    )

    resolved_path = resolve_monitoring_path(
        absolute_path
    )

    assert resolved_path == absolute_path.resolve()


def test_load_valid_monitoring_yaml(
    tmp_path: Path,
) -> None:
    """
    Vérifie la lecture d'un fichier YAML valide.
    """

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        VALID_MONITORING_YAML,
    )

    raw_configuration = load_monitoring_yaml(
        configuration_path
    )

    assert isinstance(
        raw_configuration,
        dict,
    )

    assert "monitoring" in raw_configuration


def test_missing_monitoring_yaml_is_rejected(
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
        load_monitoring_yaml(
            missing_path
        )


def test_monitoring_yaml_path_must_be_file(
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
        load_monitoring_yaml(
            directory_path
        )


def test_monitoring_yaml_root_must_be_dictionary(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la racine YAML doit être un objet.
    """

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        """
- item_one
- item_two
""",
    )

    with pytest.raises(
        ValueError,
        match="objet à sa racine",
    ):
        load_monitoring_yaml(
            configuration_path
        )


def test_normalize_monitoring_paths() -> None:
    """
    Vérifie la normalisation de tous les chemins.
    """

    configuration = (
        create_raw_monitoring_config()
    )

    result = normalize_monitoring_paths(
        configuration
    )

    assert isinstance(
        result,
        dict,
    )

    assert result[
        "prediction_log_directory"
    ].is_absolute()

    assert result[
        "drift"
    ][
        "reference_dataset_path"
    ].is_absolute()

    assert result[
        "drift"
    ][
        "report_directory"
    ].is_absolute()

    assert result[
        "alerts"
    ][
        "alert_directory"
    ].is_absolute()

    assert result[
        "drift"
    ][
        "minimum_production_records"
    ] == 30

    assert result[
        "alerts"
    ][
        "warning_failure_rate"
    ] == pytest.approx(0.05)


def test_normalization_does_not_modify_original_config(
) -> None:
    """
    Vérifie que la configuration d'origine
    reste inchangée.
    """

    configuration = (
        create_raw_monitoring_config()
    )

    normalize_monitoring_paths(
        configuration
    )

    assert isinstance(
        configuration[
            "prediction_log_directory"
        ],
        str,
    )

    assert isinstance(
        configuration[
            "drift"
        ][
            "reference_dataset_path"
        ],
        str,
    )

    assert isinstance(
        configuration[
            "alerts"
        ][
            "alert_directory"
        ],
        str,
    )


def test_missing_prediction_log_directory_is_rejected(
) -> None:
    """
    Vérifie que le dossier des journaux
    est obligatoire.
    """

    configuration = (
        create_raw_monitoring_config()
    )

    del configuration[
        "prediction_log_directory"
    ]

    with pytest.raises(
        ValueError,
        match="prediction_log_directory",
    ):
        normalize_monitoring_paths(
            configuration
        )


def test_missing_drift_section_is_rejected() -> None:
    """
    Vérifie que la section drift est obligatoire.
    """

    configuration = (
        create_raw_monitoring_config()
    )

    del configuration["drift"]

    with pytest.raises(
        ValueError,
        match="drift",
    ):
        normalize_monitoring_paths(
            configuration
        )


def test_missing_alerts_section_is_rejected() -> None:
    """
    Vérifie que la section alerts est obligatoire.
    """

    configuration = (
        create_raw_monitoring_config()
    )

    del configuration["alerts"]

    with pytest.raises(
        ValueError,
        match="alerts",
    ):
        normalize_monitoring_paths(
            configuration
        )


def test_drift_section_must_be_dictionary() -> None:
    """
    Vérifie que drift doit être un objet.
    """

    configuration = (
        create_raw_monitoring_config()
    )

    configuration["drift"] = "invalid"

    with pytest.raises(
        ValueError,
        match="section 'drift'",
    ):
        normalize_monitoring_paths(
            configuration
        )


def test_alerts_section_must_be_dictionary() -> None:
    """
    Vérifie que alerts doit être un objet.
    """

    configuration = (
        create_raw_monitoring_config()
    )

    configuration["alerts"] = "invalid"

    with pytest.raises(
        ValueError,
        match="section 'alerts'",
    ):
        normalize_monitoring_paths(
            configuration
        )


def test_load_valid_monitoring_config(
    tmp_path: Path,
) -> None:
    """
    Vérifie le chargement complet
    de la configuration.
    """

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        VALID_MONITORING_YAML,
    )

    config = load_monitoring_config(
        configuration_path
    )

    assert (
        config.prediction_log_directory
        .is_absolute()
    )

    assert (
        config.drift.reference_dataset_path
        .is_absolute()
    )

    assert (
        config.drift.report_directory
        .is_absolute()
    )

    assert (
        config.alerts.alert_directory
        .is_absolute()
    )

    assert (
        config.drift.minimum_production_records
        == 30
    )

    assert config.drift.bin_count == 10

    assert (
        config.drift.warning_threshold
        == pytest.approx(0.10)
    )

    assert (
        config.drift.drift_threshold
        == pytest.approx(0.25)
    )

    assert (
        config.alerts.warning_failure_rate
        == pytest.approx(0.05)
    )

    assert (
        config.alerts.critical_failure_rate
        == pytest.approx(0.15)
    )

    assert (
        config.alerts
        .warning_minimum_confidence
        == pytest.approx(0.70)
    )

    assert (
        config.alerts
        .critical_minimum_confidence
        == pytest.approx(0.50)
    )

    assert (
        config.alerts
        .warning_latency_seconds
        == pytest.approx(0.20)
    )

    assert (
        config.alerts
        .critical_latency_seconds
        == pytest.approx(1.00)
    )


def test_missing_monitoring_section_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que la section monitoring
    est obligatoire.
    """

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        """
other_section:
  value: 1
""",
    )

    with pytest.raises(
        ValueError,
        match="monitoring",
    ):
        load_monitoring_config(
            configuration_path
        )


def test_monitoring_section_must_be_dictionary(
    tmp_path: Path,
) -> None:
    """
    Vérifie que monitoring doit être un objet.
    """

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        """
monitoring: invalid
""",
    )

    with pytest.raises(
        ValueError,
        match="doit être un objet",
    ):
        load_monitoring_config(
            configuration_path
        )


def test_invalid_minimum_production_records_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie la validation du nombre minimum
    de prédictions.
    """

    invalid_yaml = (
        VALID_MONITORING_YAML.replace(
            "minimum_production_records: 30",
            "minimum_production_records: 0",
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_monitoring_config(
            configuration_path
        )


def test_invalid_bin_count_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie le nombre minimum de bins.
    """

    invalid_yaml = (
        VALID_MONITORING_YAML.replace(
            "bin_count: 10",
            "bin_count: 1",
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_monitoring_config(
            configuration_path
        )


def test_invalid_alert_threshold_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie la validation des seuils d'alerte.
    """

    invalid_yaml = (
        VALID_MONITORING_YAML.replace(
            "warning_failure_rate: 0.05",
            "warning_failure_rate: 1.50",
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_monitoring_config(
            configuration_path
        )


def test_unknown_monitoring_field_is_rejected(
    tmp_path: Path,
) -> None:
    """
    Vérifie que les champs inconnus sont refusés.
    """

    invalid_yaml = (
        VALID_MONITORING_YAML.replace(
            (
                '  prediction_log_directory: '
                '"data/monitoring"'
            ),
            (
                '  prediction_log_directory: '
                '"data/monitoring"\n'
                "  unknown_field: unexpected"
            ),
        )
    )

    configuration_path = write_yaml_file(
        tmp_path / "monitoring.yaml",
        invalid_yaml,
    )

    with pytest.raises(ValidationError):
        load_monitoring_config(
            configuration_path
        )