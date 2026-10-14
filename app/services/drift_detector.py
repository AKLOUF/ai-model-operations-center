from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.drift_schemas import (
    DriftReport,
    FeatureDriftResult,
)
from app.services.inference_service import (
    FEATURE_NAMES,
)
from app.services.prediction_logger import (
    PredictionLogger,
)


DEFAULT_BIN_COUNT = 10
DEFAULT_WARNING_THRESHOLD = 0.10
DEFAULT_DRIFT_THRESHOLD = 0.25
DEFAULT_MINIMUM_PRODUCTION_RECORDS = 30

PSI_EPSILON = 1e-6


def utc_now_iso() -> str:
    """
    Retourne la date UTC actuelle au format ISO 8601.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def validate_thresholds(
    warning_threshold: float,
    drift_threshold: float,
) -> None:
    """
    Vérifie la cohérence des seuils PSI.
    """

    if warning_threshold < 0.0:
        raise ValueError(
            "Le seuil WARNING ne peut pas être négatif."
        )

    if drift_threshold <= warning_threshold:
        raise ValueError(
            "Le seuil DRIFTED doit être supérieur "
            "au seuil WARNING."
        )


def load_reference_features(
    reference_dataset_path: Path,
) -> pd.DataFrame:
    """
    Charge les features du dataset de référence.
    """

    resolved_path = (
        reference_dataset_path.resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Dataset de référence introuvable : "
            f"{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            "Le chemin du dataset de référence "
            "ne correspond pas à un fichier."
        )

    dataframe = pd.read_csv(
        resolved_path
    )

    if dataframe.empty:
        raise ValueError(
            "Le dataset de référence est vide."
        )

    missing_features = []

    for feature_name in FEATURE_NAMES:
        if feature_name not in dataframe.columns:
            missing_features.append(
                feature_name
            )

    if missing_features:
        raise ValueError(
            "Features absentes du dataset "
            f"de référence : {missing_features}"
        )

    reference_features = dataframe.loc[
        :,
        FEATURE_NAMES,
    ].copy()

    if reference_features.isna().any().any():
        raise ValueError(
            "Le dataset de référence contient "
            "des valeurs manquantes."
        )

    non_numeric_features = []

    for feature_name in FEATURE_NAMES:
        if not pd.api.types.is_numeric_dtype(
            reference_features[
                feature_name
            ]
        ):
            non_numeric_features.append(
                feature_name
            )

    if non_numeric_features:
        raise ValueError(
            "Features non numériques dans "
            f"la référence : {non_numeric_features}"
        )

    return reference_features


def load_production_features(
    prediction_logger: PredictionLogger,
) -> pd.DataFrame:
    """
    Extrait les features des prédictions réussies.

    Les anciens enregistrements qui ne possèdent pas
    input_features sont ignorés.
    """

    records = prediction_logger.read_records()

    rows = []

    for record in records:
        if not record.success:
            continue

        if record.input_features is None:
            continue

        missing_features = []

        for feature_name in FEATURE_NAMES:
            if (
                feature_name
                not in record.input_features
            ):
                missing_features.append(
                    feature_name
                )

        if missing_features:
            continue

        input_row = {}

        for feature_name in FEATURE_NAMES:
            input_row[feature_name] = (
                record.input_features[
                    feature_name
                ]
            )

        rows.append(
            input_row
        )

    if not rows:
        return pd.DataFrame(
            columns=FEATURE_NAMES
        )

    production_features = pd.DataFrame(
        rows,
        columns=FEATURE_NAMES,
    )

    for feature_name in FEATURE_NAMES:
        production_features[
            feature_name
        ] = pd.to_numeric(
            production_features[
                feature_name
            ],
            errors="coerce",
        )

    production_features = (
        production_features.dropna(
            subset=FEATURE_NAMES
        )
    )

    return production_features.reset_index(
        drop=True
    )


def calculate_quantile_bin_edges(
    reference_values: np.ndarray,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> np.ndarray:
    """
    Calcule des bornes basées sur les quantiles
    de la distribution de référence.
    """

    if bin_count < 2:
        raise ValueError(
            "Le nombre de bins doit être "
            "supérieur ou égal à deux."
        )

    values = np.asarray(
        reference_values,
        dtype=float,
    )

    if values.size == 0:
        raise ValueError(
            "La distribution de référence "
            "ne peut pas être vide."
        )

    quantiles = np.linspace(
        0.0,
        1.0,
        bin_count + 1,
    )

    bin_edges = np.quantile(
        values,
        quantiles,
    )

    bin_edges = np.unique(
        bin_edges
    )

    if bin_edges.size < 2:
        value = float(
            values[0]
        )

        delta = max(
            abs(value) * 0.01,
            1.0,
        )

        bin_edges = np.asarray(
            [
                value - delta,
                value + delta,
            ],
            dtype=float,
        )

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    return bin_edges


def calculate_psi(
    reference_values: np.ndarray,
    production_values: np.ndarray,
    bin_count: int = DEFAULT_BIN_COUNT,
) -> float:
    """
    Calcule le Population Stability Index.

    Les bins sont déterminés uniquement à partir
    de la distribution de référence.
    """

    reference_array = np.asarray(
        reference_values,
        dtype=float,
    )

    production_array = np.asarray(
        production_values,
        dtype=float,
    )

    if reference_array.size == 0:
        raise ValueError(
            "La référence ne peut pas être vide."
        )

    if production_array.size == 0:
        raise ValueError(
            "La production ne peut pas être vide."
        )

    bin_edges = calculate_quantile_bin_edges(
        reference_array,
        bin_count,
    )

    reference_counts, _ = np.histogram(
        reference_array,
        bins=bin_edges,
    )

    production_counts, _ = np.histogram(
        production_array,
        bins=bin_edges,
    )

    reference_proportions = (
        reference_counts
        / reference_counts.sum()
    )

    production_proportions = (
        production_counts
        / production_counts.sum()
    )

    reference_proportions = np.clip(
        reference_proportions,
        PSI_EPSILON,
        None,
    )

    production_proportions = np.clip(
        production_proportions,
        PSI_EPSILON,
        None,
    )

    psi_values = (
        production_proportions
        - reference_proportions
    ) * np.log(
        production_proportions
        / reference_proportions
    )

    return float(
        np.sum(
            psi_values
        )
    )


def calculate_normalized_mean_difference(
    reference_values: np.ndarray,
    production_values: np.ndarray,
) -> float:
    """
    Calcule l'écart absolu des moyennes,
    normalisé par l'écart-type de référence.
    """

    reference_array = np.asarray(
        reference_values,
        dtype=float,
    )

    production_array = np.asarray(
        production_values,
        dtype=float,
    )

    reference_mean = float(
        np.mean(
            reference_array
        )
    )

    production_mean = float(
        np.mean(
            production_array
        )
    )

    reference_standard_deviation = float(
        np.std(
            reference_array,
            ddof=0,
        )
    )

    mean_difference = abs(
        production_mean
        - reference_mean
    )

    if reference_standard_deviation == 0.0:
        if mean_difference == 0.0:
            return 0.0

        return mean_difference

    return (
        mean_difference
        / reference_standard_deviation
    )


def classify_drift_status(
    psi: float,
    warning_threshold: float,
    drift_threshold: float,
) -> str:
    """
    Classe une valeur PSI.
    """

    validate_thresholds(
        warning_threshold,
        drift_threshold,
    )

    if psi >= drift_threshold:
        return "DRIFTED"

    if psi >= warning_threshold:
        return "WARNING"

    return "STABLE"


def analyze_feature_drift(
    feature_name: str,
    reference_values: pd.Series,
    production_values: pd.Series,
    warning_threshold: float,
    drift_threshold: float,
    bin_count: int,
) -> FeatureDriftResult:
    """
    Calcule la dérive d'une feature.
    """

    reference_array = reference_values.to_numpy(
        dtype=float
    )

    production_array = production_values.to_numpy(
        dtype=float
    )

    psi = calculate_psi(
        reference_array,
        production_array,
        bin_count,
    )

    normalized_difference = (
        calculate_normalized_mean_difference(
            reference_array,
            production_array,
        )
    )

    status = classify_drift_status(
        psi,
        warning_threshold,
        drift_threshold,
    )

    return FeatureDriftResult(
        feature_name=feature_name,
        reference_count=len(
            reference_array
        ),
        production_count=len(
            production_array
        ),
        reference_mean=float(
            np.mean(
                reference_array
            )
        ),
        production_mean=float(
            np.mean(
                production_array
            )
        ),
        reference_standard_deviation=float(
            np.std(
                reference_array,
                ddof=0,
            )
        ),
        normalized_mean_difference=float(
            normalized_difference
        ),
        psi=psi,
        status=status,
    )


def build_insufficient_data_report(
    reference_dataset_path: Path,
    prediction_log_path: Path,
    production_record_count: int,
    minimum_production_records: int,
) -> DriftReport:
    """
    Construit un rapport lorsque trop peu
    de prédictions sont disponibles.
    """

    return DriftReport(
        generated_at=utc_now_iso(),
        reference_dataset_path=str(
            reference_dataset_path.resolve()
        ),
        prediction_log_path=str(
            prediction_log_path.resolve()
        ),
        minimum_production_records=(
            minimum_production_records
        ),
        production_record_count=(
            production_record_count
        ),
        analyzed_feature_count=0,
        drifted_feature_count=0,
        warning_feature_count=0,
        stable_feature_count=0,
        overall_status="INSUFFICIENT_DATA",
        features=[],
    )


def determine_overall_status(
    feature_results: list[
        FeatureDriftResult
    ],
) -> str:
    """
    Détermine le statut global du rapport.
    """

    statuses = {
        result.status
        for result in feature_results
    }

    if "DRIFTED" in statuses:
        return "DRIFTED"

    if "WARNING" in statuses:
        return "WARNING"

    return "STABLE"


class DriftDetector:
    """
    Service responsable de l'analyse
    de dérive des données.
    """

    def __init__(
        self,
        reference_dataset_path: Path,
        prediction_logger: PredictionLogger,
        minimum_production_records: int = (
            DEFAULT_MINIMUM_PRODUCTION_RECORDS
        ),
        warning_threshold: float = (
            DEFAULT_WARNING_THRESHOLD
        ),
        drift_threshold: float = (
            DEFAULT_DRIFT_THRESHOLD
        ),
        bin_count: int = DEFAULT_BIN_COUNT,
    ) -> None:
        if minimum_production_records < 1:
            raise ValueError(
                "Le nombre minimum de prédictions "
                "doit être supérieur ou égal à un."
            )

        if bin_count < 2:
            raise ValueError(
                "Le nombre de bins doit être "
                "supérieur ou égal à deux."
            )

        validate_thresholds(
            warning_threshold,
            drift_threshold,
        )

        self.reference_dataset_path = (
            reference_dataset_path.resolve()
        )

        self.prediction_logger = (
            prediction_logger
        )

        self.minimum_production_records = (
            minimum_production_records
        )

        self.warning_threshold = (
            warning_threshold
        )

        self.drift_threshold = (
            drift_threshold
        )

        self.bin_count = bin_count

    def analyze(self) -> DriftReport:
        """
        Exécute l'analyse complète de dérive.
        """

        reference_features = (
            load_reference_features(
                self.reference_dataset_path
            )
        )

        production_features = (
            load_production_features(
                self.prediction_logger
            )
        )

        production_record_count = len(
            production_features
        )

        if production_record_count < (
            self.minimum_production_records
        ):
            return build_insufficient_data_report(
                self.reference_dataset_path,
                self.prediction_logger
                .prediction_log_path,
                production_record_count,
                self.minimum_production_records,
            )

        feature_results = []

        for feature_name in FEATURE_NAMES:
            result = analyze_feature_drift(
                feature_name,
                reference_features[
                    feature_name
                ],
                production_features[
                    feature_name
                ],
                self.warning_threshold,
                self.drift_threshold,
                self.bin_count,
            )

            feature_results.append(
                result
            )

        drifted_feature_count = 0
        warning_feature_count = 0
        stable_feature_count = 0

        for result in feature_results:
            if result.status == "DRIFTED":
                drifted_feature_count += 1

            elif result.status == "WARNING":
                warning_feature_count += 1

            elif result.status == "STABLE":
                stable_feature_count += 1

        overall_status = determine_overall_status(
            feature_results
        )

        return DriftReport(
            generated_at=utc_now_iso(),
            reference_dataset_path=str(
                self.reference_dataset_path
            ),
            prediction_log_path=str(
                self.prediction_logger
                .prediction_log_path.resolve()
            ),
            minimum_production_records=(
                self.minimum_production_records
            ),
            production_record_count=(
                production_record_count
            ),
            analyzed_feature_count=len(
                feature_results
            ),
            drifted_feature_count=(
                drifted_feature_count
            ),
            warning_feature_count=(
                warning_feature_count
            ),
            stable_feature_count=(
                stable_feature_count
            ),
            overall_status=overall_status,
            features=feature_results,
        )