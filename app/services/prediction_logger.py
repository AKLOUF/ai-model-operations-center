import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.monitoring_schemas import (
    PredictionLogRecord,
    PredictionMonitoringSummary,
)
from app.core.quality_schemas import (
    QualityLabel,
)


PREDICTION_LOG_FILENAME = "predictions.jsonl"


def utc_now_iso() -> str:
    """
    Retourne la date UTC actuelle au format ISO 8601.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def generate_prediction_id() -> str:
    """
    Génère un identifiant unique de prédiction.
    """

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    short_identifier = uuid4().hex[:8]

    return (
        f"pred_{timestamp}_"
        f"{short_identifier}"
    )


class PredictionLogger:
    """
    Service responsable de la journalisation
    et de la lecture des prédictions.
    """

    def __init__(
        self,
        monitoring_directory: Path,
    ) -> None:
        self.monitoring_directory = (
            monitoring_directory.resolve()
        )

    @property
    def prediction_log_path(self) -> Path:
        """
        Retourne le chemin du fichier JSON Lines.
        """

        return (
            self.monitoring_directory
            / PREDICTION_LOG_FILENAME
        )

    def log_record(
        self,
        record: PredictionLogRecord,
    ) -> Path:
        """
        Ajoute un événement au fichier JSONL.
        """

        self.monitoring_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_line = record.model_dump_json()

        with self.prediction_log_path.open(
            mode="a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                json_line
            )

            log_file.write(
                "\n"
            )

        return self.prediction_log_path

    def log_success(
        self,
        prediction: QualityLabel,
        confidence: float,
        model_name: str,
        model_version: str,
        latency_seconds: float,
        input_features: dict[
            str,
            float | int,
        ] | None = None,
    ) -> PredictionLogRecord:
        """
        Enregistre une prédiction réussie.

        Les features d'entrée sont facultatives afin
        de conserver la compatibilité avec les appels
        et les journaux précédents.
        """

        record = PredictionLogRecord(
            prediction_id=generate_prediction_id(),
            timestamp=utc_now_iso(),
            prediction=prediction,
            confidence=confidence,
            model_name=model_name,
            model_version=model_version,
            input_features=input_features,
            latency_seconds=latency_seconds,
            success=True,
            error_message=None,
        )

        self.log_record(
            record
        )

        return record

    def log_failure(
        self,
        latency_seconds: float,
        error_message: str,
    ) -> PredictionLogRecord:
        """
        Enregistre une tentative de prédiction échouée.
        """

        if not error_message.strip():
            raise ValueError(
                "Le message d'erreur "
                "ne peut pas être vide."
            )

        record = PredictionLogRecord(
            prediction_id=generate_prediction_id(),
            timestamp=utc_now_iso(),
            prediction=None,
            confidence=None,
            model_name=None,
            model_version=None,
            latency_seconds=latency_seconds,
            success=False,
            error_message=error_message,
        )

        self.log_record(
            record
        )

        return record

    def read_records(
        self,
    ) -> list[PredictionLogRecord]:
        """
        Charge tous les événements enregistrés.

        Un journal inexistant est considéré
        comme un journal vide.
        """

        if not self.prediction_log_path.exists():
            return []

        if not self.prediction_log_path.is_file():
            raise ValueError(
                "Le chemin du journal de prédictions "
                "ne correspond pas à un fichier."
            )

        records = []

        with self.prediction_log_path.open(
            mode="r",
            encoding="utf-8",
        ) as log_file:
            for line_number, line in enumerate(
                log_file,
                start=1,
            ):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    raw_record = json.loads(
                        stripped_line
                    )

                except json.JSONDecodeError as error:
                    raise ValueError(
                        "Ligne JSON invalide dans "
                        "le journal de prédictions : "
                        f"{line_number}"
                    ) from error

                record = (
                    PredictionLogRecord
                    .model_validate(
                        raw_record
                    )
                )

                records.append(
                    record
                )

        return records

    def build_summary(
        self,
    ) -> PredictionMonitoringSummary:
        """
        Calcule les métriques du journal.
        """

        records = self.read_records()

        total_count = len(
            records
        )

        successful_records = []

        for record in records:
            if record.success:
                successful_records.append(
                    record
                )

        successful_count = len(
            successful_records
        )

        failed_count = (
            total_count
            - successful_count
        )

        if total_count == 0:
            success_rate = 0.0
        else:
            success_rate = (
                successful_count
                / total_count
            )

        confidence_values = []

        for record in successful_records:
            if record.confidence is not None:
                confidence_values.append(
                    record.confidence
                )

        if confidence_values:
            average_confidence = (
                sum(confidence_values)
                / len(confidence_values)
            )
        else:
            average_confidence = None

        latency_values = []

        for record in records:
            latency_values.append(
                record.latency_seconds
            )

        if latency_values:
            average_latency = (
                sum(latency_values)
                / len(latency_values)
            )
        else:
            average_latency = None

        prediction_distribution = {}

        for record in successful_records:
            if record.prediction is None:
                continue

            prediction_label = (
                record.prediction.value
            )

            current_count = (
                prediction_distribution.get(
                    prediction_label,
                    0,
                )
            )

            prediction_distribution[
                prediction_label
            ] = current_count + 1

        model_versions = {}

        for record in successful_records:
            if (
                record.model_name is None
                or record.model_version is None
            ):
                continue

            model_identifier = (
                f"{record.model_name}:"
                f"{record.model_version}"
            )

            current_count = model_versions.get(
                model_identifier,
                0,
            )

            model_versions[
                model_identifier
            ] = current_count + 1

        return PredictionMonitoringSummary(
            total_prediction_count=total_count,
            successful_prediction_count=(
                successful_count
            ),
            failed_prediction_count=(
                failed_count
            ),
            success_rate=success_rate,
            average_confidence=(
                average_confidence
            ),
            average_latency_seconds=(
                average_latency
            ),
            prediction_distribution=(
                prediction_distribution
            ),
            model_versions=model_versions,
        )