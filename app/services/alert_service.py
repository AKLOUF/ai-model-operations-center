import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.alert_schemas import (
    AlertEvaluationReport,
    AlertThresholdConfig,
    MonitoringAlert,
)
from app.core.drift_schemas import (
    DriftReport,
)
from app.core.monitoring_schemas import (
    PredictionMonitoringSummary,
)


ALERT_REPORT_FILENAME = "alert_report.json"


def utc_now_iso() -> str:
    """
    Retourne la date UTC actuelle au format ISO 8601.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def generate_alert_id() -> str:
    """
    Génère un identifiant unique d'alerte.
    """

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    short_identifier = uuid4().hex[:8]

    return (
        f"alert_{timestamp}_"
        f"{short_identifier}"
    )


def determine_overall_severity(
    alerts: list[MonitoringAlert],
) -> str:
    """
    Détermine la sévérité globale.

    Priorité :
    CRITICAL, puis WARNING, puis INFO.
    """

    severities = {
        alert.severity
        for alert in alerts
    }

    if "CRITICAL" in severities:
        return "CRITICAL"

    if "WARNING" in severities:
        return "WARNING"

    return "INFO"


class AlertService:
    """
    Service responsable de l'évaluation
    et de la sauvegarde des alertes MLOps.
    """

    def __init__(
        self,
        thresholds: AlertThresholdConfig,
        alert_directory: Path,
    ) -> None:
        self.thresholds = thresholds

        self.alert_directory = (
            alert_directory.resolve()
        )

    @property
    def alert_report_path(self) -> Path:
        """
        Retourne le chemin du rapport d'alertes.
        """

        return (
            self.alert_directory
            / ALERT_REPORT_FILENAME
        )

    def evaluate_failure_rate(
        self,
        monitoring_summary: PredictionMonitoringSummary,
    ) -> MonitoringAlert | None:
        """
        Évalue le taux d'échec des prédictions.
        """

        if (
            monitoring_summary
            .total_prediction_count
            == 0
        ):
            return None

        failure_rate = (
            monitoring_summary
            .failed_prediction_count
            / monitoring_summary
            .total_prediction_count
        )

        if (
            failure_rate
            >= self.thresholds
            .critical_failure_rate
        ):
            severity = "CRITICAL"

            threshold = (
                self.thresholds
                .critical_failure_rate
            )

        elif (
            failure_rate
            >= self.thresholds
            .warning_failure_rate
        ):
            severity = "WARNING"

            threshold = (
                self.thresholds
                .warning_failure_rate
            )

        else:
            return None

        return MonitoringAlert(
            alert_id=generate_alert_id(),
            created_at=utc_now_iso(),
            severity=severity,
            category=(
                "PREDICTION_FAILURE_RATE"
            ),
            status="OPEN",
            title=(
                "Taux d'échec des prédictions élevé"
            ),
            message=(
                "Le taux d'échec des prédictions "
                "dépasse le seuil configuré."
            ),
            observed_value=float(
                failure_rate
            ),
            threshold_value=float(
                threshold
            ),
        )

    def evaluate_confidence(
        self,
        monitoring_summary: PredictionMonitoringSummary,
    ) -> MonitoringAlert | None:
        """
        Évalue la confiance moyenne du modèle.
        """

        average_confidence = (
            monitoring_summary
            .average_confidence
        )

        if average_confidence is None:
            return None

        if (
            average_confidence
            <= self.thresholds
            .critical_minimum_confidence
        ):
            severity = "CRITICAL"

            threshold = (
                self.thresholds
                .critical_minimum_confidence
            )

        elif (
            average_confidence
            <= self.thresholds
            .warning_minimum_confidence
        ):
            severity = "WARNING"

            threshold = (
                self.thresholds
                .warning_minimum_confidence
            )

        else:
            return None

        return MonitoringAlert(
            alert_id=generate_alert_id(),
            created_at=utc_now_iso(),
            severity=severity,
            category="LOW_CONFIDENCE",
            status="OPEN",
            title=(
                "Confiance moyenne insuffisante"
            ),
            message=(
                "La confiance moyenne des prédictions "
                "est inférieure au seuil configuré."
            ),
            observed_value=float(
                average_confidence
            ),
            threshold_value=float(
                threshold
            ),
        )

    def evaluate_latency(
        self,
        monitoring_summary: PredictionMonitoringSummary,
    ) -> MonitoringAlert | None:
        """
        Évalue la latence moyenne des prédictions.
        """

        average_latency = (
            monitoring_summary
            .average_latency_seconds
        )

        if average_latency is None:
            return None

        if (
            average_latency
            >= self.thresholds
            .critical_latency_seconds
        ):
            severity = "CRITICAL"

            threshold = (
                self.thresholds
                .critical_latency_seconds
            )

        elif (
            average_latency
            >= self.thresholds
            .warning_latency_seconds
        ):
            severity = "WARNING"

            threshold = (
                self.thresholds
                .warning_latency_seconds
            )

        else:
            return None

        return MonitoringAlert(
            alert_id=generate_alert_id(),
            created_at=utc_now_iso(),
            severity=severity,
            category="HIGH_LATENCY",
            status="OPEN",
            title=(
                "Latence moyenne élevée"
            ),
            message=(
                "La latence moyenne des prédictions "
                "dépasse le seuil configuré."
            ),
            observed_value=float(
                average_latency
            ),
            threshold_value=float(
                threshold
            ),
        )

    def evaluate_drift(
        self,
        drift_report: DriftReport,
    ) -> MonitoringAlert | None:
        """
        Évalue le statut global de dérive.
        """

        if (
            drift_report.overall_status
            == "INSUFFICIENT_DATA"
        ):
            return None

        if (
            drift_report.overall_status
            == "DRIFTED"
        ):
            severity = "CRITICAL"

        elif (
            drift_report.overall_status
            == "WARNING"
        ):
            severity = "WARNING"

        else:
            return None

        return MonitoringAlert(
            alert_id=generate_alert_id(),
            created_at=utc_now_iso(),
            severity=severity,
            category="DATA_DRIFT",
            status="OPEN",
            title=(
                "Dérive des données détectée"
            ),
            message=(
                "Le comportement des données "
                "de production diffère de la référence."
            ),
            observed_value=(
                drift_report.overall_status
            ),
            threshold_value="STABLE",
        )

    def evaluate(
        self,
        monitoring_summary: PredictionMonitoringSummary,
        drift_report: DriftReport,
    ) -> AlertEvaluationReport:
        """
        Exécute toutes les règles d'alerte.
        """

        alerts: list[MonitoringAlert] = []

        failure_alert = (
            self.evaluate_failure_rate(
                monitoring_summary
            )
        )

        if failure_alert is not None:
            alerts.append(
                failure_alert
            )

        confidence_alert = (
            self.evaluate_confidence(
                monitoring_summary
            )
        )

        if confidence_alert is not None:
            alerts.append(
                confidence_alert
            )

        latency_alert = self.evaluate_latency(
            monitoring_summary
        )

        if latency_alert is not None:
            alerts.append(
                latency_alert
            )

        drift_alert = self.evaluate_drift(
            drift_report
        )

        if drift_alert is not None:
            alerts.append(
                drift_alert
            )

        critical_alert_count = 0
        warning_alert_count = 0
        info_alert_count = 0

        for alert in alerts:
            if alert.severity == "CRITICAL":
                critical_alert_count += 1

            elif alert.severity == "WARNING":
                warning_alert_count += 1

            elif alert.severity == "INFO":
                info_alert_count += 1

        return AlertEvaluationReport(
            generated_at=utc_now_iso(),
            overall_severity=(
                determine_overall_severity(
                    alerts
                )
            ),
            alert_count=len(alerts),
            critical_alert_count=(
                critical_alert_count
            ),
            warning_alert_count=(
                warning_alert_count
            ),
            info_alert_count=(
                info_alert_count
            ),
            alerts=alerts,
        )

    def save_report(
        self,
        report: AlertEvaluationReport,
    ) -> Path:
        """
        Sauvegarde le rapport d'alertes en JSON.
        """

        self.alert_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.alert_report_path.write_text(
            report.model_dump_json(
                indent=2
            ),
            encoding="utf-8",
        )

        return self.alert_report_path

    def load_report(
        self,
    ) -> AlertEvaluationReport | None:
        """
        Charge le dernier rapport d'alertes.

        Retourne None lorsque le rapport
        n'existe pas encore.
        """

        if not self.alert_report_path.exists():
            return None

        if not self.alert_report_path.is_file():
            raise ValueError(
                "Le chemin du rapport d'alertes "
                "ne correspond pas à un fichier."
            )

        try:
            raw_content = json.loads(
                self.alert_report_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as error:
            raise ValueError(
                "Le rapport d'alertes "
                "ne contient pas un JSON valide."
            ) from error

        return AlertEvaluationReport.model_validate(
            raw_content
        )