from pathlib import Path

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    status,
)

from app.core.alert_schemas import (
    AlertEvaluationReport,
)
from app.core.drift_schemas import (
    DriftReport,
)
from app.core.monitoring_schemas import (
    PredictionLogRecord,
    PredictionMonitoringSummary,
)
from app.services.alert_service import (
    AlertService,
)
from app.services.drift_detector import (
    DriftDetector,
)
from app.services.prediction_logger import (
    PredictionLogger,
)


DRIFT_REPORT_FILENAME = "drift_report.json"


router = APIRouter(
    prefix="/api/v1/monitoring",
    tags=[
        "Monitoring",
    ],
)


def get_prediction_logger(
    request: Request,
) -> PredictionLogger:
    """
    Récupère le journal de prédictions.
    """

    prediction_logger = getattr(
        request.app.state,
        "prediction_logger",
        None,
    )

    if prediction_logger is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Le service de monitoring "
                "n'est pas initialisé."
            ),
        )

    return prediction_logger


def get_drift_detector(
    request: Request,
) -> DriftDetector:
    """
    Récupère le détecteur de dérive.
    """

    drift_detector = getattr(
        request.app.state,
        "drift_detector",
        None,
    )

    if drift_detector is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Le détecteur de dérive "
                "n'est pas initialisé."
            ),
        )

    return drift_detector


def get_alert_service(
    request: Request,
) -> AlertService:
    """
    Récupère le moteur d'alertes.
    """

    alert_service = getattr(
        request.app.state,
        "alert_service",
        None,
    )

    if alert_service is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Le service d'alertes "
                "n'est pas initialisé."
            ),
        )

    return alert_service


def get_drift_report_directory(
    request: Request,
) -> Path:
    """
    Récupère le dossier des rapports de dérive.
    """

    report_directory = getattr(
        request.app.state,
        "drift_report_directory",
        None,
    )

    if report_directory is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Le dossier des rapports de dérive "
                "n'est pas configuré."
            ),
        )

    return Path(
        report_directory
    ).resolve()


@router.get(
    "/summary",
    response_model=PredictionMonitoringSummary,
    summary="Obtenir le résumé du monitoring",
)
def get_monitoring_summary(
    request: Request,
) -> PredictionMonitoringSummary:
    """
    Retourne les statistiques calculées
    à partir du journal de prédictions.
    """

    prediction_logger = get_prediction_logger(
        request
    )

    try:
        return prediction_logger.build_summary()

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error


@router.get(
    "/predictions",
    response_model=list[PredictionLogRecord],
    summary="Consulter les dernières prédictions",
)
def get_prediction_records(
    request: Request,
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
) -> list:
    """
    Retourne les dernières prédictions enregistrées.
    """

    prediction_logger = get_prediction_logger(
        request
    )

    try:
        records = prediction_logger.read_records()

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error

    if not records:
        return []

    return records[-limit:]


@router.get(
    "/drift",
    response_model=DriftReport,
    summary="Analyser la dérive des données",
)
def get_drift_report(
    request: Request,
) -> DriftReport:
    """
    Calcule un rapport de dérive sans le sauvegarder.
    """

    drift_detector = get_drift_detector(
        request
    )

    try:
        return drift_detector.analyze()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error


@router.post(
    "/drift/report",
    response_model=DriftReport,
    summary="Générer et sauvegarder le rapport de dérive",
)
def generate_drift_report(
    request: Request,
) -> DriftReport:
    """
    Calcule puis sauvegarde le rapport de dérive.
    """

    drift_detector = get_drift_detector(
        request
    )

    report_directory = (
        get_drift_report_directory(
            request
        )
    )

    try:
        report = drift_detector.analyze()

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_directory
        / DRIFT_REPORT_FILENAME
    )

    report_path.write_text(
        report.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    return report


@router.get(
    "/alerts",
    response_model=AlertEvaluationReport | None,
    summary="Obtenir le dernier rapport d'alertes",
)
def get_alert_report(
    request: Request,
) -> AlertEvaluationReport | None:
    """
    Retourne le dernier rapport d'alertes.

    Retourne null si aucun rapport
    n'a encore été généré.
    """

    alert_service = get_alert_service(
        request
    )

    try:
        return alert_service.load_report()

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error


@router.post(
    "/alerts/evaluate",
    response_model=AlertEvaluationReport,
    summary="Évaluer et sauvegarder les alertes MLOps",
)
def evaluate_monitoring_alerts(
    request: Request,
) -> AlertEvaluationReport:
    """
    Calcule les alertes à partir :

    1. du résumé des prédictions ;
    2. du rapport de dérive ;
    3. des seuils du moteur d'alertes.
    """

    prediction_logger = get_prediction_logger(
        request
    )

    drift_detector = get_drift_detector(
        request
    )

    alert_service = get_alert_service(
        request
    )

    try:
        monitoring_summary = (
            prediction_logger.build_summary()
        )

        drift_report = drift_detector.analyze()

        alert_report = alert_service.evaluate(
            monitoring_summary,
            drift_report,
        )

        alert_service.save_report(
            alert_report
        )

        return alert_report

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error