from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes.inference import (
    router as inference_router,
)
from app.api.routes.monitoring import (
    router as monitoring_router,
)
from app.core.alert_schemas import (
    AlertThresholdConfig,
)
from app.core.config import settings
from app.core.monitoring_config import (
    load_monitoring_config,
)
from app.services.alert_service import (
    AlertService,
)
from app.services.drift_detector import (
    DriftDetector,
)
from app.services.inference_service import (
    InferenceService,
)
from app.services.prediction_logger import (
    PredictionLogger,
)


DEFAULT_PRODUCTION_DIRECTORY = (
    settings.project_root
    / "models"
    / "production"
)


def create_application(
    production_directory: Path | None = None,
    monitoring_directory: Path | None = None,
    reference_dataset_path: Path | None = None,
    drift_report_directory: Path | None = None,
    alert_directory: Path | None = None,
) -> FastAPI:
    """
    Construit l'application FastAPI.

    La configuration du monitoring est chargée
    depuis config/monitoring.yaml.

    Les chemins reçus comme paramètres sont prioritaires.
    Cette possibilité est utilisée par les tests
    d'intégration pour isoler les fichiers dans tmp_path.
    """

    monitoring_config = load_monitoring_config()

    if production_directory is None:
        selected_production_directory = (
            DEFAULT_PRODUCTION_DIRECTORY
        )
    else:
        selected_production_directory = (
            production_directory.resolve()
        )

    if monitoring_directory is None:
        selected_monitoring_directory = (
            monitoring_config
            .prediction_log_directory
        )
    else:
        selected_monitoring_directory = (
            monitoring_directory.resolve()
        )

    if reference_dataset_path is None:
        selected_reference_dataset_path = (
            monitoring_config
            .drift
            .reference_dataset_path
        )
    else:
        selected_reference_dataset_path = (
            reference_dataset_path.resolve()
        )

    if drift_report_directory is None:
        selected_drift_report_directory = (
            monitoring_config
            .drift
            .report_directory
        )
    else:
        selected_drift_report_directory = (
            drift_report_directory.resolve()
        )

    if alert_directory is None:
        selected_alert_directory = (
            monitoring_config
            .alerts
            .alert_directory
        )
    else:
        selected_alert_directory = (
            alert_directory.resolve()
        )

    alert_thresholds = AlertThresholdConfig(
        warning_failure_rate=(
            monitoring_config
            .alerts
            .warning_failure_rate
        ),
        critical_failure_rate=(
            monitoring_config
            .alerts
            .critical_failure_rate
        ),
        warning_minimum_confidence=(
            monitoring_config
            .alerts
            .warning_minimum_confidence
        ),
        critical_minimum_confidence=(
            monitoring_config
            .alerts
            .critical_minimum_confidence
        ),
        warning_latency_seconds=(
            monitoring_config
            .alerts
            .warning_latency_seconds
        ),
        critical_latency_seconds=(
            monitoring_config
            .alerts
            .critical_latency_seconds
        ),
    )

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator:
        """
        Initialise les services au démarrage
        et libère leurs références à l'arrêt.
        """

        inference_service = InferenceService(
            selected_production_directory
        )

        prediction_logger = PredictionLogger(
            selected_monitoring_directory
        )

        drift_detector = DriftDetector(
            reference_dataset_path=(
                selected_reference_dataset_path
            ),
            prediction_logger=(
                prediction_logger
            ),
            minimum_production_records=(
                monitoring_config
                .drift
                .minimum_production_records
            ),
            warning_threshold=(
                monitoring_config
                .drift
                .warning_threshold
            ),
            drift_threshold=(
                monitoring_config
                .drift
                .drift_threshold
            ),
            bin_count=(
                monitoring_config
                .drift
                .bin_count
            ),
        )

        alert_service = AlertService(
            thresholds=alert_thresholds,
            alert_directory=(
                selected_alert_directory
            ),
        )

        application.state.inference_service = (
            inference_service
        )

        application.state.prediction_logger = (
            prediction_logger
        )

        application.state.drift_detector = (
            drift_detector
        )

        application.state.drift_report_directory = (
            selected_drift_report_directory
        )

        application.state.alert_service = (
            alert_service
        )

        application.state.monitoring_config = (
            monitoring_config
        )

        application.state.startup_error = None

        try:
            inference_service.load_artifacts()

        except (
            FileNotFoundError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as error:
            application.state.startup_error = str(
                error
            )

        yield

        application.state.inference_service = None
        application.state.prediction_logger = None
        application.state.drift_detector = None

        application.state.drift_report_directory = (
            None
        )

        application.state.alert_service = None
        application.state.monitoring_config = None

    application = FastAPI(
        title="AI Model Operations Center",
        description=(
            "API de classification, de monitoring, "
            "de détection de dérive et d'alertes MLOps."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.include_router(
        inference_router
    )

    application.include_router(
        monitoring_router
    )

    @application.get(
        "/",
        tags=[
            "System",
        ],
        summary="Informations générales du service",
    )
    def root() -> dict[str, str]:
        """
        Retourne les routes principales de l'API.
        """

        return {
            "service": (
                "AI Model Operations Center"
            ),
            "status": "running",
            "documentation": "/docs",
            "health": "/api/v1/health",
            "model": "/api/v1/model",
            "prediction": "/api/v1/predict",
            "monitoring_summary": (
                "/api/v1/monitoring/summary"
            ),
            "prediction_history": (
                "/api/v1/monitoring/predictions"
            ),
            "drift_analysis": (
                "/api/v1/monitoring/drift"
            ),
            "drift_report_generation": (
                "/api/v1/monitoring/drift/report"
            ),
            "alerts": (
                "/api/v1/monitoring/alerts"
            ),
            "alert_evaluation": (
                "/api/v1/monitoring/"
                "alerts/evaluate"
            ),
        }

    return application


app = create_application()