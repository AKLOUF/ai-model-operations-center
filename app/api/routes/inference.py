from time import perf_counter

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)

from app.core.inference_schemas import (
    DatasetQualityPredictionRequest,
    DatasetQualityPredictionResponse,
    InferenceHealthResponse,
    ProductionModelInformation,
)
from app.services.inference_service import (
    InferenceService,
)
from app.services.prediction_logger import (
    PredictionLogger,
)


router = APIRouter(
    prefix="/api/v1",
    tags=[
        "Inference",
    ],
)


def get_inference_service(
    request: Request,
) -> InferenceService:
    """
    Récupère le service d'inférence
    depuis l'état de l'application.
    """

    service = getattr(
        request.app.state,
        "inference_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Le service d'inférence "
                "n'est pas initialisé."
            ),
        )

    return service


def get_prediction_logger(
    request: Request,
) -> PredictionLogger:
    """
    Récupère le journal de prédictions
    depuis l'état de l'application.
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


@router.get(
    "/health",
    response_model=InferenceHealthResponse,
    summary="Vérifier l'état du service",
)
def get_health(
    request: Request,
) -> InferenceHealthResponse:
    """
    Retourne l'état du modèle
    et du préprocesseur.
    """

    service = get_inference_service(
        request
    )

    return service.get_health()


@router.get(
    "/model",
    response_model=ProductionModelInformation,
    summary="Obtenir les informations du modèle actif",
)
def get_model_information(
    request: Request,
) -> ProductionModelInformation:
    """
    Retourne les informations du modèle
    actuellement actif en production.
    """

    service = get_inference_service(
        request
    )

    if not service.artifacts_are_loaded():
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Les artefacts de production "
                "ne sont pas chargés."
            ),
        )

    try:
        return service.get_model_information()

    except RuntimeError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(error),
        ) from error


@router.post(
    "/predict",
    response_model=DatasetQualityPredictionResponse,
    summary="Prédire la qualité d'un dataset",
)
def predict_dataset_quality(
    request_body: DatasetQualityPredictionRequest,
    request: Request,
) -> DatasetQualityPredictionResponse:
    """
    Exécute une prédiction et journalise
    son résultat.
    """

    service = get_inference_service(
        request
    )

    prediction_logger = get_prediction_logger(
        request
    )

    prediction_start = perf_counter()

    if not service.artifacts_are_loaded():
        latency_seconds = (
            perf_counter()
            - prediction_start
        )

        prediction_logger.log_failure(
            latency_seconds=latency_seconds,
            error_message=(
                "Les artefacts de production "
                "ne sont pas chargés."
            ),
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Les artefacts de production "
                "ne sont pas chargés."
            ),
        )

    try:
        response = service.predict(
            request_body
        )

        latency_seconds = (
            perf_counter()
            - prediction_start
        )

        prediction_logger.log_success(
            prediction=response.prediction,
            confidence=response.confidence,
            model_name=response.model_name,
            model_version=response.model_version,
            latency_seconds=latency_seconds,
            input_features=request_body.model_dump(),
        )

        return response

    except ValueError as error:
        latency_seconds = (
            perf_counter()
            - prediction_start
        )

        prediction_logger.log_failure(
            latency_seconds=latency_seconds,
            error_message=str(error),
        )

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(error),
        ) from error

    except (
        RuntimeError,
        TypeError,
    ) as error:
        latency_seconds = (
            perf_counter()
            - prediction_start
        )

        prediction_logger.log_failure(
            latency_seconds=latency_seconds,
            error_message=str(error),
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=str(error),
        ) from error