import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.inference_schemas import (
    DatasetQualityPredictionRequest,
    DatasetQualityPredictionResponse,
    InferenceHealthResponse,
    ProductionModelInformation,
)
from app.core.promotion_schemas import (
    ProductionModelRecord,
)
from app.core.quality_schemas import (
    QualityLabel,
)


PRODUCTION_MODEL_FILENAME = "model.joblib"
PRODUCTION_PREPROCESSOR_FILENAME = "preprocessor.joblib"
PRODUCTION_REGISTRY_FILENAME = "registry.json"


FEATURE_NAMES = [
    "row_count",
    "column_count",
    "missing_value_ratio",
    "duplicate_row_ratio",
    "invalid_type_ratio",
    "primary_key_uniqueness",
    "foreign_key_match_ratio",
    "schema_change_count",
    "outlier_ratio",
    "freshness_delay_hours",
]


class InferenceService:
    """
    Service responsable du chargement des artefacts
    et de l'exécution des prédictions.
    """

    def __init__(
        self,
        production_directory: Path,
    ) -> None:
        """
        Initialise le service.

        Les artefacts ne sont pas chargés automatiquement.
        La méthode load_artifacts doit être appelée.
        """

        self.production_directory = (
            production_directory.resolve()
        )

        self.model: Any = None
        self.preprocessor: Any = None

        self.production_record: (
            ProductionModelRecord | None
        ) = None

    @property
    def model_path(self) -> Path:
        """
        Retourne le chemin du modèle de production.
        """

        return (
            self.production_directory
            / PRODUCTION_MODEL_FILENAME
        )

    @property
    def preprocessor_path(self) -> Path:
        """
        Retourne le chemin du préprocesseur.
        """

        return (
            self.production_directory
            / PRODUCTION_PREPROCESSOR_FILENAME
        )

    @property
    def registry_path(self) -> Path:
        """
        Retourne le chemin du registre de production.
        """

        return (
            self.production_directory
            / PRODUCTION_REGISTRY_FILENAME
        )

    def load_model(self) -> Any:
        """
        Charge et valide le modèle de production.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Modèle de production introuvable : "
                f"{self.model_path}"
            )

        if not self.model_path.is_file():
            raise ValueError(
                "Le chemin du modèle de production "
                "ne correspond pas à un fichier."
            )

        model = joblib.load(
            self.model_path
        )

        if not hasattr(
            model,
            "predict",
        ):
            raise TypeError(
                "Le modèle de production ne possède "
                "pas de méthode predict."
            )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                "Le modèle de production ne possède "
                "pas de méthode predict_proba."
            )

        if not hasattr(
            model,
            "classes_",
        ):
            raise TypeError(
                "Le modèle de production ne contient "
                "pas l'attribut classes_."
            )

        self.model = model

        return model

    def load_preprocessor(self) -> Any:
        """
        Charge et valide le préprocesseur.
        """

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(
                "Préprocesseur de production introuvable : "
                f"{self.preprocessor_path}"
            )

        if not self.preprocessor_path.is_file():
            raise ValueError(
                "Le chemin du préprocesseur "
                "ne correspond pas à un fichier."
            )

        preprocessor = joblib.load(
            self.preprocessor_path
        )

        if not hasattr(
            preprocessor,
            "transform",
        ):
            raise TypeError(
                "Le préprocesseur ne possède pas "
                "de méthode transform."
            )

        self.preprocessor = preprocessor

        return preprocessor

    def load_production_record(
        self,
    ) -> ProductionModelRecord:
        """
        Charge et valide registry.json.
        """

        if not self.registry_path.exists():
            raise FileNotFoundError(
                "Registre de production introuvable : "
                f"{self.registry_path}"
            )

        if not self.registry_path.is_file():
            raise ValueError(
                "Le chemin du registre de production "
                "ne correspond pas à un fichier."
            )

        try:
            raw_content = json.loads(
                self.registry_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as error:
            raise ValueError(
                "Le registre de production "
                "ne contient pas un JSON valide."
            ) from error

        production_record = (
            ProductionModelRecord.model_validate(
                raw_content
            )
        )

        if production_record.status != "ACTIVE":
            raise ValueError(
                "Le modèle de production "
                "n'est pas actif."
            )

        self.production_record = (
            production_record
        )

        return production_record

    def load_artifacts(self) -> None:
        """
        Charge le modèle, le préprocesseur
        et le registre de production.
        """

        self.load_model()
        self.load_preprocessor()
        self.load_production_record()

    def artifacts_are_loaded(self) -> bool:
        """
        Vérifie que les trois artefacts sont chargés.
        """

        return (
            self.model is not None
            and self.preprocessor is not None
            and self.production_record is not None
        )

    def build_input_dataframe(
        self,
        request: DatasetQualityPredictionRequest,
    ) -> pd.DataFrame:
        """
        Convertit une requête Pydantic en DataFrame.

        L'ordre des colonnes est explicitement fixé
        afin de correspondre à l'entraînement.
        """

        request_data = request.model_dump()

        dataframe = pd.DataFrame(
            [
                request_data
            ],
            columns=FEATURE_NAMES,
        )

        if dataframe.shape != (1, 10):
            raise RuntimeError(
                "La requête d'inférence doit produire "
                "exactement une ligne et dix features."
            )

        if dataframe.isna().any().any():
            raise ValueError(
                "La requête d'inférence contient "
                "des valeurs manquantes."
            )

        return dataframe

    def transform_request(
        self,
        request: DatasetQualityPredictionRequest,
    ) -> pd.DataFrame:
        """
        Transforme la requête avec le préprocesseur
        entraîné sur X_train.
        """

        if self.preprocessor is None:
            raise RuntimeError(
                "Le préprocesseur n'est pas chargé."
            )

        raw_dataframe = self.build_input_dataframe(
            request
        )

        transformed_values = (
            self.preprocessor.transform(
                raw_dataframe
            )
        )

        transformed_array = np.asarray(
            transformed_values
        )

        if transformed_array.ndim != 2:
            raise RuntimeError(
                "La transformation doit produire "
                "une matrice à deux dimensions."
            )

        feature_names = FEATURE_NAMES

        if hasattr(
            self.preprocessor,
            "get_feature_names_out",
        ):
            feature_names = [
                str(feature_name)
                for feature_name
                in self.preprocessor
                .get_feature_names_out()
            ]

        if transformed_array.shape[1] != len(
            feature_names
        ):
            raise RuntimeError(
                "Le nombre de features transformées "
                "est incohérent."
            )

        return pd.DataFrame(
            transformed_array,
            columns=feature_names,
        )

    def predict(
        self,
        request: DatasetQualityPredictionRequest,
    ) -> DatasetQualityPredictionResponse:
        """
        Exécute une prédiction complète.
        """

        if not self.artifacts_are_loaded():
            raise RuntimeError(
                "Les artefacts de production "
                "ne sont pas tous chargés."
            )

        transformed_features = (
            self.transform_request(
                request
            )
        )

        raw_predictions = self.model.predict(
            transformed_features
        )

        raw_probabilities = (
            self.model.predict_proba(
                transformed_features
            )
        )

        predictions = np.asarray(
            raw_predictions,
            dtype=str,
        )

        probabilities = np.asarray(
            raw_probabilities,
            dtype=float,
        )

        if len(predictions) != 1:
            raise RuntimeError(
                "Le modèle doit produire "
                "exactement une prédiction."
            )

        if probabilities.shape[0] != 1:
            raise RuntimeError(
                "Le modèle doit produire "
                "une ligne de probabilités."
            )

        model_classes = [
            str(label)
            for label in self.model.classes_
        ]

        if probabilities.shape[1] != len(
            model_classes
        ):
            raise RuntimeError(
                "Le nombre de probabilités "
                "ne correspond pas aux classes."
            )

        predicted_label = predictions[0]

        allowed_labels = {
            label.value
            for label in QualityLabel
        }

        if predicted_label not in allowed_labels:
            raise RuntimeError(
                "Le modèle a produit un label inconnu : "
                f"{predicted_label}"
            )

        probability_dictionary = {}

        for index, label in enumerate(
            model_classes
        ):
            probability_dictionary[
                label
            ] = float(
                probabilities[0][index]
            )

        confidence = probability_dictionary[
            predicted_label
        ]

        production_record = (
            self.production_record
        )

        if production_record is None:
            raise RuntimeError(
                "Les métadonnées de production "
                "ne sont pas chargées."
            )

        return DatasetQualityPredictionResponse(
            prediction=QualityLabel(
                predicted_label
            ),
            confidence=confidence,
            probabilities=probability_dictionary,
            model_name=(
                production_record.model_name
            ),
            model_version=(
                production_record.model_version
            ),
        )

    def get_model_information(
        self,
    ) -> ProductionModelInformation:
        """
        Retourne les informations publiques
        du modèle actif.
        """

        if self.production_record is None:
            raise RuntimeError(
                "Le registre de production "
                "n'est pas chargé."
            )

        return ProductionModelInformation(
            model_name=(
                self.production_record.model_name
            ),
            model_version=(
                self.production_record.model_version
            ),
            status=(
                self.production_record.status
            ),
            promoted_at=(
                self.production_record.promoted_at
            ),
            evaluated_split=(
                self.production_record.evaluated_split
            ),
            accuracy=(
                self.production_record.accuracy
            ),
            f1_macro=(
                self.production_record.f1_macro
            ),
            poor_recall=(
                self.production_record.poor_recall
            ),
        )

    def get_health(
        self,
    ) -> InferenceHealthResponse:
        """
        Retourne l'état du service.
        """

        model_loaded = (
            self.model is not None
        )

        preprocessor_loaded = (
            self.preprocessor is not None
        )

        registry_loaded = (
            self.production_record is not None
        )

        if (
            model_loaded
            and preprocessor_loaded
            and registry_loaded
        ):
            status = "healthy"
        else:
            status = "unhealthy"

        return InferenceHealthResponse(
            status=status,
            model_loaded=model_loaded,
            preprocessor_loaded=(
                preprocessor_loaded
            ),
        )