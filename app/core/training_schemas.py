from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


ImputationStrategy = Literal[
    "mean",
    "median",
    "most_frequent",
]


class DataSplitConfig(BaseModel):
    """
    Configuration de la séparation du dataset.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    train_ratio: float = Field(
        default=0.70,
        gt=0.0,
        lt=1.0,
    )

    validation_ratio: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
    )

    test_ratio: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
    )

    random_seed: int = Field(
        default=42,
        ge=0,
    )

    stratify: bool = True

    @model_validator(mode="after")
    def validate_ratio_sum(
        self,
    ) -> "DataSplitConfig":
        """
        Vérifie que les trois proportions totalisent 1.
        """

        ratio_sum = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(
                "La somme des proportions train, "
                "validation et test doit être égale à 1."
            )

        return self


class PreprocessingConfig(BaseModel):
    """
    Configuration du prétraitement des features.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    imputation_strategy: ImputationStrategy = (
        "median"
    )

    scale_features: bool = True


class TrainingOutputConfig(BaseModel):
    """
    Configuration des dossiers de sortie.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    processed_data_directory: Path
    artifacts_directory: Path
    reports_directory: Path


class TrainingConfig(BaseModel):
    """
    Configuration complète du pipeline.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    dataset_path: Path

    target_column: str = Field(
        min_length=1,
    )

    split: DataSplitConfig
    preprocessing: PreprocessingConfig
    output: TrainingOutputConfig


class DatasetSplitSummary(BaseModel):
    """
    Résumé d'un ensemble train, validation ou test.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    row_count: int = Field(
        ge=0,
    )

    feature_count: int = Field(
        ge=0,
    )

    class_distribution: dict[str, int]


class DataPreparationReport(BaseModel):
    """
    Rapport final de préparation des données.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_dataset_path: str
    target_column: str
    random_seed: int
    stratified: bool
    feature_names: list[str]

    train: DatasetSplitSummary
    validation: DatasetSplitSummary
    test: DatasetSplitSummary