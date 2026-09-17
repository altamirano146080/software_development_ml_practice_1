from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from software_development_ml_practice_1.config import PROCESSED_DATA_DIR

app = typer.Typer()

DATASET_PATH = PROCESSED_DATA_DIR / "dataset.csv"
FEATURES_PATH = PROCESSED_DATA_DIR / "features.csv"
LABELS_PATH = PROCESSED_DATA_DIR / "labels.csv"


def create_features(
    input_path: Path = DATASET_PATH,
    features_path: Path = FEATURES_PATH,
    labels_path: Path = LABELS_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Creates numerical features and the transformed target variable.
    """

    dataframe = pd.read_csv(input_path)

    dataframe = dataframe.dropna().copy()

    dataframe["log_impact_probability"] = np.log10(
        dataframe["impact_probability"]
    )

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    numeric_columns.remove("impact_probability")
    numeric_columns.remove("log_impact_probability")

    features = dataframe[numeric_columns]
    labels = dataframe["log_impact_probability"]

    features_path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(features_path, index=False)
    labels.to_frame(name="log_impact_probability").to_csv(
        labels_path,
        index=False,
    )

    logger.info(f"Selected features: {numeric_columns}")
    logger.info(f"Number of features: {features.shape[1]}")
    logger.success(f"Features saved to {features_path}")
    logger.success(f"Labels saved to {labels_path}")

    return features, labels


@app.command()
def main():
    """
    Generates the feature and label files.
    """

    create_features()


if __name__ == "__main__":
    app()