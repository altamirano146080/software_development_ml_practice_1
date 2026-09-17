from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from software_development_ml_practice_1.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

DATASET_URI = (
    "hf://datasets/juliensimon/sentry-impact-risk/"
    "data/sentry_impact_risk.parquet"
)

RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "dataset.csv"


def download_dataset(
    output_path: Path = RAW_DATA_PATH,
    sample_size: int = 500,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Descarga el dataset y guarda una muestra local en formato CSV.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_download:
        logger.info(f"Loading local dataset from {output_path}")
        return pd.read_csv(output_path)

    logger.info("Downloading dataset from Hugging Face...")

    dataframe = pd.read_parquet(DATASET_URI)

    dataframe = dataframe.sample(
        n=min(sample_size, len(dataframe)),
        random_state=42,
    )

    dataframe.to_csv(output_path, index=False)

    logger.success(
        f"Dataset downloaded and saved to {output_path} "
        f"with {len(dataframe)} rows."
    )

    return dataframe


def prepare_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """
    Removes incomplete rows and saves the cleaned dataset.
    """

    dataframe = pd.read_csv(input_path)

    original_size = len(dataframe)

    dataframe = dataframe.dropna().copy()

    cleaned_size = len(dataframe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    logger.info(f"Original dataset size: {original_size}")
    logger.info(f"Cleaned dataset size: {cleaned_size}")
    logger.success(f"Processed dataset saved to {output_path}")

    return dataframe


@app.command()
def main(
    force_download: bool = typer.Option(
        False,
        help="Download the dataset again even if a local copy exists.",
    ),
):
    """
    Downloads and prepares the asteroid impact-risk dataset.
    """

    download_dataset(force_download=force_download)
    prepare_dataset()


if __name__ == "__main__":
    app()
