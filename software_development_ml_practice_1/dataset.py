from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from software_development_ml_practice_1.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer(
    help="Download the Sentry Impact Risk dataset."
)

DATASET_URI = "hf://datasets/juliensimon/sentry-impact-risk/data/sentry_impact_risk.parquet"


def download_dataset(
    input_path: Path,
    force_download: bool = False
) -> pd.DataFrame:
    """Load the Sentry Impact Risk dataset.

    If a local CSV file already exists and ``force_download`` is ``False``,
    the local file is loaded. Otherwise, the dataset is downloaded from
    Hugging Face, a sample is selected, and the sample is saved locally.

    Parameters
    ----------
    input_path : pathlib.Path
        Path where the local raw dataset sample is stored.
    force_download : bool, default=False
        Whether to download the dataset even if a local copy exists.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    """
    if input_path.exists() and not force_download:
        logger.info("Loading local dataset from {}", input_path)
        df = pd.read_csv(input_path)
        logger.success("Loaded {} rows from local dataset", len(df))
        return df

    logger.info("Downloading dataset from Hugging Face")

    df = pd.read_parquet(DATASET_URI)

    df = df.sample(
        n=min(500, len(df)),
        random_state=42,
    )

    input_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(input_path, index=False)

    logger.success(
        "Downloaded dataset and saved {} rows to {}",
        len(df),
        input_path,
    )

    return df


def preprocess_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Preprocess the dataset for regression.

    Missing observations are removed and the ``impact_probability`` target
    is transformed using a base-10 logarithm. All remaining numerical
    columns are selected as features.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataset.

    Returns
    -------
    X : pandas.DataFrame
        Numerical feature matrix.
    y : pandas.Series
        Log-transformed target variable.
    features : list of str
        Names of the selected numerical features.

    Raises
    ------
    KeyError
        If ``impact_probability`` is not present in the input dataset.
    ValueError
        If ``impact_probability`` contains non-positive values, since the
        base-10 logarithm is only defined for positive values.
    """
    logger.info("Processing dataset...")

    df_reg = df.dropna().copy()

    logger.info(
        "Removed {} rows containing missing values",
        len(df) - len(df_reg),
    )

    if (df_reg["impact_probability"] <= 0).any():
        raise ValueError(
            "The 'impact_probability' column must contain only "
            "positive values."
        )

    logger.info("Transforming target variable")

    df_reg["log_impact_probability"] = np.log10(
        df_reg["impact_probability"]
    )

    logger.info("Selecting numerical features")

    features = df_reg.select_dtypes(include="number").columns.tolist()

    features.remove("impact_probability")
    features.remove("log_impact_probability")

    X = df_reg[features]
    y = df_reg["log_impact_probability"]

    logger.info("Selected features: {}", features)
    logger.info("Number of features: {}", X.shape[1])

    logger.success("Dataset preprocessing complete")

    return X, y, features


def save_processed_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> None:
    """Save the processed dataset to disk.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.
    output_path : pathlib.Path
        Destination path for the processed dataset.
    """
    processed_df = X.copy()
    processed_df["log_impact_probability"] = y

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    logger.success(
        "Saved processed dataset to {}",
        output_path,
    )


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    force_download: bool = False,
) -> None:
    """Download and preprocess the Sentry Impact Risk dataset.

    Parameters
    ----------
    input_path : pathlib.Path, default=RAW_DATA_DIR / "dataset.csv"
        Location of the local raw dataset sample.
    output_path : pathlib.Path, default=PROCESSED_DATA_DIR / "dataset.csv"
        Location where the processed dataset will be saved.
    force_download : bool, default=False
        If ``True``, download the dataset even when a local copy exists.

    Notes
    -----
    The processed target is stored in the ``log_impact_probability`` column.
    """ 
    df = download_dataset(
        input_path=input_path,
        force_download=force_download,
    )

    X, y, _ = preprocess_dataset(df)

    save_processed_dataset(
        X=X,
        y=y,
        output_path=output_path,
    )


if __name__ == "__main__":
    app()