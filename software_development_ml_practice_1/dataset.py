"""
Module for data ingestion and storage.
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from software_development_ml_practice_1.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
# Importing the function we just created in features.py
from software_development_ml_practice_1.features import preprocess_features

app = typer.Typer()

def test_function() -> None:
    """
    Test function to verify module configuration.

    Returns:
        None: This function does not return any value.
    """
    return

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path_x: Path = PROCESSED_DATA_DIR / "X_processed.csv",
    output_path_y: Path = PROCESSED_DATA_DIR / "y_processed.csv",
    force_download: bool = False,
):
    """
    Downloads raw data, applies feature engineering, and saves the results.

    If the raw data file does not exist locally, it downloads it from 
    HuggingFace. Afterward, it processes the data and saves the feature 
    matrix (X) and the target vector (y) into the processed data directory.

    Args:
        input_path (Path): Path where the raw sample will be saved or read from.
        output_path_x (Path): Path to save the processed features (X).
        output_path_y (Path): Path to save the processed target vector (y).
        force_download (bool): If True, forces the download from HuggingFace, 
            ignoring any existing local file.
    """
    DATASET_URI = "hf://datasets/juliensimon/sentry-impact-risk/data/sentry_impact_risk.parquet"
    
    # ---- 1. DOWNLOAD DATA ----
    if input_path.exists() and not force_download:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded local sample with {len(df)} rows from {input_path}")
    else:
        logger.info(f"Downloading dataset from {DATASET_URI}...")
        df = pd.read_parquet(DATASET_URI)
        
        df = df.sample(n=min(500, len(df)), random_state=42)
        df.to_csv(input_path, index=False)
        logger.success(f"Downloaded dataset and saved {len(df)} rows to {input_path}")
        
    # ---- 2. PROCESS DATA ----
    logger.info("Starting Feature Engineering...")
    X, y = preprocess_features(df)
    
    # ---- 3. SAVE PROCESSED DATA ----
    # We save the data so the model can be trained later without rerunning this script
    X.to_csv(output_path_x, index=False)
    y.to_csv(output_path_y, index=False)
    logger.success(f"Processed data saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    app()