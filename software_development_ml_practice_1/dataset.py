from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer
import numpy as np

from software_development_ml_practice_1.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from pathlib import Path


app = typer.Typer()

def funcion_prueba():
    """
    Funcion prueba
    """
    return

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    force_download: bool = False,
):

    # ---- DOWNLOAD DATA ----
    DATASET_URI = "hf://datasets/juliensimon/sentry-impact-risk/data/sentry_impact_risk.parquet"
    SAMPLE_PATH = RAW_DATA_DIR / "dataset.csv"
    
    if SAMPLE_PATH.exists():
        df = pd.read_csv(SAMPLE_PATH)
        print(f"Loaded local sample with {len(df)} rows from {SAMPLE_PATH}")
    else:
        df = pd.read_parquet(DATASET_URI)
    
        df = df.sample(
            n=min(500, len(df)),
            random_state=42
        )
    
        df.to_csv(SAMPLE_PATH, index=False)
        print(f"Downloaded dataset and saved a sample of {len(df)} rows to {SAMPLE_PATH}")
    
    df.head()

    # ---- PROCESS DATA  ----
    logger.info("Processing dataset...")
    df_reg = df.dropna().copy()

    print(f"Original dataset size: {len(df)}")
    print(f"Dataset size after removing missing values: {len(df_reg)}")

    print(f"Transforming target")

    df_reg["log_impact_probability"] = np.log10(
        df_reg["impact_probability"]
    )

    print("Selecting features")

    features = df_reg.select_dtypes(include="number").columns.tolist()

    features.remove("impact_probability")
    features.remove("log_impact_probability")
    
    X = df_reg[features]
    y = df_reg["log_impact_probability"]
    
    print("Selected features:")
    print(features)
    
    print(f"\nNumber of features: {X.shape[1]}")

    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
