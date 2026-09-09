from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd

from software_development_ml_practice_1.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    force_download: bool = False,
):
    # ---- DOWNLOAD DATA ----
    if not input_path.exists() or force_download:
        logger.info("Downloading dataset...")

        df = pd.read_parquet(
            "hf://datasets/juliensimon/sentry-impact-risk/data/sentry_impact_risk.parquet"
        )
        # Save the downloaded data to a CSV file
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(input_path, index=False)

        logger.success("Dataset downloaded.")
    else:
        logger.info("Dataset already exists.")
        df = pd.read_csv(input_path)

    # ---- PROCESS DATA (adapt this section later) ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
