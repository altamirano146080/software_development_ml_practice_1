from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger

from software_development_ml_practice_1.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()

DATASET_PATH = PROCESSED_DATA_DIR / "dataset.csv"


def create_plots(
    input_path: Path = DATASET_PATH,
    output_dir: Path = FIGURES_DIR,
) -> None:
    """
    Creates exploratory data analysis plots.
    """

    dataframe = pd.read_csv(input_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # Correlation heatmap
    numeric_columns = dataframe.select_dtypes(
        include="number"
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        numeric_columns.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
    )
    plt.title("Correlation Heatmap of Asteroid Risk Metrics")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close()

    # Impact probability distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        np.log10(dataframe["impact_probability"]),
        bins=50,
    )
    plt.title("Log10 Impact Probability Distribution")
    plt.xlabel("Log10(Impact Probability)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "impact_probability_distribution.png", dpi=300)
    plt.close()

    # Velocity and Palermo scale
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=dataframe,
        x="v_infinity_kms",
        y="palermo_scale_cum",
        alpha=0.6,
    )
    plt.title("Encounter Velocity vs. Cumulative Palermo Scale")
    plt.xlabel("Relative Velocity (km/s)")
    plt.ylabel("Cumulative Palermo Scale")
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_vs_palermo.png", dpi=300)
    plt.close()

    # Absolute magnitude and Palermo scale
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=dataframe,
        x="absolute_magnitude",
        y="palermo_scale_cum",
        alpha=0.6,
    )
    plt.gca().invert_xaxis()
    plt.title("Absolute Magnitude vs. Cumulative Palermo Scale")
    plt.xlabel("Absolute Magnitude")
    plt.ylabel("Cumulative Palermo Scale")
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_vs_palermo.png", dpi=300)
    plt.close()

    # Potential impact timeline
    plt.figure(figsize=(8, 5))
    sns.histplot(
        dataframe["year_range_min"].dropna(),
        bins=50,
    )
    plt.title("Timeline of First Potential Encounters")
    plt.xlabel("Year")
    plt.ylabel("Number of Objects")
    plt.xlim(2026, 2126)
    plt.tight_layout()
    plt.savefig(output_dir / "potential_impact_timeline.png", dpi=300)
    plt.close()

    logger.success(f"Plots saved to {output_dir}")


@app.command()
def main():
    """
    Generates all exploratory data analysis plots.
    """

    create_plots()


if __name__ == "__main__":
    app()