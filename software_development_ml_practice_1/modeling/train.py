"""
Module for training and evaluating the machine learning model.
"""

from pathlib import Path
import joblib

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import typer

# Assuming MODELS_DIR is defined in your config.py. If not, add it.
from software_development_ml_practice_1.config import PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()

def build_baseline_model(input_shape: int) -> models.Sequential:
    """
    Builds and compiles a baseline neural network for regression.

    This simple architecture is designed to predict the logarithmic 
    impact probability of asteroids based on numerical features.

    Args:
        input_shape (int): The number of features in the input data.

    Returns:
        models.Sequential: A compiled Keras Sequential model.
    """
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

@app.command()
def main(
    x_path: Path = PROCESSED_DATA_DIR / "X_processed.csv",
    y_path: Path = PROCESSED_DATA_DIR / "y_processed.csv",
    model_output_path: Path = MODELS_DIR / "baseline_model.keras",
    scaler_output_path: Path = MODELS_DIR / "scaler.pkl",
    epochs: int = 50,
    batch_size: int = 32,
    test_size: float = 0.2
) -> None:
    """
    Trains a baseline neural network model on the processed dataset.

    Loads the processed features and target, splits them into training 
    and testing sets, scales the features using StandardScaler, trains 
    a Keras model, evaluates its performance, and saves both the model 
    and the scaler to disk for future inference.

    Args:
        x_path (Path): Path to the processed features CSV.
        y_path (Path): Path to the processed target CSV.
        model_output_path (Path): Path to save the trained Keras model.
        scaler_output_path (Path): Path to save the fitted StandardScaler.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    logger.info(f"Loading processed data from {PROCESSED_DATA_DIR}...")
    try:
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
    except FileNotFoundError:
        logger.error(f"Processed data not found. Please run dataset.py first.")
        raise typer.Exit(code=1)

    logger.info(f"Splitting data into train and test sets (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Building the neural network model...")
    model = build_baseline_model(input_shape=X_train_scaled.shape[1])

    logger.info(f"Training the model for {epochs} epochs...")
    # Training the model, keeping 20% of the training data for validation
    history = model.fit(
        X_train_scaled, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2,
        verbose=1
    )

    logger.info("Evaluating the model on the test set...")
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    logger.success(f"Test Loss (MSE): {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    logger.info("Saving artifacts (Model and Scaler)...")
    # Ensure directories exist before saving
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(model_output_path)
    joblib.dump(scaler, scaler_output_path)
    
    logger.success(f"Model saved to {model_output_path}")
    logger.success(f"Scaler saved to {scaler_output_path}")
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    app()