"""
Module for feature engineering.
Contains functions to clean data and generate predictor variables.
"""

import pandas as pd
import numpy as np
from loguru import logger

def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Cleans the dataset and separates the features (X) from the target (y).

    This function removes missing values, applies a base-10 logarithmic 
    transformation to the impact probability, and selects only numerical 
    columns as predictor variables.

    Args:
        df (pd.DataFrame): The raw dataset.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): Feature matrix (numerical variables).
            - y (pd.Series): Target vector (log_impact_probability).
    """
    logger.info("Processing dataset: removing missing values...")
    df_reg = df.dropna().copy()
    
    logger.info(f"Dataset size after dropping NaNs: {len(df_reg)}")
    
    logger.info("Transforming the target variable...")
    # Calculate the base-10 logarithm
    df_reg["log_impact_probability"] = np.log10(df_reg["impact_probability"])
    
    logger.info("Selecting numerical features...")
    features = df_reg.select_dtypes(include="number").columns.tolist()
    
    # Ensure the target (original or transformed) does not leak into the features
    if "impact_probability" in features:
        features.remove("impact_probability")
    if "log_impact_probability" in features:
        features.remove("log_impact_probability")
        
    X = df_reg[features]
    y = df_reg["log_impact_probability"]
    
    logger.info(f"Final number of features: {X.shape[1]}")
    
    return X, y