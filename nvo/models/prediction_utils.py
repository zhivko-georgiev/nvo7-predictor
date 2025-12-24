"""Shared prediction utilities for validate and predict commands."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from nvo.models.trainer import train_model, prepare_features
from nvo.utils.logger import get_logger

logger = get_logger("models.prediction_utils")

RELIABILITY_MIN_YEARS = 2
RELIABILITY_MAX_VOLATILITY = 20

# Gender-specific blend weights (model vs naive)
BLEND_WEIGHTS = {
    'Female': 0.6,  # 60% model, 40% naive
    'Male': 0.0,    # 0% model, 100% naive (model doesn't help for male)
    'Total': 0.4,   # 40% model, 60% naive
}


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    school: str
    profile: str
    predicted: float
    prev_score: float
    volatility: float
    n_years: int
    has_prev_year: bool
    is_new: bool
    
    @property
    def reliable(self) -> bool:
        return (self.n_years >= RELIABILITY_MIN_YEARS and 
                self.volatility < RELIABILITY_MAX_VOLATILITY and 
                self.has_prev_year)


def prepare_prediction_data(
    df_data: pd.DataFrame,
    df_train: pd.DataFrame,
    target_col: str,
    prev_year: int,
    le_school,
    le_profile,
    school_stats: Dict,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, np.ndarray]:
    """Prepare data for prediction with all required features.
    
    Returns:
        X: Feature matrix ready for prediction
        df_prep: Prepared dataframe with all columns
        prev_scores_map: Mapping of school|profile to previous year score
        mask: Boolean mask for valid rows (target > 0)
    """
    # Get previous year scores (only valid ones)
    df_prev = df_train[(df_train['Year'] == prev_year) & (df_train[target_col] > 0)].copy()
    prev_scores_map = dict(zip(
        df_prev['School'] + '|' + df_prev['Profile'],
        df_prev[target_col]
    ))
    
    # Encode categorical features
    df_prep, _, _ = prepare_features(df_data, le_school, le_profile)
    df_prep['Key'] = df_prep['School'] + '|' + df_prep['Profile']
    
    # Add school statistics
    df_prep['School_Mean'] = df_prep['Key'].map(
        lambda k: school_stats.get(k, {}).get('mean', 0))
    df_prep['School_Volatility'] = df_prep['Key'].map(
        lambda k: school_stats.get(k, {}).get('volatility', 25))
    df_prep['School_Trend'] = df_prep['Key'].map(
        lambda k: school_stats.get(k, {}).get('trend', 0))
    df_prep['School_Acceleration'] = df_prep['Key'].map(
        lambda k: school_stats.get(k, {}).get('acceleration', 0))
    df_prep['Last_Score'] = df_prep['Key'].map(
        lambda k: school_stats.get(k, {}).get('last_score', 0))
    
    # Use actual prev year score, fallback to last known score
    df_prep['Prev_Year_Score'] = df_prep['Key'].map(prev_scores_map).fillna(df_prep['Last_Score'])
    df_prep['Dist_From_Mean'] = df_prep['Prev_Year_Score'] - df_prep['School_Mean']
    
    # Prepare feature matrix
    X = df_prep[feature_cols].fillna(0)
    
    # Create mask for valid target values
    if target_col in df_prep.columns:
        mask = df_prep[target_col].fillna(0) > 0
    else:
        mask = pd.Series([True] * len(df_prep), index=df_prep.index)
    
    return X, df_prep, prev_scores_map, mask


def generate_predictions(
    model,
    X: pd.DataFrame,
    df_prep: pd.DataFrame,
    prev_scores_map: Dict,
    school_stats: Dict,
    mask: pd.Series,
    gender: str = 'Female'
) -> List[PredictionResult]:
    """Generate predictions with metadata for each school-profile."""
    X_valid = X[mask]
    df_valid = df_prep[mask]
    
    if len(X_valid) == 0:
        return []
    
    # Get blend weight for this gender
    blend_weight = BLEND_WEIGHTS.get(gender, 0.4)
    
    # Get base scores and predict changes
    prev_scores = X_valid['Prev_Year_Score'].values
    predicted_changes = model.predict(X_valid)
    
    # Apply blending: blend_weight * model_pred + (1 - blend_weight) * naive
    model_predictions = prev_scores + predicted_changes
    predictions = blend_weight * model_predictions + (1 - blend_weight) * prev_scores
    predictions = np.clip(predictions, 0, 500)  # Cap at 0-500
    
    results = []
    for idx, (_, row) in enumerate(df_valid.iterrows()):
        key = row['Key']
        stats = school_stats.get(key, {})
        
        result = PredictionResult(
            school=row['School'],
            profile=row['Profile'],
            predicted=float(predictions[idx]),
            prev_score=float(prev_scores[idx]),
            volatility=stats.get('volatility', 25),
            n_years=stats.get('n_years', 0),
            has_prev_year=key in prev_scores_map,
            is_new=key not in prev_scores_map
        )
        results.append(result)
    
    return results


def compute_prediction_intervals(
    predicted: float,
    volatility: float,
    avg_volatility: float,
    multiplier: float = 1.5
) -> Tuple[float, float, float]:
    """Compute prediction intervals and confidence score.
    
    Returns:
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval  
        confidence: Confidence score (0-100)
    """
    lower = max(0, predicted - multiplier * volatility)
    upper = min(500, predicted + multiplier * volatility)
    confidence = max(0, min(100, 100 - (volatility / avg_volatility) * 40))
    return lower, upper, confidence


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray = None
) -> Dict[str, float]:
    """Compute prediction metrics (MAE, RMSE)."""
    if mask is not None:
        actual = actual[mask]
        predicted = predicted[mask]
    
    if len(actual) == 0:
        return {'mae': 0, 'rmse': 0, 'count': 0}
    
    errors = actual - predicted
    return {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'count': len(actual)
    }
