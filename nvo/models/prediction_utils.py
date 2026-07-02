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
# These are defaults; tune_blend_weights() computes better values via CV
BLEND_WEIGHTS = {
    'Female': 0.6,
    'Male': 0.6,
    'Total': 0.4,
}


def tune_blend_weights(
    df: pd.DataFrame,
    target_col: str,
    model_params: dict,
    gender: str,
    files_dir: str = "files",
) -> float:
    """Tune blend weight using leave-one-year-out cross-validation.
    
    For each year Y in the dataset (except the first two), trains on all
    years < Y and evaluates different blend weights. Returns the average
    of per-fold optimal weights, capped at 0.8 to prevent overfitting
    to a single year's regime.
    
    Uses a coarse grid (0.0 to 1.0 in steps of 0.2) to reduce noise
    from small folds.
    """
    from nvo.models.equipercentile import compute_equipercentile_predictions
    
    years = sorted(df['Year'].unique())
    if len(years) < 3:
        # Not enough years for meaningful CV
        return BLEND_WEIGHTS.get(gender, 0.4)
    
    # Coarse grid to reduce noise from thin folds
    weight_candidates = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fold_optimal_weights = []
    
    for test_year in years[2:]:  # Need at least 2 years for training
        train_years_cv = [y for y in years if y < test_year]
        prev_year_cv = test_year - 1
        
        df_train_cv = df[df['Year'].isin(train_years_cv)].copy()
        df_test_cv = df[df['Year'] == test_year].copy()
        
        if df_train_cv.empty or df_test_cv.empty:
            continue
        
        # Train model on this fold
        model, le_school, le_profile, feature_cols, school_stats = train_model(
            df_train_cv, target_col, 1, model_params
        )
        
        if model is None:
            continue
        
        # Compute equipercentile predictions for this fold
        df_prev_cv = df_train_cv[df_train_cv['Year'] == prev_year_cv].copy()
        equi_preds = compute_equipercentile_predictions(
            df_prev_cv[df_prev_cv[target_col] > 0],
            target_col, prev_year_cv, test_year, gender, files_dir
        )
        
        # Prepare prediction data
        X, df_prep, prev_scores_map, mask = prepare_prediction_data(
            df_test_cv, df_train_cv, target_col, prev_year_cv,
            le_school, le_profile, school_stats, feature_cols
        )
        
        y_actual = df_prep[target_col].fillna(0)
        valid_mask = (y_actual > 0) & (df_prep['Prev_Year_Score'] > 0)
        
        X_valid = X[valid_mask]
        df_valid = df_prep[valid_mask]
        
        if len(X_valid) == 0:
            continue
        
        prev_scores = X_valid['Prev_Year_Score'].values
        predicted_changes = model.predict(X_valid)
        model_preds = prev_scores + predicted_changes
        actuals = y_actual[valid_mask].values
        
        # Find best weight for this fold
        fold_maes = {}
        for w in weight_candidates:
            preds = np.zeros(len(X_valid))
            for idx, (_, row) in enumerate(df_valid.iterrows()):
                key = row['Key']
                equi_pred = equi_preds.get(key)
                if equi_pred is not None:
                    model_residual = model_preds[idx] - equi_pred
                    preds[idx] = equi_pred + w * model_residual
                else:
                    preds[idx] = w * model_preds[idx] + (1 - w) * prev_scores[idx]
            
            fold_maes[w] = np.mean(np.abs(actuals - preds))
        
        best_w_this_fold = min(fold_maes, key=fold_maes.get)
        fold_optimal_weights.append(best_w_this_fold)
    
    if not fold_optimal_weights:
        return BLEND_WEIGHTS.get(gender, 0.4)
    
    # Average fold-optimal weights and cap at 0.8
    avg_weight = np.mean(fold_optimal_weights)
    capped_weight = min(0.8, avg_weight)
    # Round to nearest 0.2 (grid point)
    final_weight = round(capped_weight / 0.2) * 0.2
    
    logger.info(
        f"Blend weight CV for {gender}: fold_bests={[f'{w:.1f}' for w in fold_optimal_weights]}, "
        f"avg={avg_weight:.2f}, final={final_weight:.1f}"
    )
    
    return float(final_weight)


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
    gender: str = 'Female',
    equi_preds: Optional[Dict[str, float]] = None,
    blend_weight: Optional[float] = None
) -> List[PredictionResult]:
    """Generate predictions with metadata for each school-profile."""
    X_valid = X[mask]
    df_valid = df_prep[mask]
    
    if len(X_valid) == 0:
        return []
    
    # Get blend weight: explicit param > global default
    if blend_weight is None:
        blend_weight = BLEND_WEIGHTS.get(gender, 0.4)
    
    # Get base scores and predict changes
    prev_scores = X_valid['Prev_Year_Score'].values
    predicted_changes = model.predict(X_valid)
    
    # Model predictions (prev + delta)
    model_predictions = prev_scores + predicted_changes
    
    # Build final predictions with equipercentile if available
    if equi_preds:
        predictions = np.zeros(len(X_valid))
        for idx, (_, row) in enumerate(df_valid.iterrows()):
            key = row['Key']
            equi_pred = equi_preds.get(key)
            if equi_pred is not None:
                # Use equi as base + blend_weight * model residual from equi
                model_residual = model_predictions[idx] - equi_pred
                predictions[idx] = equi_pred + blend_weight * model_residual
            else:
                # Fallback: blend model vs naive
                predictions[idx] = blend_weight * model_predictions[idx] + (1 - blend_weight) * prev_scores[idx]
    else:
        # Original blending: blend_weight * model_pred + (1 - blend_weight) * naive
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
    multiplier: float = 2.5
) -> Tuple[float, float, float]:
    """Compute prediction intervals and confidence score.
    
    The multiplier of 2.5× volatility targets ~90% empirical coverage
    (calibrated against backtest on 2025 data).
    
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
