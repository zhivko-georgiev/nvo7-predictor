"""Model training with mean reversion."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from typing import Tuple, List, Optional, Dict
from nvo.utils.logger import get_logger

logger = get_logger("models.trainer")


def prepare_features(
    df: pd.DataFrame,
    le_school: Optional[LabelEncoder] = None,
    le_profile: Optional[LabelEncoder] = None
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Encode categorical features."""
    df = df.copy()
    
    if le_school is None:
        le_school = LabelEncoder()
        df['School_Encoded'] = le_school.fit_transform(df['School'].astype(str))
    else:
        known = set(le_school.classes_)
        df['School_Encoded'] = df['School'].apply(
            lambda x: le_school.transform([x])[0] if x in known else -1
        )
    
    if le_profile is None:
        le_profile = LabelEncoder()
        df['Profile_Encoded'] = le_profile.fit_transform(df['Profile'].astype(str))
    else:
        known = set(le_profile.classes_)
        df['Profile_Encoded'] = df['Profile'].apply(
            lambda x: le_profile.transform([x])[0] if x in known else -1
        )
    
    return df, le_school, le_profile


def compute_school_stats(df: pd.DataFrame, target_col: str) -> Dict[str, Dict]:
    """Compute per-school statistics using ALL available years.
    
    Used at prediction time when we want stats built from the full
    historical window (all years are in the past relative to the
    prediction target).
    """
    stats = {}
    for (school, profile), group in df.groupby(['School', 'Profile']):
        scores = group.sort_values('Year')[target_col].values
        scores = scores[scores > 0]
        key = f"{school}|{profile}"
        
        if len(scores) >= 2:
            changes = np.diff(scores)
            accel = np.diff(changes).mean() if len(changes) >= 2 else 0
            stats[key] = {
                'mean': scores.mean(),
                'volatility': np.abs(changes).mean(),
                'trend': changes.mean(),
                'acceleration': accel,
                'last_score': scores[-1],
                'n_years': len(scores),
                'max_change': np.abs(changes).max()
            }
        elif len(scores) == 1:
            stats[key] = {
                'mean': scores[0],
                'volatility': 25.0,
                'trend': 0,
                'acceleration': 0,
                'last_score': scores[0],
                'n_years': 1,
                'max_change': 0
            }
    return stats


def compute_walkforward_stats(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Compute per-school stats using only years STRICTLY BEFORE each row.
    
    This avoids data leakage during training: a row's features never
    include information from its own year or future years.
    
    Returns a DataFrame with columns:
        School, Profile, Year, School_Mean, School_Volatility,
        School_Trend, School_Acceleration
    """
    df_sorted = df.sort_values(['School', 'Profile', 'Year'])
    stats_rows = []
    
    for (school, profile), group in df_sorted.groupby(['School', 'Profile']):
        group = group.sort_values('Year')
        scores = group[target_col].values
        years = group['Year'].values
        
        for i in range(len(scores)):
            # Only use scores from strictly earlier years with valid (>0) values
            past_scores = scores[:i]
            past_scores = past_scores[past_scores > 0]
            
            key = f"{school}|{profile}"
            
            if len(past_scores) >= 2:
                changes = np.diff(past_scores)
                accel = np.diff(changes).mean() if len(changes) >= 2 else 0
                stats_rows.append({
                    'School': school,
                    'Profile': profile,
                    'Year': years[i],
                    'School_Mean': past_scores.mean(),
                    'School_Volatility': np.abs(changes).mean(),
                    'School_Trend': changes.mean(),
                    'School_Acceleration': accel,
                })
            elif len(past_scores) == 1:
                stats_rows.append({
                    'School': school,
                    'Profile': profile,
                    'Year': years[i],
                    'School_Mean': past_scores[0],
                    'School_Volatility': 25.0,
                    'School_Trend': 0,
                    'School_Acceleration': 0,
                })
            else:
                # No past data — row will be dropped later (no prev_score either)
                stats_rows.append({
                    'School': school,
                    'Profile': profile,
                    'Year': years[i],
                    'School_Mean': 0,
                    'School_Volatility': 25.0,
                    'School_Trend': 0,
                    'School_Acceleration': 0,
                })
    
    return pd.DataFrame(stats_rows)


def train_model(
    df: pd.DataFrame,
    target_col: str,
    round_num: int,
    model_params: dict
) -> Tuple[Optional[xgb.XGBRegressor], LabelEncoder, LabelEncoder, List[str], Dict]:
    """Train model to predict year-over-year change."""
    logger.info(f"Training model for {target_col}...")
    
    df = df.copy()
    df, le_school, le_profile = prepare_features(df)
    
    # Compute walk-forward stats (no leakage: each row only sees past years)
    wf_stats = compute_walkforward_stats(df, target_col)
    # Drop any pre-existing stat columns to avoid suffix conflicts on merge
    stat_cols = ['School_Mean', 'School_Volatility', 'School_Trend', 'School_Acceleration']
    df = df.drop(columns=[c for c in stat_cols if c in df.columns], errors='ignore')
    df = df.merge(wf_stats, on=['School', 'Profile', 'Year'], how='left')
    # Fill NaN stats (first year for a school with no prior data)
    df['School_Mean'] = df['School_Mean'].fillna(0)
    df['School_Volatility'] = df['School_Volatility'].fillna(25.0)
    df['School_Trend'] = df['School_Trend'].fillna(0)
    df['School_Acceleration'] = df['School_Acceleration'].fillna(0)
    
    # Also compute full stats for use at prediction time (returned to caller)
    school_stats = compute_school_stats(df, target_col)
    
    df_sorted = df.sort_values(['School', 'Profile', 'Year'])
    df['Prev_Year_Score'] = df_sorted.groupby(['School', 'Profile'])[target_col].shift(1)
    df['YoY_Change'] = df[target_col] - df['Prev_Year_Score']
    
    df['Key'] = df['School'] + '|' + df['Profile']
    df['Dist_From_Mean'] = df['Prev_Year_Score'] - df['School_Mean']
    
    feature_cols = [
        'Prev_Year_Score',
        'School_Mean',
        'School_Volatility',
        'School_Trend',
        'School_Acceleration',
        'Dist_From_Mean',
        'School_Encoded',
        'Profile_Encoded',
    ]
    
    # Add capacity features if available
    capacity_cols = [c for c in df.columns if c.startswith('Capacity_')]
    feature_cols.extend(capacity_cols)
    
    # Add exam features
    exam_features = [c for c in df.columns if c.startswith('Exam_')]
    feature_cols.extend(exam_features)
    
    valid_mask = (df[target_col] > 0) & (df['Prev_Year_Score'] > 0)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) < 10:
        logger.warning(f"Only {len(df_valid)} samples, skipping")
        return None, le_school, le_profile, feature_cols, school_stats
    
    X = df_valid[feature_cols].fillna(0)
    y = df_valid['YoY_Change']
    
    years = df_valid['Year'].values
    weights = np.where(years == years.max(), 2.0, 1.0)
    
    # Temporal split: leave-last-year-out for honest internal validation
    max_year = years.max()
    train_mask = years < max_year
    test_mask = years == max_year
    
    if train_mask.sum() >= 10 and test_mask.sum() >= 5:
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        w_train = weights[train_mask]
        
        model = xgb.XGBRegressor(
            n_estimators=model_params.get('n_estimators', 50),
            max_depth=model_params.get('max_depth', 3),
            learning_rate=model_params.get('learning_rate', 0.1),
            reg_alpha=model_params.get('reg_alpha', 1.0),
            reg_lambda=model_params.get('reg_lambda', 2.0),
            min_child_weight=model_params.get('min_child_weight', 5),
            random_state=model_params.get('random_state', 42)
        )
        
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        pred_test = X_test['Prev_Year_Score'].values + model.predict(X_test)
        actual_test = X_test['Prev_Year_Score'].values + y_test.values
        test_mae = np.abs(pred_test - actual_test).mean()
        baseline_mae = np.abs(y_test).mean()
        
        logger.info(f"Internal validation (leave-last-year-out): MAE={test_mae:.1f} (naive baseline={baseline_mae:.1f})")
    else:
        logger.info("Insufficient years for temporal split; skipping internal validation")
    
    # Retrain on ALL data for the final production model
    model = xgb.XGBRegressor(
        n_estimators=model_params.get('n_estimators', 50),
        max_depth=model_params.get('max_depth', 3),
        learning_rate=model_params.get('learning_rate', 0.1),
        reg_alpha=model_params.get('reg_alpha', 1.0),
        reg_lambda=model_params.get('reg_lambda', 2.0),
        min_child_weight=model_params.get('min_child_weight', 5),
        random_state=model_params.get('random_state', 42)
    )
    
    model.fit(X, y, sample_weight=weights, verbose=False)
    
    return model, le_school, le_profile, feature_cols, school_stats
