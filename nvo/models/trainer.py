"""Model training with mean reversion."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    """Compute per-school statistics."""
    stats = {}
    for (school, profile), group in df.groupby(['School', 'Profile']):
        scores = group.sort_values('Year')[target_col].values
        scores = scores[scores > 0]
        key = f"{school}|{profile}"
        
        if len(scores) >= 2:
            changes = np.diff(scores)
            # Acceleration: is trend increasing?
            accel = np.diff(changes).mean() if len(changes) >= 2 else 0
            stats[key] = {
                'mean': scores.mean(),
                'volatility': np.abs(changes).mean(),
                'trend': changes.mean(),  # Average trend across all years
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
    return stats


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
    
    school_stats = compute_school_stats(df, target_col)
    
    df_sorted = df.sort_values(['School', 'Profile', 'Year'])
    df['Prev_Year_Score'] = df_sorted.groupby(['School', 'Profile'])[target_col].shift(1)
    df['YoY_Change'] = df[target_col] - df['Prev_Year_Score']
    
    df['Key'] = df['School'] + '|' + df['Profile']
    df['School_Mean'] = df['Key'].map(lambda k: school_stats.get(k, {}).get('mean', 0))
    df['School_Volatility'] = df['Key'].map(lambda k: school_stats.get(k, {}).get('volatility', 25))
    df['School_Trend'] = df['Key'].map(lambda k: school_stats.get(k, {}).get('trend', 0))
    df['School_Acceleration'] = df['Key'].map(lambda k: school_stats.get(k, {}).get('acceleration', 0))
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
    
    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42
    )
    
    model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
    
    pred_test = X_test['Prev_Year_Score'].values + model.predict(X_test)
    actual_test = X_test['Prev_Year_Score'].values + y_test.values
    test_mae = np.abs(pred_test - actual_test).mean()
    baseline_mae = np.abs(y_test).mean()
    
    logger.info(f"Model MAE={test_mae:.1f} (baseline={baseline_mae:.1f})")
    
    return model, le_school, le_profile, feature_cols, school_stats
