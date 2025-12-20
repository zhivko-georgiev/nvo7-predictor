"""Validation service."""
import numpy as np
from typing import Dict, List, Optional, Tuple

from nvo.data.processors import build_dataset
from nvo.models.trainer import train_model
from nvo.models.prediction_utils import (
    prepare_prediction_data,
    generate_predictions,
    compute_metrics,
)
from nvo.services.common import get_gender_list


def run_validation(
    train_years: List[int],
    test_year: int,
    files_dir: str,
    model_params: dict,
    gender_filter: Optional[str] = None,
    school_filter: Optional[List[str]] = None
) -> Tuple[Dict, Dict]:
    """Run validation against actual results.
    
    Returns:
        all_results: Dict of validation results keyed by (school, profile)
        metrics: Dict of metrics per round/gender
    """
    prev_year = test_year - 1
    
    df_all = build_dataset(train_years + [test_year], files_dir)
    df_train = df_all[df_all['Year'].isin(train_years)].copy()
    df_test = df_all[df_all['Year'] == test_year].copy()
    
    if school_filter:
        mask = df_test['School'].str.contains('|'.join(school_filter), case=False, na=False)
        df_test = df_test[mask]
    
    train_schools = set(df_train['School'].unique())
    df_test = df_test[df_test['School'].isin(train_schools)].copy()
    
    all_results = {}
    metrics = {}
    genders = get_gender_list(gender_filter)
    
    for round_num in [1, 2]:
        for g in genders:
            target_col = f'R{round_num}_Min_{g}'
            if target_col not in df_train.columns:
                continue
            
            model, le_school, le_profile, feature_cols, school_stats = train_model(
                df_train, target_col, round_num, model_params
            )
            
            if model is None:
                continue
            
            X, df_prep, prev_scores_map, mask = prepare_prediction_data(
                df_test, df_train, target_col, prev_year,
                le_school, le_profile, school_stats, feature_cols
            )
            
            y_test = df_prep[target_col].fillna(0)
            valid_mask = y_test > 0
            
            pred_results = generate_predictions(
                model, X, df_prep, prev_scores_map, school_stats, valid_mask
            )
            
            if not pred_results:
                continue
            
            y_actual = np.array([df_test.loc[df_test['School'] == r.school].loc[
                df_test['Profile'] == r.profile, target_col].values[0] for r in pred_results])
            y_pred = np.array([r.predicted for r in pred_results])
            
            existing_mask = np.array([not r.is_new for r in pred_results])
            reliable_mask = np.array([r.reliable for r in pred_results])
            
            metrics_existing = compute_metrics(y_actual, y_pred, existing_mask)
            metrics_reliable = compute_metrics(y_actual, y_pred, reliable_mask)
            
            metrics[f'R{round_num}_{g}'] = {
                'mae_existing': metrics_existing['mae'],
                'mae_reliable': metrics_reliable['mae'],
                'new_profiles': sum(r.is_new for r in pred_results),
                'total': len(pred_results),
                'reliable': metrics_reliable['count']
            }
            
            for r, actual in zip(pred_results, y_actual):
                key = (r.school, r.profile)
                if key not in all_results:
                    all_results[key] = {'School': r.school, 'Profile': r.profile}
                
                all_results[key][f'R{round_num}_{g}_Actual'] = actual
                all_results[key][f'R{round_num}_{g}_Predicted'] = round(r.predicted, 2)
                all_results[key][f'R{round_num}_{g}_Error'] = round(actual - r.predicted, 2)
                all_results[key][f'R{round_num}_{g}_Abs_Error'] = round(abs(actual - r.predicted), 2)
                all_results[key][f'R{round_num}_{g}_Volatility'] = round(r.volatility, 1)
                all_results[key][f'R{round_num}_{g}_Years_Data'] = r.n_years
                all_results[key][f'R{round_num}_{g}_Reliable'] = r.reliable
    
    return all_results, metrics
