"""Validation service."""
import numpy as np
from typing import Dict, List, Optional, Tuple

from nvo.data.processors import build_dataset
from nvo.models.trainer import train_model
from nvo.models.prediction_utils import (
    prepare_prediction_data,
    generate_predictions,
    compute_metrics,
    compute_prediction_intervals,
    tune_blend_weights,
    BLEND_WEIGHTS,
)
from nvo.models.equipercentile import compute_equipercentile_predictions
from nvo.services.common import get_gender_list


def _compute_interval_coverage(
    all_results: Dict,
    round_num: int,
    gender: str,
    multiplier: float = 2.5,
) -> Dict[str, float]:
    """Compute empirical coverage of prediction intervals.
    
    Checks what fraction of actuals fall within the ±multiplier×volatility bands.
    """
    inside = 0
    total = 0
    
    for key, result in all_results.items():
        actual_key = f'R{round_num}_{gender}_Actual'
        pred_key = f'R{round_num}_{gender}_Predicted'
        vol_key = f'R{round_num}_{gender}_Volatility'
        
        if actual_key not in result or pred_key not in result or vol_key not in result:
            continue
        
        actual = result[actual_key]
        predicted = result[pred_key]
        volatility = result[vol_key]
        
        if actual <= 0 or predicted <= 0:
            continue
        
        lower = predicted - multiplier * volatility
        upper = predicted + multiplier * volatility
        
        total += 1
        if lower <= actual <= upper:
            inside += 1
    
    coverage = inside / total if total > 0 else 0
    return {
        'coverage': coverage,
        'inside': inside,
        'total': total,
    }


def run_validation(
    train_years: List[int],
    test_year: int,
    files_dir: str,
    model_params: dict,
    gender_filter: Optional[str] = None,
    school_filter: Optional[List[str]] = None
) -> Tuple[Dict, Dict]:
    """Run validation against actual results.
    
    R1 is validated using the trained delta model (same as production).
    R2 is validated using the production method: R1_pred + global avg(R2-R1) diff
    computed from training data. This ensures reported MAE matches what we ship.
    
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
    
    # Store R1 predictions per gender for R2 derivation
    r1_predictions_by_gender = {}
    
    for g in genders:
        # === ROUND 1: trained delta model + equipercentile (same as production) ===
        target_col = f'R1_Min_{g}'
        if target_col not in df_train.columns:
            continue
        
        # Tune blend weight using leave-one-year-out CV on training data
        if len(train_years) >= 3:
            tuned_blend_weight = tune_blend_weights(
                df_train, target_col, model_params, g, files_dir
            )
        else:
            tuned_blend_weight = BLEND_WEIGHTS.get(g, 0.4)
        
        model, le_school, le_profile, feature_cols, school_stats = train_model(
            df_train, target_col, 1, model_params
        )
        
        if model is None:
            continue
        
        # Compute equipercentile predictions (maps prev year through distributions)
        df_prev = df_train[df_train['Year'] == prev_year].copy()
        equi_preds = compute_equipercentile_predictions(
            df_prev[df_prev[target_col] > 0],
            target_col, prev_year, test_year, g, files_dir
        )
        
        X, df_prep, prev_scores_map, mask = prepare_prediction_data(
            df_test, df_train, target_col, prev_year,
            le_school, le_profile, school_stats, feature_cols
        )
        
        y_test_r1 = df_prep[target_col].fillna(0)
        valid_mask = y_test_r1 > 0
        
        pred_results = generate_predictions(
            model, X, df_prep, prev_scores_map, school_stats, valid_mask,
            gender=g, equi_preds=equi_preds if equi_preds else None,
            blend_weight=tuned_blend_weight
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
        
        metrics[f'R1_{g}'] = {
            'mae_existing': metrics_existing['mae'],
            'mae_reliable': metrics_reliable['mae'],
            'new_profiles': sum(r.is_new for r in pred_results),
            'total': len(pred_results),
            'reliable': metrics_reliable['count'],
        }
        
        # Store R1 predictions for R2 derivation
        r1_pred_map = {}
        avg_volatility = np.mean([s['volatility'] for s in school_stats.values()]) if school_stats else 25.0
        
        for r, actual in zip(pred_results, y_actual):
            key = (r.school, r.profile)
            r1_pred_map[key] = r.predicted
            
            if key not in all_results:
                all_results[key] = {'School': r.school, 'Profile': r.profile}
            
            all_results[key][f'R1_{g}_Actual'] = actual
            all_results[key][f'R1_{g}_Predicted'] = round(r.predicted, 2)
            all_results[key][f'R1_{g}_Error'] = round(actual - r.predicted, 2)
            all_results[key][f'R1_{g}_Abs_Error'] = round(abs(actual - r.predicted), 2)
            all_results[key][f'R1_{g}_Volatility'] = round(r.volatility, 1)
            all_results[key][f'R1_{g}_Years_Data'] = r.n_years
            all_results[key][f'R1_{g}_Reliable'] = r.reliable
        
        r1_predictions_by_gender[g] = r1_pred_map
        
        # === ROUND 2: production method (R1_pred + per-profile R2-R1 offset) ===
        r2_target_col = f'R2_Min_{g}'
        if r2_target_col not in df_train.columns:
            continue
        
        # Compute per-profile R2-R1 diffs and a global fallback
        r1_col = f'R1_Min_{g}'
        per_profile_r2_diffs = {}  # key → list of (r2 - r1) values
        global_r2_r1_diffs = []
        
        for (school, profile), group in df_train.groupby(['School', 'Profile']):
            key = (school, profile)
            diffs = []
            for _, row in group.iterrows():
                r1_val = row.get(r1_col, 0)
                r2_val = row.get(r2_target_col, 0)
                if r1_val > 0 and r2_val > 0:
                    diffs.append(r2_val - r1_val)
                    global_r2_r1_diffs.append(r2_val - r1_val)
            if diffs:
                per_profile_r2_diffs[key] = diffs
        
        global_avg_diff = np.mean(global_r2_r1_diffs) if global_r2_r1_diffs else 0
        global_std_diff = np.std(global_r2_r1_diffs) if global_r2_r1_diffs else 20
        
        # Generate R2 predictions using per-profile offset (mirrors production)
        r2_actuals = []
        r2_preds = []
        r2_results_list = []
        
        for r in pred_results:
            key = (r.school, r.profile)
            r1_pred = r1_pred_map.get(key, 0)
            if r1_pred <= 0:
                continue
            
            # Check if test data has R2 actual
            r2_actual_vals = df_test.loc[
                (df_test['School'] == r.school) & (df_test['Profile'] == r.profile),
                r2_target_col
            ].values
            if len(r2_actual_vals) == 0 or r2_actual_vals[0] <= 0:
                continue
            
            # Use per-profile offset if available, otherwise global
            profile_diffs = per_profile_r2_diffs.get(key)
            if profile_diffs:
                offset = np.median(profile_diffs)
            else:
                offset = global_avg_diff
            
            r2_actual = r2_actual_vals[0]
            r2_pred = max(0, r1_pred + offset)
            
            r2_actuals.append(r2_actual)
            r2_preds.append(r2_pred)
            r2_results_list.append((key, r2_actual, r2_pred, r))
        
        if r2_results_list:
            r2_actuals_arr = np.array(r2_actuals)
            r2_preds_arr = np.array(r2_preds)
            
            # Separate existing vs new profiles for R2 metrics
            r2_existing_mask = np.array([not r.is_new for _, _, _, r in r2_results_list])
            r2_reliable_mask = np.array([r.reliable for _, _, _, r in r2_results_list])
            
            r2_metrics_existing = compute_metrics(r2_actuals_arr, r2_preds_arr, r2_existing_mask)
            r2_metrics_reliable = compute_metrics(r2_actuals_arr, r2_preds_arr, r2_reliable_mask)
            
            metrics[f'R2_{g}'] = {
                'mae_existing': r2_metrics_existing['mae'],
                'mae_reliable': r2_metrics_reliable['mae'],
                'new_profiles': int((~r2_existing_mask).sum()),
                'total': len(r2_results_list),
                'reliable': r2_metrics_reliable['count'],
                'method': 'global_diff',
                'avg_r2_r1_diff': round(global_avg_diff, 2),
            }
            
            for key, r2_actual, r2_pred, r in r2_results_list:
                if key not in all_results:
                    all_results[key] = {'School': r.school, 'Profile': r.profile}
                
                profile_diffs = per_profile_r2_diffs.get(key)
                if profile_diffs and len(profile_diffs) >= 2:
                    r2_vol = max(r.volatility, np.std(profile_diffs))
                else:
                    r2_vol = max(r.volatility, global_std_diff)
                
                all_results[key][f'R2_{g}_Actual'] = r2_actual
                all_results[key][f'R2_{g}_Predicted'] = round(r2_pred, 2)
                all_results[key][f'R2_{g}_Error'] = round(r2_actual - r2_pred, 2)
                all_results[key][f'R2_{g}_Abs_Error'] = round(abs(r2_actual - r2_pred), 2)
                all_results[key][f'R2_{g}_Volatility'] = round(r2_vol, 1)
                all_results[key][f'R2_{g}_Years_Data'] = r.n_years
                all_results[key][f'R2_{g}_Reliable'] = r.reliable
    
    # === Compute interval coverage metrics ===
    for g in genders:
        for round_num in [1, 2]:
            coverage = _compute_interval_coverage(all_results, round_num, g)
            metric_key = f'R{round_num}_{g}'
            if metric_key in metrics and coverage['total'] > 0:
                metrics[metric_key]['interval_coverage'] = coverage['coverage']
                metrics[metric_key]['interval_inside'] = coverage['inside']
                metrics[metric_key]['interval_total'] = coverage['total']
    
    return all_results, metrics
