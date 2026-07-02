"""Prediction service."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from nvo.data.processors import build_dataset
from nvo.data.exam_loaders import load_exam_distribution
from nvo.models.trainer import train_model
from nvo.models.persistence import save_model, load_model, is_model_valid
from nvo.models.prediction_utils import (
    prepare_prediction_data,
    generate_predictions,
    compute_prediction_intervals,
    tune_blend_weights,
    BLEND_WEIGHTS,
)
from nvo.models.equipercentile import compute_equipercentile_predictions
from nvo.services.common import get_gender_list
from nvo.utils.logger import get_logger

logger = get_logger("services.prediction")


def _get_or_train_model(df_hist, target_col, round_num, model_params, gender, historical_years, use_cache=True):
    """Get cached model or train new one."""
    if use_cache and is_model_valid(gender, round_num, historical_years):
        bundle = load_model(gender, round_num)
        if bundle and bundle['model'] is not None:
            logger.info(f"Using cached model for R{round_num} {gender}")
            return (bundle['model'], bundle['le_school'], bundle['le_profile'], 
                    bundle['feature_cols'], bundle['school_stats'])
    
    logger.info(f"Training model for R{round_num} {gender}...")
    model, le_school, le_profile, feature_cols, school_stats = train_model(
        df_hist, target_col, round_num, model_params
    )
    
    if model is not None and use_cache:
        save_model(model, le_school, le_profile, feature_cols, school_stats,
                   gender, round_num, historical_years)
    
    return model, le_school, le_profile, feature_cols, school_stats


def run_predictions(
    historical_years: List[int],
    predict_year: int,
    files_dir: str,
    model_params: dict,
    gender_filter: Optional[str] = None,
    school_filter: Optional[List[str]] = None,
    use_cache: bool = True
) -> Tuple[Dict, Dict]:
    """Run predictions for all rounds and genders.
    
    R1: Trained delta model + equipercentile backbone.
    R2: R1_pred + global average (R2-R1) diff from training data.
    
    Returns:
        all_results: Dict of predictions keyed by (school, profile)
        metrics: Dict of metrics per round/gender
    """
    template_year = max(historical_years)
    prev_year = template_year
    
    df_hist = build_dataset(historical_years, files_dir)
    df_template = df_hist[df_hist['Year'] == template_year].copy()
    
    exam_features = load_exam_distribution(predict_year, files_dir)
    if exam_features:
        for key, val in exam_features.items():
            df_template[key] = val
    
    if school_filter:
        mask = df_template['School'].str.contains('|'.join(school_filter), case=False, na=False)
        df_template = df_template[mask]
    
    all_results = {}
    metrics = {}
    genders = get_gender_list(gender_filter)
    
    for g in genders:
        # Tune blend weight via leave-one-year-out CV
        r1_target = f'R1_Min_{g}'
        if r1_target in df_hist.columns and len(historical_years) >= 3:
            blend_weight = tune_blend_weights(
                df_hist, r1_target, model_params, g, files_dir
            )
        else:
            blend_weight = BLEND_WEIGHTS.get(g, 0.4)
        
        # === ROUND 1: Delta model + equipercentile ===
        target_col = f'R1_Min_{g}'
        if target_col not in df_hist.columns:
            continue
        
        model, le_school, le_profile, feature_cols, school_stats = _get_or_train_model(
            df_hist, target_col, 1, model_params, g, historical_years, use_cache
        )
        
        if model is None:
            continue
        
        avg_volatility = np.mean([s['volatility'] for s in school_stats.values()])
        
        # Compute equipercentile predictions
        df_prev = df_hist[df_hist['Year'] == prev_year].copy()
        equi_preds = compute_equipercentile_predictions(
            df_prev[df_prev[target_col] > 0],
            target_col, prev_year, predict_year, g, files_dir
        )
        
        X, df_prep, prev_scores_map, _ = prepare_prediction_data(
            df_template, df_hist, target_col, prev_year,
            le_school, le_profile, school_stats, feature_cols
        )
        
        # Zero-anchor filter: only predict profiles with valid previous score
        valid_mask = df_prep['Prev_Year_Score'] > 0
        
        results = generate_predictions(
            model, X, df_prep, prev_scores_map, school_stats,
            valid_mask, gender=g,
            equi_preds=equi_preds if equi_preds else None,
            blend_weight=blend_weight
        )
        
        # Store R1 predictions for R2 derivation
        r1_pred_map = {}
        
        for r in results:
            key = (r.school, r.profile)
            r1_pred_map[f"{r.school}|{r.profile}"] = r.predicted
            
            if key not in all_results:
                all_results[key] = {'School': r.school, 'Profile': r.profile}
            
            lower, upper, confidence = compute_prediction_intervals(
                r.predicted, r.volatility, avg_volatility
            )
            
            all_results[key][f'R1_{g}_Predicted'] = round(r.predicted, 2)
            all_results[key][f'R1_{g}_Lower'] = round(lower, 2)
            all_results[key][f'R1_{g}_Upper'] = round(upper, 2)
            all_results[key][f'R1_{g}_Confidence'] = round(confidence, 1)
            all_results[key][f'R1_{g}_Volatility'] = round(r.volatility, 1)
            all_results[key][f'R1_{g}_Reliable'] = r.reliable
        
        reliable_count = sum(1 for r in results if r.reliable)
        metrics[f'R1_{g}'] = {
            'total': len(results),
            'reliable': reliable_count
        }
        
        # === ROUND 2: R1_pred + per-profile R2-R1 offset (hybrid) ===
        r2_target = f'R2_Min_{g}'
        if r2_target not in df_hist.columns:
            continue
        
        # Compute per-profile R2-R1 diffs and a global fallback
        r1_col = f'R1_Min_{g}'
        per_profile_r2_diffs = {}  # key → list of (r2 - r1) values
        global_r2_r1_diffs = []
        
        for (school, profile), group in df_hist.groupby(['School', 'Profile']):
            key = f"{school}|{profile}"
            diffs = []
            for _, row in group.iterrows():
                r1_val = row.get(r1_col, 0)
                r2_val = row.get(r2_target, 0)
                if r1_val > 0 and r2_val > 0:
                    diffs.append(r2_val - r1_val)
                    global_r2_r1_diffs.append(r2_val - r1_val)
            if diffs:
                per_profile_r2_diffs[key] = diffs
        
        global_avg_diff = np.mean(global_r2_r1_diffs) if global_r2_r1_diffs else 0
        global_std_diff = np.std(global_r2_r1_diffs) if global_r2_r1_diffs else 20
        
        logger.info(f"R2 {g}: hybrid method (global avg={global_avg_diff:.1f}, "
                    f"{len(per_profile_r2_diffs)} profiles with own R2 history)")
        
        r2_count = 0
        r2_reliable = 0
        
        for r in results:
            key = (r.school, r.profile)
            profile_key = f"{r.school}|{r.profile}"
            r1_pred = r1_pred_map.get(profile_key, 0)
            if r1_pred <= 0:
                continue
            
            # Use per-profile offset if available, otherwise global
            profile_diffs = per_profile_r2_diffs.get(profile_key)
            if profile_diffs:
                # Use median (more robust to outliers than mean for small samples)
                offset = np.median(profile_diffs)
                r2_vol = max(r.volatility, np.std(profile_diffs)) if len(profile_diffs) >= 2 else max(r.volatility, global_std_diff)
            else:
                offset = global_avg_diff
                r2_vol = max(r.volatility, global_std_diff)
            
            r2_pred = max(0, min(500, r1_pred + offset))
            
            lower, upper, confidence = compute_prediction_intervals(
                r2_pred, r2_vol, avg_volatility
            )
            
            all_results[key][f'R2_{g}_Predicted'] = round(r2_pred, 2)
            all_results[key][f'R2_{g}_Lower'] = round(lower, 2)
            all_results[key][f'R2_{g}_Upper'] = round(upper, 2)
            all_results[key][f'R2_{g}_Confidence'] = round(confidence, 1)
            all_results[key][f'R2_{g}_Volatility'] = round(r2_vol, 1)
            all_results[key][f'R2_{g}_Reliable'] = r.reliable
            
            r2_count += 1
            if r.reliable:
                r2_reliable += 1
        
        metrics[f'R2_{g}'] = {
            'total': r2_count,
            'reliable': r2_reliable
        }
    
    return all_results, metrics
