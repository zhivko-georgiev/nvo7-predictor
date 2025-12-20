"""Prediction service."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from nvo.data.processors import build_dataset
from nvo.data.exam_loaders import load_exam_distribution
from nvo.models.trainer import train_model
from nvo.models.prediction_utils import (
    prepare_prediction_data,
    generate_predictions,
    compute_prediction_intervals,
)
from nvo.services.common import get_gender_list


def run_predictions(
    historical_years: List[int],
    predict_year: int,
    files_dir: str,
    model_params: dict,
    gender_filter: Optional[str] = None,
    school_filter: Optional[List[str]] = None
) -> Tuple[Dict, Dict]:
    """Run predictions for all rounds and genders.
    
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
    
    for round_num in [1, 2]:
        for g in genders:
            target_col = f'R{round_num}_Min_{g}'
            if target_col not in df_hist.columns:
                continue
            
            model, le_school, le_profile, feature_cols, school_stats = train_model(
                df_hist, target_col, round_num, model_params
            )
            
            if model is None:
                continue
            
            avg_volatility = np.mean([s['volatility'] for s in school_stats.values()])
            
            X, df_prep, prev_scores_map, mask = prepare_prediction_data(
                df_template, df_hist, target_col, prev_year,
                le_school, le_profile, school_stats, feature_cols
            )
            
            results = generate_predictions(
                model, X, df_prep, prev_scores_map, school_stats,
                pd.Series([True] * len(X), index=X.index)
            )
            
            for r in results:
                key = (r.school, r.profile)
                if key not in all_results:
                    all_results[key] = {'School': r.school, 'Profile': r.profile}
                
                lower, upper, confidence = compute_prediction_intervals(
                    r.predicted, r.volatility, avg_volatility
                )
                
                all_results[key][f'R{round_num}_{g}_Predicted'] = round(r.predicted, 2)
                all_results[key][f'R{round_num}_{g}_Lower'] = round(lower, 2)
                all_results[key][f'R{round_num}_{g}_Upper'] = round(upper, 2)
                all_results[key][f'R{round_num}_{g}_Confidence'] = round(confidence, 1)
                all_results[key][f'R{round_num}_{g}_Volatility'] = round(r.volatility, 1)
                all_results[key][f'R{round_num}_{g}_Years_Data'] = r.n_years
                all_results[key][f'R{round_num}_{g}_Reliable'] = r.reliable
            
            reliable_count = sum(1 for r in results if r.reliable)
            metrics[f'R{round_num}_{g}'] = {
                'total': len(results),
                'reliable': reliable_count
            }
    
    return all_results, metrics
