"""Prediction with gender-specific blending and model caching."""
import pandas as pd
import numpy as np
from typing import Dict, List
from nvo.data.processors import build_dataset
from nvo.data.exam_loaders import load_exam_distribution
from nvo.models.trainer import train_model, prepare_features
from nvo.models.persistence import save_model, load_model, is_model_valid
from nvo.utils.logger import get_logger

logger = get_logger("models.predictor")

# Gender-specific blend weights (model vs naive)
BLEND_WEIGHTS = {
    'Female': 0.6,  # 60% model, 40% naive
    'Male': 0.0,    # 0% model, 100% naive (model doesn't help for male)
    'Total': 0.4,   # 40% model, 60% naive
}


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


def predict_with_intervals(
    historical_years: List[int],
    predict_year: int,
    files_dir: str,
    model_params: dict,
    gender_filter: str = None,
    school_filter: List[str] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """Generate predictions for R1 and R2."""
    
    logger.info(f"Building dataset for prediction...")
    df_hist = build_dataset(historical_years, files_dir)
    
    template_year = max(historical_years)
    logger.info(f"Using {template_year} as template for {predict_year} predictions")
    
    df_template = df_hist[df_hist['Year'] == template_year].copy()
    
    logger.info(f"Loading {predict_year} exam scores...")
    exam_features = load_exam_distribution(predict_year, files_dir)
    if exam_features:
        logger.info(f"Loaded {len(exam_features)} exam features for {predict_year}")
    
    if df_template.empty:
        logger.error(f"No template data found")
        return pd.DataFrame()
    
    if school_filter:
        mask = df_template['School'].str.contains('|'.join(school_filter), case=False, na=False)
        df_template = df_template[mask]
        logger.info(f"Filtered to {len(df_template)} schools matching: {school_filter}")
    
    genders = ['Female'] if gender_filter and gender_filter.lower().startswith('f') else \
              ['Male'] if gender_filter and gender_filter.lower().startswith('m') else \
              ['Total', 'Male', 'Female']
    
    all_results = {}
    
    for gender in genders:
        model_weight = BLEND_WEIGHTS.get(gender, 0.4)
        
        # === ROUND 1 ===
        target_col = f'R1_Min_{gender}'
        if target_col not in df_hist.columns:
            continue
        
        model, le_school, le_profile, feature_cols, school_stats = _get_or_train_model(
            df_hist, target_col, 1, model_params, gender, historical_years, use_cache
        )
        
        if model is None:
            continue
        
        avg_volatility = np.mean([s['volatility'] for s in school_stats.values()])
        
        df_pred = df_template[['School', 'Profile']].copy()
        df_pred['Year'] = predict_year
        df_pred['Key'] = df_pred['School'] + '|' + df_pred['Profile']
        df_pred['Prev_Year_Score'] = df_template[target_col].values
        
        for idx in range(len(df_pred)):
            key = df_pred.iloc[idx]['Key']
            stats = school_stats.get(key, {})
            df_pred.loc[df_pred.index[idx], 'School_Mean'] = stats.get('mean', df_pred.iloc[idx]['Prev_Year_Score'])
            df_pred.loc[df_pred.index[idx], 'School_Volatility'] = stats.get('volatility', avg_volatility)
            df_pred.loc[df_pred.index[idx], 'School_Trend'] = stats.get('trend', 0)
            df_pred.loc[df_pred.index[idx], 'School_Acceleration'] = stats.get('acceleration', 0)
        
        df_pred['Dist_From_Mean'] = df_pred['Prev_Year_Score'] - df_pred['School_Mean']
        
        if exam_features:
            for key, val in exam_features.items():
                df_pred[key] = val
        else:
            for col in [c for c in df_template.columns if c.startswith('Exam_')]:
                df_pred[col] = df_template[col].values
        
        df_pred_enc, _, _ = prepare_features(df_pred, le_school, le_profile)
        
        for col in feature_cols:
            if col not in df_pred_enc.columns:
                df_pred_enc[col] = 0
        
        X_pred = df_pred_enc[feature_cols].fillna(0)
        predicted_change = model.predict(X_pred)
        
        r1_predictions = {}
        
        for idx in range(len(df_pred)):
            school = df_pred.iloc[idx]['School']
            profile = df_pred.iloc[idx]['Profile']
            key = df_pred.iloc[idx]['Key']
            prev_score = df_pred.iloc[idx]['Prev_Year_Score']
            volatility = df_pred.iloc[idx]['School_Volatility']
            
            model_pred = prev_score + predicted_change[idx]
            pred = model_weight * model_pred + (1 - model_weight) * prev_score
            pred = max(0, pred)
            
            r1_predictions[key] = pred
            
            lower = max(0, pred - 1.5 * volatility)
            upper = pred + 1.5 * volatility
            confidence = max(0, min(100, 100 - (volatility / avg_volatility) * 40))
            
            result_key = (school, profile)
            if result_key not in all_results:
                all_results[result_key] = {'School': school, 'Profile': profile}
            
            all_results[result_key][f'R1_{gender}_Predicted'] = round(pred, 2)
            all_results[result_key][f'R1_{gender}_Lower'] = round(lower, 2)
            all_results[result_key][f'R1_{gender}_Upper'] = round(upper, 2)
            all_results[result_key][f'R1_{gender}_Confidence'] = round(confidence, 1)
            all_results[result_key][f'R1_{gender}_Volatility'] = round(volatility, 1)
        
        # === ROUND 2 ===
        r2_target = f'R2_Min_{gender}'
        if r2_target not in df_hist.columns:
            continue
        
        logger.info(f"Predicting R2 {gender} from R1...")
        
        r2_r1_diffs = []
        for (school, profile), group in df_hist.groupby(['School', 'Profile']):
            for _, row in group.iterrows():
                r1 = row.get(f'R1_Min_{gender}', 0)
                r2 = row.get(f'R2_Min_{gender}', 0)
                if r1 > 0 and r2 > 0:
                    r2_r1_diffs.append(r2 - r1)
        
        avg_r2_r1_diff = np.mean(r2_r1_diffs) if r2_r1_diffs else 0
        std_r2_r1_diff = np.std(r2_r1_diffs) if r2_r1_diffs else 20
        
        for idx in range(len(df_pred)):
            school = df_pred.iloc[idx]['School']
            profile = df_pred.iloc[idx]['Profile']
            key = df_pred.iloc[idx]['Key']
            volatility = df_pred.iloc[idx]['School_Volatility']
            
            r1_pred = r1_predictions.get(key, 0)
            if r1_pred <= 0:
                continue
            
            r2_pred = max(0, r1_pred + avg_r2_r1_diff)
            
            r2_vol = max(volatility, std_r2_r1_diff)
            lower = max(0, r2_pred - 1.5 * r2_vol)
            upper = r2_pred + 1.5 * r2_vol
            confidence = max(0, min(100, 100 - (r2_vol / avg_volatility) * 40))
            
            result_key = (school, profile)
            all_results[result_key][f'R2_{gender}_Predicted'] = round(r2_pred, 2)
            all_results[result_key][f'R2_{gender}_Lower'] = round(lower, 2)
            all_results[result_key][f'R2_{gender}_Upper'] = round(upper, 2)
            all_results[result_key][f'R2_{gender}_Confidence'] = round(confidence, 1)
            all_results[result_key][f'R2_{gender}_Volatility'] = round(r2_vol, 1)
    
    if not all_results:
        return pd.DataFrame()
    
    final_df = pd.DataFrame(list(all_results.values()))
    
    sort_col = [c for c in final_df.columns if c.startswith('R1_') and c.endswith('_Predicted')]
    if sort_col:
        final_df = final_df.sort_values(sort_col[0], ascending=False)
    
    logger.info(f"Generated predictions for {len(final_df)} school-profile combinations")
    return final_df
