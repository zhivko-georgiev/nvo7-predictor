import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
HISTORICAL_YEARS = [2022, 2023, 2024, 2025]
PREDICT_YEAR = 2026  # Year to predict
GENDER_FILTER = 'female'  # 'male', 'female', or None for both
SCENARIO_NAME = 'worse'  # Scenario name for output file ('better', 'worse', etc.)

# Filter by specific schools (None = all schools)
SCHOOLS_OF_INTEREST = ['119 Средно училище "Академик Михаил Арнаудов"', '35. СРЕДНО ЕЗИКОВО УЧИЛИЩЕ "ДОБРИ ВОЙНИКОВ"', '18 Средно училище "Уилям Гладстон"']  # Example: ['Математическа', 'Английска']

# Exam scores will be auto-loaded from files/{PREDICT_YEAR}/average_grades_*.xlsx
# Just place the files there when you get them!

# ==========================================
# DATA LOADING
# ==========================================

def parse_bg_float(value):
    if pd.isna(value) or value == '0': return 0.0
    if isinstance(value, (int, float)): return float(value)
    try:
        return float(str(value).strip().replace('"', '').replace(',', '.'))
    except ValueError:
        return 0.0

def load_exam_averages(year):
    """Load BEL and MAT exam averages by gender from combined file"""
    try:
        df = pd.read_excel(f'files/{year}/average_grades_BEL_MAT-{year}.xlsx', header=None)
        header_row = df[df[0].astype(str).str.contains('Точки', na=False)].index[0]
        data_start = header_row + 3
        
        max_score = 200.999 if year == 2025 else 200.499
        scores, total_c, male_c, female_c = [], [], [], []
        
        for i, val in enumerate(df.iloc[data_start:, 0]):
            if pd.isna(val): continue
            s = str(val).strip()
            if '-' in s:
                parts = s.split('-')
                low = float(parts[0].strip().replace(',', '.'))
                high = float(parts[1].strip().replace(',', '.'))
                if low > max_score: break
                scores.append((low + high) / 2)
            elif s == '200':
                scores.append(200.0)
            else: continue
            
            idx = data_start + i
            total_c.append(parse_bg_float(df.iloc[idx, 13]))
            male_c.append(parse_bg_float(df.iloc[idx, 15]))
            female_c.append(parse_bg_float(df.iloc[idx, 17]))
        
        scores = pd.Series(scores)
        total_c, male_c, female_c = pd.Series(total_c), pd.Series(male_c), pd.Series(female_c)
        
        combined_total = (scores * total_c).sum() / total_c.sum() if total_c.sum() > 0 else 0
        combined_male = (scores * male_c).sum() / male_c.sum() if male_c.sum() > 0 else 0
        combined_female = (scores * female_c).sum() / female_c.sum() if female_c.sum() > 0 else 0
        
        return {
            'BEL_Total': combined_total / 2,
            'BEL_Male': combined_male / 2,
            'BEL_Female': combined_female / 2,
            'MAT_Total': combined_total / 2,
            'MAT_Male': combined_male / 2,
            'MAT_Female': combined_female / 2
        }
    except Exception as e:
        print(f"Warning: Could not load exam averages for {year}: {e}")
        return {f'{s}_{g}': 0 for s in ['BEL', 'MAT'] for g in ['Total', 'Male', 'Female']}

def load_school_capacity(year):
    """Load school capacity data"""
    try:
        df = pd.read_excel(f'files/{year}/schools_{year}.xlsx', header=None)
        df.columns = df.iloc[1]
        df = df.iloc[3:].reset_index(drop=True)
        
        capacity_df = pd.DataFrame({
            'School': df['Училище - име'],
            'Profile': df['Паралелка - име'],
            'Capacity_Total': df['Общо основание'].apply(parse_bg_float),
            'Capacity_Male': df['Мъже'].apply(parse_bg_float),
            'Capacity_Female': df['Жени'].apply(parse_bg_float)
        })
        return capacity_df
    except Exception as e:
        print(f"Warning: Could not load capacity for {year}: {e}")
        return pd.DataFrame()

def load_rankings(year):
    """Load all ranking rounds for a year"""
    import glob
    files = sorted(glob.glob(f'files/{year}/klasirane_*_{year}.xlsx'))
    
    all_rounds = []
    for i, filepath in enumerate(files, 1):
        df = pd.read_excel(filepath, header=None)
        
        # Find header row
        header_row = -1
        for idx, row in df.iterrows():
            row_str = row.astype(str).str.lower().tolist()
            if row_str.count('мъже') >= 2:
                header_row = idx
                break
        
        if header_row == -1:
            continue
        
        # Extract data
        data_df = df.iloc[header_row+1:].copy()
        
        # Find column indices
        row_str = df.iloc[header_row].astype(str).str.lower().tolist()
        male_idx = [j for j, x in enumerate(row_str) if 'мъже' in x]
        female_idx = [j for j, x in enumerate(row_str) if 'жени' in x]
        total_idx = [j for j, x in enumerate(row_str) if 'общо' in x]
        name_idx = [j for j, x in enumerate(row_str) if 'име' in x]
        
        school_col = name_idx[0] if len(name_idx) >= 2 else 2
        profile_col = name_idx[1] if len(name_idx) >= 2 else 4
        
        round_df = pd.DataFrame({
            'School': data_df.iloc[:, school_col],
            'Profile': data_df.iloc[:, profile_col],
            f'R{i}_Min_Total': data_df.iloc[:, total_idx[0]].apply(parse_bg_float) if total_idx else 0,
            f'R{i}_Min_Male': data_df.iloc[:, male_idx[0]].apply(parse_bg_float) if male_idx else 0,
            f'R{i}_Min_Female': data_df.iloc[:, female_idx[0]].apply(parse_bg_float) if female_idx else 0,
        })
        
        round_df = round_df.dropna(subset=['School']).query("~School.str.contains('Име', na=False)")
        all_rounds.append(round_df)
    
    # Merge all rounds
    result = all_rounds[0]
    for round_df in all_rounds[1:]:
        result = pd.merge(result, round_df, on=['School', 'Profile'], how='outer')
    
    return result

def build_dataset():
    """Build complete dataset with all features"""
    all_data = []
    
    for year in HISTORICAL_YEARS:
        print(f"Loading data for {year}...")
        
        rankings = load_rankings(year)
        exam_scores = load_exam_averages(year)
        capacity = load_school_capacity(year)
        
        # Add year and exam scores
        rankings['Year'] = year
        for key, val in exam_scores.items():
            rankings[key] = val
        
        # Merge capacity
        if not capacity.empty:
            rankings = pd.merge(rankings, capacity, on=['School', 'Profile'], how='left')
        
        all_data.append(rankings)
    
    return pd.concat(all_data, ignore_index=True)

# ==========================================
# MODEL TRAINING
# ==========================================

def prepare_features(df, le_school=None, le_profile=None):
    """Encode categorical features"""
    df = df.copy()
    
    if le_school is None:
        le_school = LabelEncoder()
        df['School_Encoded'] = le_school.fit_transform(df['School'].astype(str))
    else:
        df['School_Encoded'] = le_school.transform(df['School'].astype(str))
    
    if le_profile is None:
        le_profile = LabelEncoder()
        df['Profile_Encoded'] = le_profile.fit_transform(df['Profile'].astype(str))
    else:
        df['Profile_Encoded'] = le_profile.transform(df['Profile'].astype(str))
    
    return df, le_school, le_profile

def train_model(df, target_col, round_num):
    """Train XGBoost model for a specific round"""
    print(f"\nTraining model for {target_col}...")
    
    # Prepare features
    df, le_school, le_profile = prepare_features(df)
    
    feature_cols = ['Year', 'School_Encoded', 'Profile_Encoded', 
                    'BEL_Total', 'MAT_Total', 'BEL_Male', 'BEL_Female', 
                    'MAT_Male', 'MAT_Female']
    
    # Add capacity if available
    if 'Capacity_Total' in df.columns:
        feature_cols.extend(['Capacity_Total', 'Capacity_Male', 'Capacity_Female'])
    
    # Add previous round scores as features (if not R1)
    if round_num > 1:
        for prev_round in range(1, round_num):
            prev_cols = [f'R{prev_round}_Min_Total', f'R{prev_round}_Min_Male', f'R{prev_round}_Min_Female']
            feature_cols.extend([c for c in prev_cols if c in df.columns])
    
    # Add previous year's score for same school+profile
    df_sorted = df.sort_values(['School', 'Profile', 'Year'])
    df['Prev_Year_Score'] = df_sorted.groupby(['School', 'Profile'])[target_col].shift(1)
    
    # IMPORTANT: Add year-over-year trend (optimistic feature)
    # This captures if a profile is improving or declining
    df['YoY_Trend'] = df_sorted.groupby(['School', 'Profile'])[target_col].diff()
    
    if 'Prev_Year_Score' in df.columns:
        feature_cols.append('Prev_Year_Score')
    if 'YoY_Trend' in df.columns:
        feature_cols.append('YoY_Trend')
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    
    # Remove rows with zero target (no data)
    mask = y > 0
    X, y = X[mask], y[mask]
    
    if len(X) < 10:
        print(f"  Warning: Only {len(X)} samples, skipping...")
        return None, le_school, le_profile, feature_cols
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost - adjusted for trend-following
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha=0.05,  # Reduced regularization to allow trends
        reg_lambda=0.5
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"  Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    
    return model, le_school, le_profile, feature_cols

# ==========================================
# PREDICTION
# ==========================================

def predict_round(model, base_df, feature_cols, exam_scores, round_num, prev_predictions=None):
    """Predict scores for a specific round"""
    pred_df = base_df.copy()
    pred_df['Year'] = PREDICT_YEAR
    
    # Add exam scores
    for key, val in exam_scores.items():
        pred_df[key] = val
    
    # Add previous round predictions if available
    if prev_predictions is not None and round_num > 1:
        for prev_round in range(1, round_num):
            for gender in ['Total', 'Male', 'Female']:
                col = f'R{prev_round}_Min_{gender}'
                if col in prev_predictions.columns:
                    pred_df[col] = prev_predictions[col]
    
    # Add Prev_Year_Score if needed - use last year's R1 score for this profile
    if 'Prev_Year_Score' in feature_cols and 'Prev_Year_Score' not in pred_df.columns:
        # Use R1 scores from base_df (which has 2025 data)
        for gender in ['Total', 'Male', 'Female']:
            col = f'R1_Min_{gender}'
            if col in base_df.columns:
                pred_df['Prev_Year_Score'] = base_df[col]
                break
        if 'Prev_Year_Score' not in pred_df.columns:
            pred_df['Prev_Year_Score'] = 0
    
    # Add YoY_Trend if needed - calculate from 2024 to 2025 change
    if 'YoY_Trend' in feature_cols and 'YoY_Trend' not in pred_df.columns:
        # Try to get 2024 and 2025 scores to calculate trend
        if 'R1_Min_Total' in base_df.columns:
            # We have 2025 scores, need to get 2024 scores
            # For now, estimate trend from exam score changes
            # If exams improved, assume positive trend
            exam_2025 = (exam_scores.get('BEL_Total', 0) + exam_scores.get('MAT_Total', 0)) / 2
            # Rough estimate: if exams are above 60, assume positive trend
            pred_df['YoY_Trend'] = max(0, (exam_2025 - 60) * 0.5)  # Optimistic bias
        else:
            pred_df['YoY_Trend'] = 0
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0
    
    X_pred = pred_df[feature_cols].fillna(0)
    predictions = model.predict(X_pred)
    
    # Estimate confidence
    confidence = 100 - (np.std(predictions) / np.mean(predictions) * 100) if np.mean(predictions) > 0 else 50
    
    return predictions, confidence

def run_predictions():
    """Main prediction pipeline"""
    print("="*60)
    print(f"Building ML Prediction Model for {PREDICT_YEAR}")
    print("="*60)
    
    # Try to load exam scores for prediction year
    print(f"\nLoading exam scores for {PREDICT_YEAR}...")
    try:
        predict_exam_scores = load_exam_averages(PREDICT_YEAR)
        print(f"✓ Loaded: BEL={predict_exam_scores['BEL_Total']:.2f}, MAT={predict_exam_scores['MAT_Total']:.2f}")
    except Exception as e:
        print(f"⚠ Could not load exam scores for {PREDICT_YEAR}: {e}")
        print("Using average of historical years as fallback...")
        # Fallback: use average of historical years
        hist_scores = [load_exam_averages(y) for y in HISTORICAL_YEARS]
        predict_exam_scores = {k: np.mean([s[k] for s in hist_scores]) for k in hist_scores[0].keys()}
    
    # Load historical data
    df = build_dataset()
    print(f"\nLoaded {len(df)} historical records")
    
    # Get base template for predictions (last year's schools)
    last_year = HISTORICAL_YEARS[-1]
    base_df = load_rankings(last_year)[['School', 'Profile']].drop_duplicates()
    
    # Load last year's full rankings to get R1 scores for Prev_Year_Score feature
    last_year_full = load_rankings(last_year)
    base_df = pd.merge(base_df, last_year_full, on=['School', 'Profile'], how='left')
    
    # Filter by schools of interest
    if SCHOOLS_OF_INTEREST:
        print(f"\nFiltering for schools containing: {SCHOOLS_OF_INTEREST}")
        mask = base_df['School'].str.contains('|'.join(SCHOOLS_OF_INTEREST), case=False, na=False)
        base_df = base_df[mask]
        print(f"  → {len(base_df)} school/profile combinations selected")
    
    capacity = load_school_capacity(last_year)
    if not capacity.empty:
        base_df = pd.merge(base_df, capacity, on=['School', 'Profile'], how='left')
    
    # Prepare encoders
    base_df, le_school, le_profile = prepare_features(base_df)
    
    # Train models and predict for each round
    results = base_df[['School', 'Profile']].copy()
    prev_predictions = base_df.copy()  # Start with base_df that has all columns
    
    for round_num in range(1, 5):  # R1 to R4
        # Predict for all three genders
        for gender in ['Total', 'Male', 'Female']:
            target_col = f'R{round_num}_Min_{gender}'
            
            if target_col not in df.columns:
                continue
            
            # Train model for this gender
            model, _, _, feature_cols = train_model(df, target_col, round_num)
            
            # Predict
            predictions, confidence = predict_round(model, prev_predictions, feature_cols, predict_exam_scores, round_num, prev_predictions)
            
            # Store predictions
            results[f'R{round_num}_{gender}_Predicted'] = predictions
            results[f'R{round_num}_{gender}_Confidence'] = confidence
            
            # Add to prev_predictions for next round
            prev_predictions[target_col] = predictions
    
    # Sort by R1 Total prediction
    if 'R1_Total_Predicted' in results.columns:
        results = results.sort_values('R1_Total_Predicted', ascending=False)
    
    # Filter columns based on GENDER_FILTER
    if GENDER_FILTER:
        target = 'Female' if GENDER_FILTER.lower().startswith('f') else 'Male'
        cols_to_keep = ['School', 'Profile']
        for col in results.columns:
            if col not in cols_to_keep and target in col:
                cols_to_keep.append(col)
        results = results[cols_to_keep]
        
        # Sort by female R1 prediction if filtering for female
        if f'R1_{target}_Predicted' in results.columns:
            results = results.sort_values(f'R1_{target}_Predicted', ascending=False)
    
    # Save
    gender_suffix = GENDER_FILTER if GENDER_FILTER else "Full"
    output = f'Predictions_{PREDICT_YEAR}_{SCENARIO_NAME}_{gender_suffix}.xlsx'
    results.to_excel(output, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Predictions saved to {output}")
    print(f"{'='*60}")
    
    return results

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    predictions = run_predictions()
    
    if GENDER_FILTER:
        target = 'Female' if GENDER_FILTER.lower().startswith('f') else 'Male'
        print(f"\nTop 10 Predicted Schools (Round 1 - {target}):")
        cols = ['School', 'Profile', f'R1_{target}_Predicted', f'R1_{target}_Confidence']
    else:
        print("\nTop 10 Predicted Schools (Round 1 - Total):")
        cols = ['School', 'Profile', 'R1_Total_Predicted', 'R1_Total_Confidence']
    
    display_cols = [c for c in cols if c in predictions.columns]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 60)
    print(predictions[display_cols].head(10).to_string(index=False))
