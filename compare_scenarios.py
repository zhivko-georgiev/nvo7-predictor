import pandas as pd
import shutil
import os
from predict import build_dataset, train_model, predict_round, load_exam_averages, load_rankings, load_school_capacity, prepare_features, PREDICT_YEAR, GENDER_FILTER, HISTORICAL_YEARS, SCHOOLS_OF_INTEREST
import numpy as np

SCENARIOS = ['better', 'worse']

print("="*60)
print("Running Predictions for Multiple Scenarios")
print("="*60)

results = {}

for scenario in SCENARIOS:
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"{'='*60}")
    
    # Copy scenario file to main folder
    src = f'files/{PREDICT_YEAR}_{scenario}/average_grades_BEL_MAT-{PREDICT_YEAR}.xlsx'
    dst = f'files/{PREDICT_YEAR}/average_grades_BEL_MAT-{PREDICT_YEAR}.xlsx'
    shutil.copy(src, dst)
    print(f"Using: {src}")
    
    # Load exam scores
    print(f"\nLoading exam scores for {PREDICT_YEAR}...")
    predict_exam_scores = load_exam_averages(PREDICT_YEAR)
    print(f"✓ Loaded: BEL={predict_exam_scores['BEL_Total']:.2f}, MAT={predict_exam_scores['MAT_Total']:.2f}")
    
    # Load historical data
    print("\nBuilding dataset...")
    df = build_dataset()
    print(f"Loaded {len(df)} historical records")
    
    # Get base template
    last_year = HISTORICAL_YEARS[-1]
    base_df = load_rankings(last_year)[['School', 'Profile']].drop_duplicates()
    last_year_full = load_rankings(last_year)
    base_df = pd.merge(base_df, last_year_full, on=['School', 'Profile'], how='left')
    
    if SCHOOLS_OF_INTEREST:
        print(f"\nFiltering for schools containing: {SCHOOLS_OF_INTEREST}")
        mask = base_df['School'].str.contains('|'.join(SCHOOLS_OF_INTEREST), case=False, na=False)
        base_df = base_df[mask]
        print(f"  → {len(base_df)} school/profile combinations selected")
    
    capacity = load_school_capacity(last_year)
    if not capacity.empty:
        base_df = pd.merge(base_df, capacity, on=['School', 'Profile'], how='left')
    
    base_df, le_school, le_profile = prepare_features(base_df)
    
    # Train and predict
    predictions = base_df[['School', 'Profile']].copy()
    prev_predictions = base_df.copy()
    
    for round_num in range(1, 5):
        for gender in ['Total', 'Male', 'Female']:
            target_col = f'R{round_num}_Min_{gender}'
            if target_col not in df.columns:
                continue
            
            model, _, _, feature_cols = train_model(df, target_col, round_num)
            preds, confidence = predict_round(model, prev_predictions, feature_cols, predict_exam_scores, round_num, prev_predictions)
            
            predictions[f'R{round_num}_{gender}_Predicted'] = preds
            predictions[f'R{round_num}_{gender}_Confidence'] = confidence
            prev_predictions[target_col] = preds
    
    # Sort and filter
    if 'R1_Total_Predicted' in predictions.columns:
        predictions = predictions.sort_values('R1_Total_Predicted', ascending=False)
    
    if GENDER_FILTER:
        target = 'Female' if GENDER_FILTER.lower().startswith('f') else 'Male'
        cols_to_keep = ['School', 'Profile']
        for col in predictions.columns:
            if col not in cols_to_keep and target in col:
                cols_to_keep.append(col)
        predictions = predictions[cols_to_keep]
        
        if f'R1_{target}_Predicted' in predictions.columns:
            predictions = predictions.sort_values(f'R1_{target}_Predicted', ascending=False)
    
    # Save scenario-specific file
    gender_suffix = GENDER_FILTER if GENDER_FILTER else "Full"
    output = f'Predictions_{PREDICT_YEAR}_{scenario}_{gender_suffix}.xlsx'
    predictions.to_excel(output, index=False)
    print(f"\n✓ Saved to {output}")
    
    results[scenario] = predictions.copy()

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

target = 'Female' if GENDER_FILTER and GENDER_FILTER.lower().startswith('f') else 'Male' if GENDER_FILTER else 'Total'
pred_col = f'R1_{target}_Predicted'

comparison = results['better'][['School', 'Profile']].copy()
comparison['Better'] = results['better'][pred_col]
comparison['Worse'] = results['worse'][pred_col]
comparison['Difference'] = comparison['Better'] - comparison['Worse']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print(f"\nPredicted R1 {target} Scores Comparison:")
print(comparison.to_string(index=False))

print(f"\n{'='*60}")
print("INTERPRETATION:")
print(f"{'='*60}")
avg_diff = comparison['Difference'].mean()
if avg_diff < 0:
    print(f"⚠ Better exam scores predict LOWER admission cutoffs ({avg_diff:.2f} points)")
    print("This can happen when:")
    print("  • More qualified students spread across more schools")
    print("  • Increased competition at top-tier schools pushes students down")
    print("  • Model predicts based on historical patterns")
else:
    print(f"✓ Better exam scores predict HIGHER admission cutoffs (+{avg_diff:.2f} points)")
    print("This indicates stronger competition for these profiles")

# Save comparison
output = f'Comparison_{PREDICT_YEAR}_{GENDER_FILTER or "Full"}.xlsx'
comparison.to_excel(output, index=False)
print(f"\n✓ Comparison saved to {output}")
