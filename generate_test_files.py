"""
Generate test exam score files with better/worse grade distributions
Usage: python generate_test_files.py
"""
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================
BASE_YEAR = 2025  # Year to use as baseline
TARGET_YEAR = 2026  # Year for generated files

# Scenarios: 'better' shifts distribution UP, 'worse' shifts DOWN
SCENARIOS = {
    'better': 0.80,  # Move 80% of students to higher scores
    'worse': 0.80    # Move 80% of students to lower scores
}

# ==========================================
# GENERATION LOGIC
# ==========================================

def shift_distribution(df, shift_percent, direction):
    """Shift combined BEL+MAT grade distribution"""
    df_new = df.copy()
    data_start = 3
    
    score_rows = [i for i in range(data_start, len(df_new)) if pd.notna(df_new.iloc[i, 0])]
    
    if direction == 'better':
        for i in range(len(score_rows) - 1):
            curr, next = score_rows[i], score_rows[i + 1]
            total = df_new.iloc[curr, 13] if pd.notna(df_new.iloc[curr, 13]) else 0
            male = df_new.iloc[curr, 15] if pd.notna(df_new.iloc[curr, 15]) else 0
            female = df_new.iloc[curr, 17] if pd.notna(df_new.iloc[curr, 17]) else 0
            
            if total > 0:
                move_total = int(total * shift_percent)
                move_male = int(move_total * (male / total)) if total > 0 else 0
                move_female = move_total - move_male
                
                df_new.iloc[curr, 13] -= move_total
                df_new.iloc[curr, 15] -= move_male
                df_new.iloc[curr, 17] -= move_female
                
                df_new.iloc[next, 13] = (df_new.iloc[next, 13] if pd.notna(df_new.iloc[next, 13]) else 0) + move_total
                df_new.iloc[next, 15] = (df_new.iloc[next, 15] if pd.notna(df_new.iloc[next, 15]) else 0) + move_male
                df_new.iloc[next, 17] = (df_new.iloc[next, 17] if pd.notna(df_new.iloc[next, 17]) else 0) + move_female
    else:
        for i in range(len(score_rows) - 1, 0, -1):
            curr, prev = score_rows[i], score_rows[i - 1]
            total = df_new.iloc[curr, 13] if pd.notna(df_new.iloc[curr, 13]) else 0
            male = df_new.iloc[curr, 15] if pd.notna(df_new.iloc[curr, 15]) else 0
            female = df_new.iloc[curr, 17] if pd.notna(df_new.iloc[curr, 17]) else 0
            
            if total > 0:
                move_total = int(total * shift_percent)
                move_male = int(move_total * (male / total)) if total > 0 else 0
                move_female = move_total - move_male
                
                df_new.iloc[curr, 13] -= move_total
                df_new.iloc[curr, 15] -= move_male
                df_new.iloc[curr, 17] -= move_female
                
                df_new.iloc[prev, 13] = (df_new.iloc[prev, 13] if pd.notna(df_new.iloc[prev, 13]) else 0) + move_total
                df_new.iloc[prev, 15] = (df_new.iloc[prev, 15] if pd.notna(df_new.iloc[prev, 15]) else 0) + move_male
                df_new.iloc[prev, 17] = (df_new.iloc[prev, 17] if pd.notna(df_new.iloc[prev, 17]) else 0) + move_female
    
    # Recalculate cumulative columns
    for col_count, col_cumul in [(13, 14), (15, 16), (17, 18)]:
        for i in range(len(score_rows)):
            cumulative = sum(df_new.iloc[score_rows[j], col_count] for j in range(i + 1, len(score_rows)) if pd.notna(df_new.iloc[score_rows[j], col_count]))
            df_new.iloc[score_rows[i], col_cumul] = cumulative
    
    return df_new

def generate_scenario(scenario_name, shift_percent, direction):
    """Generate test files for a scenario"""
    print(f"\n{'='*60}")
    print(f"Generating: {scenario_name.upper()} ({direction}, {shift_percent*100:.0f}% shift)")
    print(f"{'='*60}")
    
    output_dir = f'files/{TARGET_YEAR}_{scenario_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = f'files/{BASE_YEAR}/average_grades_BEL_MAT-{BASE_YEAR}.xlsx'
    output_file = f'{output_dir}/average_grades_BEL_MAT-{TARGET_YEAR}.xlsx'
    
    try:
        df = pd.read_excel(input_file, header=None)
        df_shifted = shift_distribution(df, shift_percent, direction)
        df_shifted.to_excel(output_file, index=False, header=False)
        print(f"  ✓ BEL_MAT: {output_file}")
    except Exception as e:
        print(f"  ✗ BEL_MAT: Error - {e}")
    
    print(f"  → Files saved to: {output_dir}/")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("Test Exam File Generator")
    print("="*60)
    print(f"Base year: {BASE_YEAR}")
    print(f"Target year: {TARGET_YEAR}")
    
    # Generate each scenario
    for scenario_name, shift_percent in SCENARIOS.items():
        generate_scenario(scenario_name, shift_percent, scenario_name)
    
    print("\n" + "="*60)
    print("✓ Generation complete!")
    print("="*60)
    print("\nTo use these files with predict.py:")
    print(f"  1. Copy files from 'files/{TARGET_YEAR}_better/' or 'files/{TARGET_YEAR}_worse/'")
    print(f"  2. Place them in 'files/{TARGET_YEAR}/'")
    print(f"  3. Update PREDICT_YEAR={TARGET_YEAR} in predict.py")
    print("  4. Run: python predict.py")
