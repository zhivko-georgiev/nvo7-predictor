import pandas as pd
import glob

# ==========================================
# CONFIGURATION
# ==========================================
YEARS = [2024, 2025]
GENDER_FILTER = 'female'  # 'male', 'female', or None

# ==========================================
# HELPERS
# ==========================================

def parse_bg_float(value):
    """Converts Bulgarian decimal format to float: '493,25' -> 493.25"""
    if pd.isna(value) or value == '0': return 0.0
    if isinstance(value, (int, float)): return float(value)
    try:
        return float(str(value).strip().replace('"', '').replace(',', '.'))
    except ValueError:
        return 0.0

def find_header_row(df):
    """Finds row containing 'Мъже'/'Жени' headers and maps column indices"""
    for idx, row in df.iterrows():
        row_str = row.astype(str).str.lower().tolist()
        if row_str.count('мъже') >= 2:
            male_idx = [i for i, x in enumerate(row_str) if 'мъже' in x]
            female_idx = [i for i, x in enumerate(row_str) if 'жени' in x]
            total_idx = [i for i, x in enumerate(row_str) if 'общо' in x]
            name_idx = [i for i, x in enumerate(row_str) if 'име' in x]
            
            return idx, {
                'School': name_idx[0] if len(name_idx) >= 2 else 2,
                'Profile': name_idx[1] if len(name_idx) >= 2 else 4,
                'Min_Male': male_idx[0] if male_idx else None,
                'Min_Female': female_idx[0] if female_idx else None,
                'Min_Total': total_idx[0] if total_idx else None,
                'Max_Male': male_idx[1] if len(male_idx) > 1 else None,
                'Max_Female': female_idx[1] if len(female_idx) > 1 else None,
                'Max_Total': total_idx[1] if len(total_idx) > 1 else None,
            }
    return -1, {}

def extract_columns(df_data, col_map, stage_name):
    """Extracts and formats columns from raw data"""
    result = pd.DataFrame({
        'School': df_data.iloc[:, col_map['School']],
        'Profile': df_data.iloc[:, col_map['Profile']]
    })
    
    for key, col_name in [
        ('Min_Total', f'Min_Total_{stage_name}'),
        ('Min_Male', f'Min_Male_{stage_name}'),
        ('Min_Female', f'Min_Female_{stage_name}'),
        ('Max_Total', f'Max_Total_{stage_name}'),
        ('Max_Male', f'Max_Male_{stage_name}'),
        ('Max_Female', f'Max_Female_{stage_name}')
    ]:
        if col_map.get(key) is not None:
            result[col_name] = df_data.iloc[:, col_map[key]].apply(parse_bg_float)
        else:
            result[col_name] = 0.0
    
    return result.dropna(subset=['School']).query("~School.str.contains('Име', na=False)")

def load_file(filepath, stage_name):
    """Loads and processes a single klasirane file"""
    print(f"Processing: {filepath} ({stage_name})...")
    
    try:
        df = pd.read_excel(filepath, header=None)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    header_row, col_map = find_header_row(df)
    if header_row == -1:
        print(f"ERROR: Could not find header row in {filepath}")
        return None
    
    print(f"  -> Headers at row {header_row}")
    return extract_columns(df.iloc[header_row+1:].copy(), col_map, stage_name)

def calculate_changes(df, stage_tag, prev_tag):
    """Calculates score changes between stages"""
    for metric in ['Min_Total', 'Min_Male', 'Min_Female']:
        curr_col = f"{metric}_{stage_tag}"
        prev_col = f"{metric}_{prev_tag}"
        if curr_col in df.columns and prev_col in df.columns:
            df[f"Change_{metric}_{stage_tag}_vs_{prev_tag}"] = (
                df[curr_col].fillna(0) - df[prev_col].fillna(0)
            )

def apply_gender_filter(df, gender):
    """Filters columns by gender"""
    target = "Male" if gender.lower().startswith('m') else "Female"
    cols = ['School', 'Profile'] + [c for c in df.columns if target in c]
    return df[cols]

def process_year(year):
    """Processes all klasirane files for a given year"""
    print(f"\n{'='*60}\nProcessing Year: {year}\n{'='*60}")
    
    files = sorted(glob.glob(f'files/{year}/klasirane_*_{year}.xlsx'))
    if not files:
        print(f"No klasirane files found for {year}")
        return
    
    print(f"Found {len(files)} files")
    
    # Load and merge files
    df = load_file(files[0], "R1")
    if df is None: return
    
    for i, filepath in enumerate(files[1:], start=2):
        stage = f"R{i}"
        next_df = load_file(filepath, stage)
        if next_df is not None:
            df = pd.merge(df, next_df, on=['School', 'Profile'], how='outer')
            calculate_changes(df, stage, f"R{i-1}")
    
    # Apply filter and save
    if GENDER_FILTER:
        print(f"Applying {GENDER_FILTER} filter...")
        df = apply_gender_filter(df, GENDER_FILTER)
    
    output = f'BG_Schools_Analysis_{year}_{GENDER_FILTER or "Full"}.xlsx'
    df.fillna(0).to_excel(output, index=False)
    print(f"✓ Saved to {output}")

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    for year in YEARS:
        process_year(year)
