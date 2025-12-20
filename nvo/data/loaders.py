"""Data loading functions for Excel files."""
import pandas as pd
import glob
from pathlib import Path
from typing import Dict, Optional
from nvo.data.parsers import parse_bg_float, find_header_row, extract_columns
from nvo.utils.logger import get_logger

logger = get_logger("data.loaders")


def load_exam_averages(year: int, files_dir: str = "files") -> Dict[str, float]:
    """Load BEL and MAT exam averages by gender from combined file."""
    filepath = Path(files_dir) / str(year) / f"average_grades_BEL_MAT-{year}.xlsx"
    
    try:
        df = pd.read_excel(filepath, header=None)
        header_row = df[df[0].astype(str).str.contains('Точки', na=False)].index[0]
        data_start = header_row + 3
        
        max_score = 200.999 if year == 2025 else 200.499
        scores, total_c, male_c, female_c = [], [], [], []
        
        for i, val in enumerate(df.iloc[data_start:, 0]):
            if pd.isna(val):
                continue
            s = str(val).strip()
            if '-' in s:
                parts = s.split('-')
                low = float(parts[0].strip().replace(',', '.'))
                high = float(parts[1].strip().replace(',', '.'))
                if low > max_score:
                    break
                scores.append((low + high) / 2)
            elif s == '200':
                scores.append(200.0)
            else:
                continue
            
            idx = data_start + i
            total_c.append(parse_bg_float(df.iloc[idx, 13]))
            male_c.append(parse_bg_float(df.iloc[idx, 15]))
            female_c.append(parse_bg_float(df.iloc[idx, 17]))
        
        scores = pd.Series(scores)
        total_c, male_c, female_c = pd.Series(total_c), pd.Series(male_c), pd.Series(female_c)
        
        combined_total = (scores * total_c).sum() / total_c.sum() if total_c.sum() > 0 else 0
        combined_male = (scores * male_c).sum() / male_c.sum() if male_c.sum() > 0 else 0
        combined_female = (scores * female_c).sum() / female_c.sum() if female_c.sum() > 0 else 0
        
        logger.info(f"Loaded exam averages for {year}: BEL+MAT Total={combined_total:.2f}")
        
        return {
            'BEL_Total': combined_total / 2,
            'BEL_Male': combined_male / 2,
            'BEL_Female': combined_female / 2,
            'MAT_Total': combined_total / 2,
            'MAT_Male': combined_male / 2,
            'MAT_Female': combined_female / 2
        }
    except Exception as e:
        logger.warning(f"Could not load exam averages for {year}: {e}")
        return {f'{s}_{g}': 0 for s in ['BEL', 'MAT'] for g in ['Total', 'Male', 'Female']}


def load_school_capacity(year: int, files_dir: str = "files") -> pd.DataFrame:
    """Load school capacity data."""
    filepath = Path(files_dir) / str(year) / f"schools_{year}.xlsx"
    
    try:
        df = pd.read_excel(filepath, header=None)
        df.columns = df.iloc[1]
        df = df.iloc[3:].reset_index(drop=True)
        
        capacity_df = pd.DataFrame({
            'School': df['Училище - име'],
            'Profile': df['Паралелка - име'],
            'Capacity_Total': df['Общо основание'].apply(parse_bg_float),
            'Capacity_Male': df['Мъже'].apply(parse_bg_float),
            'Capacity_Female': df['Жени'].apply(parse_bg_float)
        })
        
        logger.info(f"Loaded capacity data for {year}: {len(capacity_df)} entries")
        return capacity_df
    except Exception as e:
        logger.warning(f"Could not load capacity for {year}: {e}")
        return pd.DataFrame()


def load_rankings(year: int, files_dir: str = "files") -> pd.DataFrame:
    """Load all ranking rounds for a year."""
    files = sorted(glob.glob(str(Path(files_dir) / str(year) / f"klasirane_*_{year}.xlsx")))
    
    if not files:
        logger.error(f"No ranking files found for {year}")
        return pd.DataFrame()
    
    logger.info(f"Loading {len(files)} ranking files for {year}")
    
    all_rounds = []
    for i, filepath in enumerate(files, 1):
        try:
            df = pd.read_excel(filepath, header=None)
            header_row, col_map = find_header_row(df)
            
            if header_row == -1:
                logger.warning(f"Skipping {filepath}: no header found")
                continue
            
            data_df = df.iloc[header_row+1:].copy()
            
            round_df = pd.DataFrame({
                'School': data_df.iloc[:, col_map['School']],
                'Profile': data_df.iloc[:, col_map['Profile']],
                f'R{i}_Min_Total': data_df.iloc[:, col_map['Min_Total']].apply(parse_bg_float) if col_map['Min_Total'] else 0,
                f'R{i}_Min_Male': data_df.iloc[:, col_map['Min_Male']].apply(parse_bg_float) if col_map['Min_Male'] else 0,
                f'R{i}_Min_Female': data_df.iloc[:, col_map['Min_Female']].apply(parse_bg_float) if col_map['Min_Female'] else 0,
            })
            
            round_df = round_df.dropna(subset=['School']).query("~School.str.contains('Име', na=False)")
            all_rounds.append(round_df)
            logger.debug(f"Loaded round {i}: {len(round_df)} entries")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    
    if not all_rounds:
        return pd.DataFrame()
    
    result = all_rounds[0]
    for round_df in all_rounds[1:]:
        result = pd.merge(result, round_df, on=['School', 'Profile'], how='outer')
    
    logger.info(f"Merged rankings for {year}: {len(result)} school/profile combinations")
    return result
