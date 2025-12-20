"""Data parsing utilities for Bulgarian format Excel files."""
import pandas as pd
from typing import Tuple, Dict, Optional
from nvo.utils.logger import get_logger

logger = get_logger("data.parsers")


def parse_bg_float(value) -> float:
    """Convert Bulgarian decimal format to float: '493,25' -> 493.25"""
    if pd.isna(value) or value == '0':
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace('"', '').replace(',', '.'))
    except ValueError:
        logger.warning(f"Could not parse value: {value}")
        return 0.0


def find_header_row(df: pd.DataFrame) -> Tuple[int, Dict[str, Optional[int]]]:
    """Find row containing 'Мъже'/'Жени' headers and map column indices."""
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
    
    logger.error("Could not find header row in DataFrame")
    return -1, {}


def extract_columns(df_data: pd.DataFrame, col_map: Dict, stage_name: str) -> pd.DataFrame:
    """Extract and format columns from raw data."""
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
