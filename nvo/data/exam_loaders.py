"""Data loading for exam score distribution features."""
import pandas as pd
import numpy as np
from pathlib import Path
from nvo.data.parsers import parse_bg_float
from nvo.utils.logger import get_logger

logger = get_logger("data.exam_loaders")


def load_exam_distribution(year: int, files_dir: str = "files") -> pd.DataFrame:
    """Load full exam score distribution with percentiles."""
    filepath = Path(files_dir) / str(year) / f"average_grades_BEL_MAT-{year}.xlsx"
    
    try:
        df = pd.read_excel(filepath, header=None)
        header_row = df[df[0].astype(str).str.contains('Точки', na=False)].index[0]
        data_start = header_row + 3
        
        max_score = 200.999 if year == 2025 else 200.499
        
        # Collect score intervals and counts
        scores_total, scores_male, scores_female = [], [], []
        counts_total, counts_male, counts_female = [], [], []
        
        for i, val in enumerate(df.iloc[data_start:, 0]):
            if pd.isna(val):
                continue
            s = str(val).strip()
            
            # Parse score range
            if '-' in s:
                parts = s.split('-')
                low = float(parts[0].strip().replace(',', '.'))
                high = float(parts[1].strip().replace(',', '.'))
                if low > max_score:
                    break
                score = (low + high) / 2
            elif s == '200':
                score = 200.0
            else:
                continue
            
            idx = data_start + i
            count_total = parse_bg_float(df.iloc[idx, 13])
            count_male = parse_bg_float(df.iloc[idx, 15])
            count_female = parse_bg_float(df.iloc[idx, 17])
            
            scores_total.append(score)
            counts_total.append(count_total)
            scores_male.append(score)
            counts_male.append(count_male)
            scores_female.append(score)
            counts_female.append(count_female)
        
        # Calculate percentiles for each gender
        def calc_percentiles(scores, counts):
            """Calculate key percentiles from distribution."""
            if not scores or sum(counts) == 0:
                return {}
            
            # Create cumulative distribution
            total = sum(counts)
            cumsum = np.cumsum(counts)
            percentiles = (cumsum / total) * 100
            
            # Find scores at key percentiles
            result = {}
            for p in [10, 25, 50, 75, 90, 95]:
                idx = np.searchsorted(percentiles, p)
                if idx < len(scores):
                    result[f'P{p}'] = scores[idx]
                else:
                    result[f'P{p}'] = scores[-1]
            
            # Add mean and std
            weighted_mean = sum(s * c for s, c in zip(scores, counts)) / total
            weighted_var = sum(c * (s - weighted_mean)**2 for s, c in zip(scores, counts)) / total
            result['Mean'] = weighted_mean
            result['Std'] = np.sqrt(weighted_var)
            
            return result
        
        total_stats = calc_percentiles(scores_total, counts_total)
        male_stats = calc_percentiles(scores_male, counts_male)
        female_stats = calc_percentiles(scores_female, counts_female)
        
        # Combine into single dict
        features = {}
        for key, val in total_stats.items():
            features[f'Exam_Total_{key}'] = val
        for key, val in male_stats.items():
            features[f'Exam_Male_{key}'] = val
        for key, val in female_stats.items():
            features[f'Exam_Female_{key}'] = val
        
        logger.info(f"Loaded exam distribution for {year}: {len(features)} features")
        return features
        
    except Exception as e:
        logger.error(f"Could not load exam distribution for {year}: {e}")
        return {}
