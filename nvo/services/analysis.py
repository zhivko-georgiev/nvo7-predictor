"""Analysis service."""
from typing import Dict, List, Optional, Tuple

from nvo.data.loaders import load_rankings


def run_analysis(
    years: List[int],
    files_dir: str,
    gender_filter: Optional[str] = None,
    school_filter: Optional[List[str]] = None
) -> Tuple[Dict[int, Dict], Dict]:
    """Analyze historical data.
    
    Returns:
        yearly_stats: Dict of stats per year
        trends: Dict of trends per (school, profile)
    """
    target = 'Female' if gender_filter and gender_filter.lower().startswith('f') else \
             'Male' if gender_filter and gender_filter.lower().startswith('m') else 'Total'
    
    yearly_stats = {}
    all_data = {}
    
    for year in years:
        df = load_rankings(year, files_dir)
        if df is None:
            continue
        
        if school_filter:
            mask = df['School'].str.contains('|'.join(school_filter), case=False, na=False)
            df = df[mask]
        
        stats = {
            'records': len(df),
            'schools': df['School'].nunique(),
            'rounds': {}
        }
        
        for round_num in [1, 2, 3]:
            col = f'R{round_num}_Min_{target}'
            if col not in df.columns:
                continue
            valid = df[df[col] > 0]
            if len(valid) == 0:
                continue
            stats['rounds'][round_num] = {
                'valid': len(valid),
                'mean': valid[col].mean(),
                'median': valid[col].median(),
                'min': valid[col].min(),
                'max': valid[col].max()
            }
            
            for _, row in valid.iterrows():
                key = (row['School'], row['Profile'])
                if key not in all_data:
                    all_data[key] = {}
                all_data[key][(year, round_num)] = row[col]
        
        yearly_stats[year] = stats
    
    # Compute trends
    trends = {}
    for (school, profile), scores in all_data.items():
        r1_scores = sorted([(y, s) for (y, r), s in scores.items() if r == 1])
        r2_scores = sorted([(y, s) for (y, r), s in scores.items() if r == 2])
        r3_scores = sorted([(y, s) for (y, r), s in scores.items() if r == 3])
        
        if len(r1_scores) >= 2 or len(r2_scores) >= 2 or len(r3_scores) >= 2:
            trends[(school, profile)] = {'R1': r1_scores, 'R2': r2_scores, 'R3': r3_scores}
    
    return yearly_stats, trends


def format_trend(scores: List[Tuple[int, float]]) -> str:
    """Format trend scores with year-over-year changes."""
    if len(scores) < 2:
        return ""
    parts = [f"{scores[0][0]}:{scores[0][1]:.0f}"]
    for i in range(1, len(scores)):
        diff = scores[i][1] - scores[i-1][1]
        parts.append(f"{scores[i][0]}:{scores[i][1]:.0f}({diff:+.0f})")
    return " → ".join(parts)
