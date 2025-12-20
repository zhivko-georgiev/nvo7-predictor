"""Output formatting utilities."""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from nvo.services.common import get_gender_list
from nvo.services.analysis import format_trend


def save_results(results: Dict, output_dir: str, filename: str) -> pd.DataFrame:
    """Save results to Excel and return DataFrame."""
    results_df = pd.DataFrame(list(results.values()))
    Path(output_dir).mkdir(exist_ok=True)
    output_file = Path(output_dir) / filename
    results_df.to_excel(output_file, index=False)
    return results_df, output_file


def format_prediction_metrics(metrics: Dict) -> List[str]:
    """Format prediction metrics for display."""
    lines = []
    for key, m in metrics.items():
        lines.append(f"{key}: Predictions={m['total']}, Reliable={m['reliable']}")
    return lines


def format_validation_metrics(metrics: Dict) -> List[str]:
    """Format validation metrics for display."""
    lines = []
    for key, m in metrics.items():
        lines.append(f"{key}:")
        lines.append(f"  MAE: {m['mae_existing']:.2f} points (existing profiles)")
        lines.append(f"  MAE: {m['mae_reliable']:.2f} points (reliable only)")
        lines.append(f"  New profiles: {m['new_profiles']}/{m['total']}")
        lines.append(f"  Reliable: {m['reliable']}/{m['total']}")
    return lines


def format_top_predictions(df: pd.DataFrame, gender_filter: Optional[str], n: int = 10) -> str:
    """Format top N predictions for display."""
    genders = get_gender_list(gender_filter)
    target = genders[0] if len(genders) == 1 else 'Female'
    
    sort_col = [c for c in df.columns if c.endswith('_Predicted')]
    if sort_col:
        df = df.sort_values(sort_col[0], ascending=False)
    
    display_cols = ['School', 'Profile', f'R1_{target}_Predicted', 
                    f'R1_{target}_Lower', f'R1_{target}_Upper', f'R1_{target}_Confidence']
    display_cols = [c for c in display_cols if c in df.columns]
    
    return df[display_cols].head(n).to_string(index=False)


def format_worst_predictions(df: pd.DataFrame, gender_filter: Optional[str], n: int = 10) -> str:
    """Format worst N predictions for display."""
    genders = get_gender_list(gender_filter)
    target = genders[0] if len(genders) == 1 else 'Female'
    
    abs_err_col = f'R1_{target}_Abs_Error'
    if abs_err_col in df.columns:
        df = df.sort_values(abs_err_col, ascending=False)
    
    display_cols = ['School', 'Profile', f'R1_{target}_Actual', 
                    f'R1_{target}_Predicted', f'R1_{target}_Error']
    display_cols = [c for c in display_cols if c in df.columns]
    
    return df[display_cols].head(n).to_string(index=False)


def format_yearly_stats(yearly_stats: Dict) -> List[str]:
    """Format yearly statistics for display."""
    lines = []
    for year, stats in yearly_stats.items():
        lines.append(f"\n=== Year {year} ===")
        lines.append(f"Records: {stats['records']}, Schools: {stats['schools']}")
        for rnd, rnd_stats in stats['rounds'].items():
            lines.append(f"  R{rnd}: Min={rnd_stats['min']:.1f}, Max={rnd_stats['max']:.1f}, Mean={rnd_stats['mean']:.1f}")
    return lines


def format_trends(trends: Dict, target: str) -> List[str]:
    """Format trends for display."""
    lines = [f"\n=== Trends - Cutoff Scores ({target}) ==="]
    current_school = None
    
    for (school, profile), data in sorted(trends.items()):
        has_data = len(data['R1']) >= 2 or len(data['R2']) >= 2
        if not has_data:
            continue
        if school != current_school:
            lines.append(f"\n  {school}")
            current_school = school
        lines.append(f"    {profile[:50]}")
        if len(data['R1']) >= 2:
            lines.append(f"      R1: {format_trend(data['R1'])}")
        if len(data['R2']) >= 2:
            lines.append(f"      R2: {format_trend(data['R2'])}")
        # Calculate R1->R2 drop
        if len(data['R1']) >= 1 and len(data['R2']) >= 1:
            r1_by_year = {y: s for y, s in data['R1']}
            drops = []
            for y, r2_score in data['R2']:
                if y in r1_by_year:
                    drop = r2_score - r1_by_year[y]
                    drops.append(f"{y}:{drop:+.0f}")
            if drops:
                lines.append(f"      R1→R2: {', '.join(drops)}")
    
    return lines
