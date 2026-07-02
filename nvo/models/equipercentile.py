"""Equipercentile rank-mapping for exam difficulty adjustment.

The core idea: a school's prestige (rank in applicant preference) is more
stable than raw cutoff scores. If we convert last year's cutoff to a
percentile rank in last year's score distribution, and then map that rank
forward through this year's distribution, we get a difficulty-adjusted
prediction that handles exam shifts by construction — no extrapolation needed.

The cutoff score (бал) is: 2×BEL + 2×MAT + grade1 + grade2.
  - BEL, MAT: 0-100 each (NVO exam subjects)
  - grade1, grade2: certificate grades converted to points, max 50 each
  - Max cutoff: 2×100 + 2×100 + 50 + 50 = 500
  - Max grades component: 50 + 50 = 100
  - NVO component (BEL+MAT) range: 0-200

The exam distribution gives us the NVO component (BEL+MAT, 0-200 scale).
We map through the NVO distribution, then convert back to the full scale.

The grades component correlates strongly with cutoff level:
  - Elite schools (cutoff ~480): students have near-perfect grades (~98-100)
  - Mid-tier (cutoff ~350): students have good grades (~75-85)
  - Lower-tier (cutoff ~200): more variable (~40-60)
We use a score-dependent estimate rather than a flat constant.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from nvo.data.exam_loaders import load_exam_distribution
from nvo.utils.logger import get_logger

logger = get_logger("models.equipercentile")


def estimate_grades_component(cutoff: float) -> float:
    """Estimate the grades component of a cutoff score.
    
    The grades component consists of two certificate subjects converted to
    points (max 50 each = 100 total). It correlates with overall cutoff:
    high-cutoff students have near-perfect grades.
    
    Based on the structure:
      cutoff = 2*BEL + 2*MAT + grade1 + grade2
      max_grades = 100, min practical ~20
    
    We use a piecewise linear approximation calibrated to:
      - cutoff 500 → grades ~100 (perfect)
      - cutoff 400 → grades ~92 
      - cutoff 300 → grades ~75
      - cutoff 200 → grades ~55
      - cutoff 100 → grades ~35
    """
    # Clamp to valid range
    cutoff = max(0, min(500, cutoff))
    # Linear approximation: grades ≈ 0.16 * cutoff + 20, capped at 100
    grades = min(100.0, 0.16 * cutoff + 20.0)
    return grades


def load_full_distribution(year: int, files_dir: str, gender: str = 'Female') -> Optional[Dict]:
    """Load the full NVO score distribution for a given year and gender.
    
    Returns dict with:
        scores: array of score midpoints
        counts: array of student counts per bin
        cumulative_pct: cumulative percentile at each score point
        total_students: total number of students
    """
    from pathlib import Path
    from nvo.data.parsers import parse_bg_float
    
    filepath = Path(files_dir) / str(year) / f"average_grades_BEL_MAT-{year}.xlsx"
    
    try:
        df = pd.read_excel(filepath, header=None)
        header_row = df[df[0].astype(str).str.contains('Точки', na=False)].index[0]
        data_start = header_row + 3
        
        max_score = 200.999 if year == 2025 else 200.499
        
        # Column indices by gender
        gender_col_map = {
            'Total': 13,
            'Male': 15,
            'Female': 17,
        }
        col_idx = gender_col_map.get(gender, 17)
        
        scores = []
        counts = []
        
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
                score = (low + high) / 2
            elif s == '200':
                score = 200.0
            else:
                continue
            
            idx = data_start + i
            count = parse_bg_float(df.iloc[idx, col_idx])
            
            scores.append(score)
            counts.append(count)
        
        if not scores or sum(counts) == 0:
            return None
        
        scores = np.array(scores)
        counts = np.array(counts)
        total = counts.sum()
        
        # Cumulative from bottom (lower scores → higher percentile means better)
        # We want: percentile = fraction of students scoring AT OR BELOW this score
        cumsum = np.cumsum(counts)
        cumulative_pct = cumsum / total * 100.0
        
        return {
            'scores': scores,
            'counts': counts,
            'cumulative_pct': cumulative_pct,
            'total_students': total,
        }
    except Exception as e:
        logger.error(f"Failed to load distribution for {year}/{gender}: {e}")
        return None


def cutoff_to_nvo_score(cutoff: float) -> float:
    """Convert a full cutoff score (0-500) to approximate NVO score (0-200).
    
    Full score = 2*NVO + grades_component
    NVO ≈ (cutoff - grades_component) / 2
    """
    grades = estimate_grades_component(cutoff)
    nvo = (cutoff - grades) / 2.0
    return float(np.clip(nvo, 0, 200))


def nvo_score_to_cutoff(nvo: float, target_cutoff_hint: float = 0) -> float:
    """Convert NVO score (0-200) back to full cutoff scale (0-500).
    
    Since grades depend on cutoff level, we solve iteratively:
    cutoff = 2*nvo + grades(cutoff)
    
    Start with an initial estimate and converge.
    """
    # Initial estimate: assume average grades ~80
    cutoff = 2.0 * nvo + 80.0
    # Iterate to convergence (usually 2-3 iterations)
    for _ in range(5):
        grades = estimate_grades_component(cutoff)
        cutoff = 2.0 * nvo + grades
    return float(np.clip(cutoff, 0, 500))


def score_to_percentile(score: float, dist: Dict) -> float:
    """Convert a score to its percentile rank in the distribution.
    
    Uses linear interpolation between distribution bins.
    """
    scores = dist['scores']
    cum_pct = dist['cumulative_pct']
    
    if score <= scores[0]:
        return cum_pct[0] * (score / scores[0]) if scores[0] > 0 else 0
    if score >= scores[-1]:
        return 100.0
    
    # Linear interpolation
    idx = np.searchsorted(scores, score) - 1
    idx = max(0, min(idx, len(scores) - 2))
    
    # Interpolate between bins
    frac = (score - scores[idx]) / (scores[idx + 1] - scores[idx]) if scores[idx + 1] != scores[idx] else 0
    pct = cum_pct[idx] + frac * (cum_pct[idx + 1] - cum_pct[idx])
    
    return float(np.clip(pct, 0, 100))


def percentile_to_score(percentile: float, dist: Dict) -> float:
    """Convert a percentile rank back to a score in the distribution.
    
    Inverse of score_to_percentile. Uses linear interpolation.
    """
    cum_pct = dist['cumulative_pct']
    scores = dist['scores']
    
    if percentile <= cum_pct[0]:
        return scores[0] * (percentile / cum_pct[0]) if cum_pct[0] > 0 else scores[0]
    if percentile >= cum_pct[-1]:
        return scores[-1]
    
    # Linear interpolation (inverse)
    idx = np.searchsorted(cum_pct, percentile) - 1
    idx = max(0, min(idx, len(cum_pct) - 2))
    
    frac = (percentile - cum_pct[idx]) / (cum_pct[idx + 1] - cum_pct[idx]) \
        if cum_pct[idx + 1] != cum_pct[idx] else 0
    score = scores[idx] + frac * (scores[idx + 1] - scores[idx])
    
    return float(score)


def equipercentile_predict(
    prev_cutoff: float,
    prev_year_dist: Dict,
    curr_year_dist: Dict,
) -> float:
    """Predict current year cutoff using equipercentile mapping.
    
    Steps:
    1. Convert prev_cutoff → NVO score
    2. Find percentile rank of that NVO score in prev_year distribution
    3. Map that percentile to a score in curr_year distribution
    4. Convert back to full cutoff scale
    
    This handles exam difficulty shift by construction.
    """
    # Step 1: cutoff → NVO score
    nvo_prev = cutoff_to_nvo_score(prev_cutoff)
    
    # Step 2: NVO score → percentile in last year's distribution
    pct = score_to_percentile(nvo_prev, prev_year_dist)
    
    # Step 3: percentile → NVO score in current year's distribution
    nvo_curr = percentile_to_score(pct, curr_year_dist)
    
    # Step 4: NVO score → cutoff
    predicted_cutoff = nvo_score_to_cutoff(nvo_curr)
    
    return float(np.clip(predicted_cutoff, 0, 500))


def compute_equipercentile_predictions(
    df_prev_year: pd.DataFrame,
    target_col: str,
    prev_year: int,
    curr_year: int,
    gender: str,
    files_dir: str,
) -> Dict[str, float]:
    """Compute equipercentile predictions for all school-profiles.
    
    Returns a dict mapping "School|Profile" → predicted cutoff.
    """
    prev_dist = load_full_distribution(prev_year, files_dir, gender)
    curr_dist = load_full_distribution(curr_year, files_dir, gender)
    
    if prev_dist is None or curr_dist is None:
        logger.warning(f"Cannot compute equipercentile: missing distribution for {prev_year} or {curr_year}")
        return {}
    
    predictions = {}
    
    for _, row in df_prev_year.iterrows():
        key = f"{row['School']}|{row['Profile']}"
        prev_cutoff = row.get(target_col, 0)
        
        if prev_cutoff <= 0:
            continue
        
        pred = equipercentile_predict(prev_cutoff, prev_dist, curr_dist)
        predictions[key] = pred
    
    n_preds = len(predictions)
    if n_preds > 0:
        logger.info(
            f"Equipercentile predictions ({gender}): {n_preds} profiles, "
            f"avg shift={np.mean(list(predictions.values())) - df_prev_year[target_col].mean():.1f} points"
        )
    
    return predictions
