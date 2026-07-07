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


def load_subject_distributions(year: int, files_dir: str, gender: str = 'Female') -> Optional[Dict]:
    """Load separate BEL and MAT score distributions for a given year and gender.
    
    Returns dict with keys 'bel' and 'mat', each containing:
        scores: array of score midpoints (0-100 scale per subject)
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
        
        # Per-subject column indices:
        # BEL: Total=1, Male=3, Female=5
        # MAT: Total=7, Male=9, Female=11
        gender_bel_col = {'Total': 1, 'Male': 3, 'Female': 5}
        gender_mat_col = {'Total': 7, 'Male': 9, 'Female': 11}
        
        bel_col = gender_bel_col.get(gender, 5)
        mat_col = gender_mat_col.get(gender, 11)
        
        # Per-subject scores are 0-100 scale
        max_score = 100.999
        
        scores = []
        bel_counts = []
        mat_counts = []
        
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
            elif s == '100':
                score = 100.0
            elif s == '200':
                # We've gone past per-subject range into combined
                break
            else:
                continue
            
            idx = data_start + i
            bel_count = parse_bg_float(df.iloc[idx, bel_col])
            mat_count = parse_bg_float(df.iloc[idx, mat_col])
            
            scores.append(score)
            bel_counts.append(bel_count)
            mat_counts.append(mat_count)
        
        if not scores:
            return None
        
        scores = np.array(scores)
        bel_counts = np.array(bel_counts)
        mat_counts = np.array(mat_counts)
        
        def _make_dist(counts):
            total = counts.sum()
            if total == 0:
                return None
            cumsum = np.cumsum(counts)
            return {
                'scores': scores.copy(),
                'counts': counts,
                'cumulative_pct': cumsum / total * 100.0,
                'total_students': total,
            }
        
        bel_dist = _make_dist(bel_counts)
        mat_dist = _make_dist(mat_counts)
        
        if bel_dist is None or mat_dist is None:
            return None
        
        return {'bel': bel_dist, 'mat': mat_dist}
    
    except Exception as e:
        logger.error(f"Failed to load subject distributions for {year}/{gender}: {e}")
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


def equipercentile_predict_weighted(
    prev_cutoff: float,
    prev_subj_dists: Dict,
    curr_subj_dists: Dict,
    w_bel: int = 2,
    w_mat: int = 2,
) -> float:
    """Predict current year cutoff using per-subject equipercentile mapping.
    
    Instead of mapping through the combined distribution (which assumes 2+2),
    this computes the per-subject percentile shift and applies it with the
    school's actual weighting formula.
    
    Formula: cutoff = w_bel * BEL + w_mat * MAT + grades
    
    Steps:
    1. Estimate per-subject scores from prev_cutoff
    2. Find each subject's percentile rank in prev_year distribution
    3. Map each percentile to a score in curr_year distribution
    4. Reconstruct cutoff with the school's weights
    """
    grades = estimate_grades_component(prev_cutoff)
    
    # Estimate per-subject scores: cutoff = w_bel*BEL + w_mat*MAT + grades
    # Assume BEL and MAT are roughly equal for initial split (will be corrected by mapping)
    nvo_total = prev_cutoff - grades  # = w_bel*BEL + w_mat*MAT
    # Approximate: BEL ≈ MAT ≈ nvo_total / (w_bel + w_mat)
    avg_subject = nvo_total / (w_bel + w_mat) if (w_bel + w_mat) > 0 else 50
    bel_prev = float(np.clip(avg_subject, 0, 100))
    mat_prev = float(np.clip(avg_subject, 0, 100))
    
    prev_bel_dist = prev_subj_dists['bel']
    prev_mat_dist = prev_subj_dists['mat']
    curr_bel_dist = curr_subj_dists['bel']
    curr_mat_dist = curr_subj_dists['mat']
    
    # Map each subject through its own distribution
    bel_pct = score_to_percentile(bel_prev, prev_bel_dist)
    mat_pct = score_to_percentile(mat_prev, prev_mat_dist)
    
    bel_curr = percentile_to_score(bel_pct, curr_bel_dist)
    mat_curr = percentile_to_score(mat_pct, curr_mat_dist)
    
    # Reconstruct cutoff with school's weights
    predicted_cutoff = w_bel * bel_curr + w_mat * mat_curr + grades
    
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
    
    Uses per-subject (BEL/MAT) distributions when the school's weighting
    formula is available in the data (via BEL_Weight/MAT_Weight columns).
    Falls back to combined distribution for profiles without weight info
    or where per-subject data isn't available.
    
    Returns a dict mapping "School|Profile" → predicted cutoff.
    """
    prev_dist = load_full_distribution(prev_year, files_dir, gender)
    curr_dist = load_full_distribution(curr_year, files_dir, gender)
    
    if prev_dist is None or curr_dist is None:
        logger.warning(f"Cannot compute equipercentile: missing distribution for {prev_year} or {curr_year}")
        return {}
    
    # Try to load per-subject distributions
    prev_subj = load_subject_distributions(prev_year, files_dir, gender)
    curr_subj = load_subject_distributions(curr_year, files_dir, gender)
    has_subject_dists = prev_subj is not None and curr_subj is not None
    
    predictions = {}
    weighted_count = 0
    
    for _, row in df_prev_year.iterrows():
        key = f"{row['School']}|{row['Profile']}"
        prev_cutoff = row.get(target_col, 0)
        
        if prev_cutoff <= 0:
            continue
        
        # Check if per-subject weighted prediction is possible
        w_bel = row.get('BEL_Weight', 0)
        w_mat = row.get('MAT_Weight', 0)
        
        # Always use combined distribution as base (most robust)
        pred = equipercentile_predict(prev_cutoff, prev_dist, curr_dist)
        
        # Per-subject correction for non-standard weights (3+1 or 1+3).
        # Only activates when subjects diverge strongly between years (>10 pts).
        # In mild-divergence years (like 2025→2026, ~±3 pts), the correction
        # adds noise due to the crude BEL/MAT split estimation.
        # The threshold should be lowered once we have better per-subject
        # score estimation (e.g., from per-student data or score-band calibration).
        if has_subject_dists and w_bel > 0 and w_mat > 0 and (w_bel != 2 or w_mat != 2):
            pred_weighted = equipercentile_predict_weighted(
                prev_cutoff, prev_subj, curr_subj,
                w_bel=int(w_bel), w_mat=int(w_mat)
            )
            pred_default_weight = equipercentile_predict_weighted(
                prev_cutoff, prev_subj, curr_subj,
                w_bel=2, w_mat=2
            )
            divergence_correction = pred_weighted - pred_default_weight
            # Only apply if subjects diverged strongly (>10 points)
            if abs(divergence_correction) > 10.0:
                pred += divergence_correction
                pred = float(np.clip(pred, 0, 500))
                weighted_count += 1
        
        predictions[key] = pred
    
    n_preds = len(predictions)
    if n_preds > 0:
        logger.info(
            f"Equipercentile predictions ({gender}): {n_preds} profiles "
            f"({weighted_count} per-subject weighted, {n_preds - weighted_count} combined)"
        )
    
    return predictions
