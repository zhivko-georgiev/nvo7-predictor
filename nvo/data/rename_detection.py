"""Profile name rename detection across years.

Profile names sometimes get renamed between years (e.g., slight text
changes, reordering, or official renaming). This module detects probable
renames so that continuing profiles aren't silently treated as "new".

Strategy:
- For each school, compare profiles in year Y vs year Y-1
- Profiles that exist in Y but not Y-1 (and vice versa) are candidates
- Score candidates by string similarity (difflib SequenceMatcher)
- Filter: require language tokens to match (swapping the second foreign
  language is a different program, not a rename)
- Accept matches above a threshold as renames
- Apply renames sequentially in chronological order so that chains
  (A→B in year 1→2, B→C in year 2→3) and oscillations resolve naturally
"""
import re
import difflib
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
from nvo.utils.logger import get_logger

logger = get_logger("data.rename_detection")

# Minimum similarity ratio to consider a rename (0-1 scale)
RENAME_THRESHOLD = 0.75

# Known language tokens in profile names
# These identify the foreign language taught in the program
LANGUAGE_TOKENS = {
    'АЕ', 'НЕ', 'ФЕ', 'РЕ', 'ИЕ', 'ИтЕ', 'ИспЕ', 'ГрЕ', 'КитЕ',
    'КорЕ', 'ЯЕ', 'АрЕ', 'ТурЕ', 'РумЕ', 'ШвЕ', 'ДатЕ', 'НорвЕ',
    'Иврит', 'Японски', 'Испански', 'Китайски', 'Немски',
}

# Pattern to extract language tokens from profile names
_LANG_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in sorted(LANGUAGE_TOKENS, key=len, reverse=True)) + r')\b'
)


def _extract_language_tokens(profile: str) -> Set[str]:
    """Extract language tokens from a profile name.
    
    Returns the set of language identifiers found after the ' - ' separator.
    """
    parts = profile.split(' - ', 1)
    if len(parts) < 2:
        return set()
    
    lang_part = parts[1]
    return set(_LANG_PATTERN.findall(lang_part))


def _languages_compatible(old_profile: str, new_profile: str) -> bool:
    """Check if two profiles have compatible language tokens.
    
    A rename should preserve the language configuration. If both profiles
    specify languages, they must match exactly. Changing the second foreign
    language (e.g., НЕ→ФЕ, АЕ→ЯЕ) is a different program, not a rename.
    
    Exception: adding a second language (e.g., "АЕ интензивно" → "АЕ интензивно, НЕ")
    is allowed since it's a format clarification, not a program change.
    """
    old_langs = _extract_language_tokens(old_profile)
    new_langs = _extract_language_tokens(new_profile)
    
    # If either has no language tokens, allow (non-language profile or no lang part)
    if not old_langs or not new_langs:
        return True
    
    # Exact match — definitely compatible
    if old_langs == new_langs:
        return True
    
    # One is a subset of the other (language was added/removed from format)
    if old_langs < new_langs or new_langs < old_langs:
        return True
    
    # Any other case: languages changed → different program
    return False


def _detect_year_pair_renames(
    df: pd.DataFrame,
    year_from: int,
    year_to: int,
) -> Dict[Tuple[str, str], str]:
    """Detect probable profile renames between two consecutive years.
    
    Returns:
        Dict mapping (school, old_profile) → new_profile
    """
    df_from = df[df['Year'] == year_from][['School', 'Profile']].drop_duplicates()
    df_to = df[df['Year'] == year_to][['School', 'Profile']].drop_duplicates()
    
    renames = {}  # (school, old_name) → new_name
    
    schools = set(df_from['School'].unique()) | set(df_to['School'].unique())
    
    for school in schools:
        profiles_from = set(df_from[df_from['School'] == school]['Profile'].values)
        profiles_to = set(df_to[df_to['School'] == school]['Profile'].values)
        
        continuing = profiles_from & profiles_to
        disappeared = profiles_from - continuing
        appeared = profiles_to - continuing
        
        if not disappeared or not appeared:
            continue
        
        used_from = set()
        
        for new_profile in sorted(appeared):
            best_match = None
            best_score = 0
            
            for old_profile in sorted(disappeared):
                if old_profile in used_from:
                    continue
                
                if not _languages_compatible(old_profile, new_profile):
                    continue
                
                score = difflib.SequenceMatcher(None, old_profile, new_profile).ratio()
                
                if score > best_score and score >= RENAME_THRESHOLD:
                    best_score = score
                    best_match = old_profile
            
            if best_match:
                renames[(school, best_match)] = new_profile
                used_from.add(best_match)
                logger.info(
                    f"Detected rename ({year_from}→{year_to}): "
                    f"[{school}] '{best_match}' → '{new_profile}' "
                    f"(similarity: {best_score:.2f})"
                )
    
    return renames


def detect_and_apply_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and apply renames sequentially in chronological order.
    
    For each consecutive year pair, detect renames on the current state of
    the dataframe, then apply them (renaming old→new in ALL rows). This
    handles transitive chains (A→B→C) and oscillations naturally:
    - After applying A→B, all rows now say B
    - When B→C is detected and applied, all rows (including former A) become C
    
    Returns the dataframe with all profile names normalized to their latest version.
    """
    years = sorted(df['Year'].unique())
    if len(years) < 2:
        return df
    
    df = df.copy()
    total_renamed = 0
    
    for i in range(len(years) - 1):
        year_from, year_to = years[i], years[i + 1]
        
        # Detect renames on the CURRENT state of the dataframe
        renames = _detect_year_pair_renames(df, year_from, year_to)
        
        if not renames:
            continue
        
        # Apply this year-pair's renames to ALL rows in the dataframe
        # This is the key insight: renaming everywhere means chains resolve naturally
        renamed_count = 0
        for idx, row in df.iterrows():
            key = (row['School'], row['Profile'])
            if key in renames:
                df.at[idx, 'Profile'] = renames[key]
                renamed_count += 1
        
        total_renamed += renamed_count
    
    if total_renamed > 0:
        logger.info(f"Applied {total_renamed} profile name normalizations across dataset")
    
    return df
