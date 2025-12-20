"""Data processing with exam distribution features."""
import pandas as pd
from typing import List
from nvo.data.loaders import load_rankings, load_school_capacity
from nvo.data.exam_loaders import load_exam_distribution
from nvo.utils.logger import get_logger

logger = get_logger("data.processors")


def build_dataset(historical_years: List[int], files_dir: str = "files") -> pd.DataFrame:
    """Build dataset with exam distribution features."""
    all_data = []
    
    for year in historical_years:
        logger.info(f"Processing data for {year}...")
        
        rankings = load_rankings(year, files_dir)
        if rankings.empty:
            logger.warning(f"No rankings data for {year}, skipping")
            continue
        
        # Load exam distribution features (percentiles, mean, std)
        exam_features = load_exam_distribution(year, files_dir)
        
        # Load capacity
        capacity = load_school_capacity(year, files_dir)
        
        # Add year
        rankings['Year'] = year
        
        # Add exam distribution features
        for key, val in exam_features.items():
            rankings[key] = val
        
        # Merge capacity
        if not capacity.empty:
            rankings = pd.merge(rankings, capacity, on=['School', 'Profile'], how='left')
        
        all_data.append(rankings)
    
    if not all_data:
        logger.error("No data loaded from any year")
        return pd.DataFrame()
    
    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"Built dataset: {len(result)} records, {len(result.columns)} features")
    return result
