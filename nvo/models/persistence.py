"""Model persistence utilities for saving/loading trained models."""
import pickle
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from nvo.utils.logger import get_logger

logger = get_logger("models.persistence")

MODELS_DIR = Path("models")


def get_model_path(gender: str, round_num: int, version: str = "latest") -> Path:
    """Get path for a model file."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR / f"model_R{round_num}_{gender}_{version}.pkl"


def get_metadata_path() -> Path:
    """Get path for metadata file."""
    MODELS_DIR.mkdir(exist_ok=True)
    return MODELS_DIR / "metadata.json"


def compute_data_hash(years: list, gender: str) -> str:
    """Compute hash of training configuration for cache invalidation."""
    config_str = f"{sorted(years)}_{gender}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def save_model(
    model,
    le_school,
    le_profile,
    feature_cols: list,
    school_stats: dict,
    gender: str,
    round_num: int,
    training_years: list,
) -> Path:
    """Save trained model and associated artifacts."""
    MODELS_DIR.mkdir(exist_ok=True)
    
    data_hash = compute_data_hash(training_years, gender)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model bundle
    bundle = {
        'model': model,
        'le_school': le_school,
        'le_profile': le_profile,
        'feature_cols': feature_cols,
        'school_stats': school_stats,
        'gender': gender,
        'round_num': round_num,
        'training_years': training_years,
        'data_hash': data_hash,
        'created_at': timestamp,
    }
    
    # Save versioned and latest
    versioned_path = get_model_path(gender, round_num, timestamp)
    latest_path = get_model_path(gender, round_num, "latest")
    
    with open(versioned_path, 'wb') as f:
        pickle.dump(bundle, f)
    with open(latest_path, 'wb') as f:
        pickle.dump(bundle, f)
    
    # Update metadata
    _update_metadata(gender, round_num, timestamp, data_hash, training_years)
    
    logger.info(f"Saved model to {latest_path}")
    return latest_path


def load_model(
    gender: str,
    round_num: int,
    version: str = "latest"
) -> Optional[Dict[str, Any]]:
    """Load a trained model bundle."""
    path = get_model_path(gender, round_num, version)
    
    if not path.exists():
        logger.debug(f"No cached model found at {path}")
        return None
    
    try:
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        logger.info(f"Loaded cached model from {path}")
        return bundle
    except Exception as e:
        logger.warning(f"Failed to load model from {path}: {e}")
        return None


def is_model_valid(
    gender: str,
    round_num: int,
    training_years: list
) -> bool:
    """Check if cached model matches current training configuration."""
    bundle = load_model(gender, round_num)
    if bundle is None:
        return False
    
    expected_hash = compute_data_hash(training_years, gender)
    return bundle.get('data_hash') == expected_hash


def _update_metadata(gender: str, round_num: int, timestamp: str, data_hash: str, years: list):
    """Update metadata file with model info."""
    meta_path = get_metadata_path()
    
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'models': {}}
    
    key = f"R{round_num}_{gender}"
    metadata['models'][key] = {
        'version': timestamp,
        'data_hash': data_hash,
        'training_years': years,
        'updated_at': datetime.now().isoformat(),
    }
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def list_models() -> Dict[str, Any]:
    """List all available models."""
    meta_path = get_metadata_path()
    if not meta_path.exists():
        return {}
    
    with open(meta_path, 'r') as f:
        return json.load(f).get('models', {})


def clear_cache():
    """Remove all cached models."""
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.pkl"):
            f.unlink()
        meta_path = get_metadata_path()
        if meta_path.exists():
            meta_path.unlink()
        logger.info("Cleared model cache")
