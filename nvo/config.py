"""Configuration management for NVO Rankings."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration loader and accessor."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'data.predict_year')."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    @property
    def data(self) -> Dict:
        return self._config.get('data', {})
    
    @property
    def model(self) -> Dict:
        return self._config.get('model', {})
    
    @property
    def filters(self) -> Dict:
        return self._config.get('filters', {})
    
    @property
    def scenarios(self) -> Dict:
        return self._config.get('scenarios', {})
    
    @property
    def output(self) -> Dict:
        return self._config.get('output', {})
    
    @property
    def logging(self) -> Dict:
        return self._config.get('logging', {})


# Global config instance
_config: Optional[Config] = None


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file."""
    global _config
    _config = Config(config_path)
    return _config


def get_config() -> Config:
    """Get current configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
