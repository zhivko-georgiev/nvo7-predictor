"""Services package."""
from nvo.services.prediction import run_predictions
from nvo.services.validation import run_validation
from nvo.services.analysis import run_analysis, format_trend
from nvo.services.common import get_gender_list

__all__ = [
    'run_predictions',
    'run_validation', 
    'run_analysis',
    'format_trend',
    'get_gender_list'
]
