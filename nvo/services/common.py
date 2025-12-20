"""Common utilities for services."""
from typing import List, Optional


def get_gender_list(gender_filter: Optional[str]) -> List[str]:
    """Get list of genders to process based on filter."""
    if gender_filter and gender_filter.lower().startswith('f'):
        return ['Female']
    elif gender_filter and gender_filter.lower().startswith('m'):
        return ['Male']
    return ['Total', 'Male', 'Female']
