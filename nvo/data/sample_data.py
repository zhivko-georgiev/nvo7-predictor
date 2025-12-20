"""Sample data generator for demo mode."""
import pandas as pd
import numpy as np
from pathlib import Path

# Sample schools (anonymized/fictional)
SAMPLE_SCHOOLS = [
    ("Elite Math School", "Mathematics - English"),
    ("Elite Math School", "Mathematics - German"),
    ("Science Academy", "Physics and Mathematics"),
    ("Science Academy", "Chemistry and Biology"),
    ("Language High School", "English Philology"),
    ("Language High School", "German Philology"),
    ("Technical School", "Software Engineering"),
    ("Technical School", "Computer Science"),
    ("Arts Academy", "Fine Arts"),
    ("Arts Academy", "Music"),
    ("Sports School", "Athletics"),
    ("General High School", "General Education"),
]

def generate_sample_rankings(year: int, base_seed: int = 42) -> pd.DataFrame:
    """Generate sample ranking data for a year."""
    np.random.seed(base_seed + year)
    
    rows = []
    for school, profile in SAMPLE_SCHOOLS:
        # Base scores vary by school prestige
        if "Elite" in school or "Science" in school:
            base = np.random.uniform(450, 490)
        elif "Language" in school or "Technical" in school:
            base = np.random.uniform(380, 440)
        else:
            base = np.random.uniform(300, 380)
        
        # Add year-over-year variation
        year_effect = (year - 2022) * np.random.uniform(-5, 10)
        
        for round_num in range(1, 5):
            round_drop = round_num * np.random.uniform(2, 8)
            
            r_total = max(0, base + year_effect - round_drop + np.random.normal(0, 5))
            r_male = max(0, r_total - np.random.uniform(5, 15))
            r_female = max(0, r_total + np.random.uniform(0, 10))
            
            rows.append({
                'School': school,
                'Profile': profile,
                f'R{round_num}_Min_Total': round(r_total, 2),
                f'R{round_num}_Min_Male': round(r_male, 2),
                f'R{round_num}_Min_Female': round(r_female, 2),
            })
    
    # Aggregate by school/profile
    df = pd.DataFrame(rows)
    df_agg = df.groupby(['School', 'Profile']).first().reset_index()
    
    return df_agg


def generate_sample_exam_distribution(year: int) -> dict:
    """Generate sample exam distribution features."""
    np.random.seed(42 + year)
    
    base_mean = 140 + (year - 2022) * 2  # Slight increase over years
    
    features = {}
    for gender in ['Total', 'Male', 'Female']:
        gender_offset = 0 if gender == 'Total' else (5 if gender == 'Female' else -5)
        mean = base_mean + gender_offset + np.random.normal(0, 3)
        std = 25 + np.random.normal(0, 2)
        
        features[f'Exam_{gender}_Mean'] = mean
        features[f'Exam_{gender}_Std'] = std
        features[f'Exam_{gender}_P10'] = mean - 1.28 * std
        features[f'Exam_{gender}_P25'] = mean - 0.67 * std
        features[f'Exam_{gender}_P50'] = mean
        features[f'Exam_{gender}_P75'] = mean + 0.67 * std
        features[f'Exam_{gender}_P90'] = mean + 1.28 * std
        features[f'Exam_{gender}_P95'] = mean + 1.65 * std
    
    return features


def create_sample_data(output_dir: str = "sample_data"):
    """Create sample data files for demo mode."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for year in [2022, 2023, 2024]:
        year_dir = output_path / str(year)
        year_dir.mkdir(exist_ok=True)
        
        # Generate rankings
        df = generate_sample_rankings(year)
        
        # Save as Excel (mimicking real format)
        # For simplicity, save as CSV that can be loaded
        df.to_csv(year_dir / f"rankings_{year}.csv", index=False)
        
        # Generate exam features
        exam = generate_sample_exam_distribution(year)
        pd.DataFrame([exam]).to_csv(year_dir / f"exam_{year}.csv", index=False)
    
    print(f"Sample data created in {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_data()
