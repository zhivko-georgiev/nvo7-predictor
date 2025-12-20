# NVO 7th Grade Rankings Prediction System

ML-based prediction system for Bulgarian 7th grade high school admission cutoff scores (NVO - Национално външно оценяване).

## Overview

This system predicts admission cutoff scores for Sofia high schools using machine learning:

- 🎯 XGBoost model predicting year-over-year score changes
- 📊 Historical trend and acceleration features
- 🔍 Prediction intervals with confidence scores
- ✅ Reliability indicators for each prediction
- 📈 Support for Round 1 and Round 2 predictions

## Performance

Validated on 2025 data (trained on 2022-2024):

**Round 1:**
| Metric | All Profiles | Reliable Only |
|--------|--------------|---------------|
| MAE | 19.15 points | **13.42 points** |
| Within 10 pts | 45.6% | 54.8% |
| Within 20 pts | 65.8% | 77.4% |

**Round 2:**
| Metric | All Profiles | Reliable Only |
|--------|--------------|---------------|
| MAE | 24.31 points | **18.42 points** |

**Reliable predictions** = profiles with ≥2 years history, volatility <25, and previous year data available.

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/zhivko-georgiev/nvo-7mi-klas-rankings.git
cd nvo-7mi-klas-rankings
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

### Basic Usage

```bash
# Predict for 2026 (female students)
nvo predict --year 2026 --gender female

# Validate model against 2025 actual results
nvo validate --test-year 2025 --gender female

# Predict for specific schools
nvo predict --year 2026 --gender female --schools "119,35,18"

# Analyze historical data for specific schools
nvo analyze --years 2023,2024,2025 --gender female --schools "СМГ,НПМГ"
```

### Output Example

```
School: Софийска математическа гимназия "Паисий Хилендарски"
Profile: Математически (математика и информатика) - АЕ (РИЧЕ)
R1_Predicted: 487.11
R1_Range: 485.61 - 488.61
R1_Confidence: 98.3%
R1_Reliable: True
```

## Features

### Reliability Scoring

Each prediction includes reliability indicators:

| Column | Description |
|--------|-------------|
| `Years_Data` | Number of historical years available |
| `Volatility` | Historical year-over-year variation |
| `Reliable` | True if prediction is trustworthy |

**Reliability criteria**: ≥2 years of data, volatility <25 points, previous year data available.

### Prediction Intervals

- Lower/Upper bounds based on historical volatility
- ~87% confidence coverage
- Wider intervals for volatile schools

### Model Features

The XGBoost model uses:
- Previous year score
- School historical mean
- Year-over-year trend (average)
- Trend acceleration
- Distance from historical mean
- Exam score distributions (24 percentile features)

## Project Structure

```
nvo-7mi-klas-rankings/
├── nvo/                          # Main package
│   ├── cli/commands.py          # CLI commands (predict, validate)
│   ├── data/                    # Data loading & processing
│   │   ├── loaders.py          # Excel file loaders
│   │   ├── exam_loaders.py     # Exam distribution features
│   │   └── processors.py       # Data processing
│   ├── models/                  # ML models
│   │   ├── trainer.py          # Model training
│   │   ├── predictor.py        # Prediction generation
│   │   └── prediction_utils.py # Shared prediction utilities
│   └── utils/logger.py         # Logging
├── config/default.yaml          # Configuration
├── files/                       # Data directory (not in repo)
│   └── YYYY/                   # Year-specific data
├── output/                      # Generated predictions
└── setup.py                    # Installation
```

## Data Requirements

Place Excel files in `files/YEAR/`:

```
files/2025/
├── klasirane_1_2025.xlsx       # Round 1 rankings
├── klasirane_2_2025.xlsx       # Round 2 rankings
├── klasirane_3_2025.xlsx       # Round 3 rankings
├── klasirane_4_2025.xlsx       # Round 4 rankings
├── schools_2025.xlsx           # School metadata
└── average_grades_BEL_MAT-2025.xlsx  # Exam score distribution
```

Data source: Bulgarian Ministry of Education (РУО София-град)

## Configuration

Edit `config/default.yaml`:

```yaml
data:
  historical_years: [2022, 2023, 2024, 2025]
  predict_year: 2026

model:
  n_estimators: 50
  max_depth: 3
  learning_rate: 0.1

filters:
  gender: "female"  # male, female, or null for all
```

## Understanding Results

### Confidence Scores

| Score | Meaning |
|-------|---------|
| 80-100 | Very reliable - trust the prediction |
| 60-80 | Reliable - use with some caution |
| 40-60 | Moderate - consider wider range |
| 0-40 | Low - school is very volatile |

### When Predictions Are Less Reliable

- **New profiles**: No historical data (marked `New_Profile: True`)
- **High volatility**: Schools with >25 point swings between years
- **Missing previous year**: Gap in data continuity

## Limitations

1. **New schools/profiles**: Cannot predict without historical data
2. **Extreme volatility**: Some schools change 100+ points between years
3. **External factors**: Policy changes, new programs not captured
4. **Sofia only**: Currently trained on Sofia (РУО София-град) data

## License

MIT License - see [LICENSE](LICENSE)

## Author

Zhivko Georgiev

## Acknowledgments

- Historical data from Bulgarian Ministry of Education
- Built with XGBoost, pandas, scikit-learn
