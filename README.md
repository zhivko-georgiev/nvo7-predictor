# NVO 7th Grade Rankings Prediction System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

ML-based prediction system for Bulgarian 7th grade high school admission cutoff scores (NVO - Национално външно оценяване).

## 🎯 What This Does

Every year, thousands of Bulgarian students compete for spots in elite high schools based on their NVO exam scores. This system predicts the minimum admission scores (cutoffs) for each school, helping families make informed decisions about their applications.

**Key Features:**
- 🔮 Predicts Round 1 and Round 2 cutoff scores
- 📊 Provides confidence intervals for each prediction
- ✅ Marks reliable vs uncertain predictions
- 📈 Analyzes historical trends
- 🌐 Web interface (Streamlit) and CLI

## 📊 Performance

Validated on 2025 data (trained on 2022-2024):

| Round | All Profiles | Reliable Only |
|-------|--------------|---------------|
| R1 MAE | 19.15 points | **13.42 points** |
| R1 Within 10 pts | 45.6% | 54.8% |
| R1 Within 20 pts | 65.8% | 77.4% |
| R2 MAE | 24.31 points | **18.42 points** |

**Reliable predictions** = profiles with ≥2 years history, volatility <25, and previous year data.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/zhivko-georgiev/nvo7-predictor.git
cd nvo7-predictor
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### CLI Usage

```bash
# Predict for 2026 (female students)
nvo predict --year 2026 --gender female

# Validate model against 2025 actual results
nvo validate --test-year 2025 --gender female

# Predict for specific schools
nvo predict --year 2026 --gender female --schools "СМГ,НПМГ"

# Force retrain (ignore cached models)
nvo predict --year 2026 --gender female --no-cache
```

### Web Interface

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 📁 Data Sources

Data comes from the Bulgarian Ministry of Education (РУО София-град):

1. **Ranking files** (`klasirane_1-4_YEAR.xlsx`): Admission results for each round
2. **Exam distributions** (`average_grades_BEL_MAT-YEAR.xlsx`): Score distributions by gender
3. **School info** (`schools_YEAR.xlsx`): School capacity data

Place files in `files/YEAR/` directory:
```
files/
├── 2022/
│   ├── klasirane_1_2022.xlsx
│   ├── klasirane_2_2022.xlsx
│   └── average_grades_BEL_MAT-2022.xlsx
├── 2023/
│   └── ...
└── 2024/
    └── ...
```

**Official data source:** [РУО София-град](https://ruo-sofia-grad.com/)

## 🧠 How It Works

### The Problem

Predicting admission cutoffs is hard because:
- Scores depend on exam difficulty (varies yearly)
- Student preferences shift between schools
- New programs open, others close
- Competition levels change

### Our Approach

1. **Delta Prediction**: Instead of predicting absolute scores, we predict year-over-year *change*. The previous year's score is the strongest predictor.

2. **Feature Engineering**:
   - Historical mean and volatility per school
   - Trend direction and acceleration
   - Exam score distribution (percentiles)
   - Distance from historical mean

3. **Gender-Specific Blending**: Female predictions benefit from the model (60% model, 40% naive), while male predictions use pure naive baseline (previous year's score).

4. **Confidence Intervals**: Based on historical volatility - stable schools get tighter intervals.

### Model Details

- **Algorithm**: XGBoost with shallow trees (max_depth=3)
- **Regularization**: Strong L1/L2 to prevent overfitting
- **Training data**: ~800 samples (3 years × ~270 schools)
- **Training time**: <2 seconds

## 📖 Understanding Results

### Output Columns

| Column | Description |
|--------|-------------|
| `R1_Female_Predicted` | Predicted Round 1 cutoff score |
| `R1_Female_Lower/Upper` | Confidence interval bounds |
| `R1_Female_Confidence` | Confidence score (0-100) |
| `R1_Female_Volatility` | Historical year-over-year variation |
| `R1_Female_Reliable` | True if prediction is trustworthy |

### Confidence Scores

| Score | Meaning |
|-------|---------|
| 80-100 | Very reliable - trust the prediction |
| 60-80 | Reliable - use with some caution |
| 40-60 | Moderate - consider wider range |
| 0-40 | Low - school is very volatile |

### When to Be Cautious

- **New profiles**: No historical data (marked `Reliable: False`)
- **High volatility**: Schools with >25 point swings between years
- **Missing previous year**: Gap in data continuity

## 🔧 Configuration

Edit `config/default.yaml`:

```yaml
data:
  historical_years: [2022, 2023, 2024, 2025]
  predict_year: 2026
  files_dir: "files"
  output_dir: "output"

model:
  n_estimators: 50
  max_depth: 3
  learning_rate: 0.1

filters:
  gender: "female"
```

## 📂 Project Structure

```
nvo-7mi-klas-rankings/
├── nvo/                    # Main package
│   ├── cli/               # CLI commands
│   ├── data/              # Data loading & processing
│   ├── models/            # ML training & prediction
│   ├── services/          # Business logic
│   └── utils/             # Logging, config
├── app.py                 # Streamlit web interface
├── config/                # Configuration files
├── files/                 # Data directory (not in repo)
├── models/                # Cached trained models
└── output/                # Generated predictions
```

## ⚠️ Limitations

1. **Sofia only**: Currently trained on Sofia (РУО София-град) data
2. **Cold start**: Cannot predict new schools without history
3. **Extreme volatility**: Some schools change 100+ points between years
4. **External factors**: Policy changes, new programs not captured

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 👤 Author

**Zhivko Georgiev**
- GitHub: [@zhivko-georgiev](https://github.com/zhivko-georgiev)

## 🙏 Acknowledgments

- Historical data from Bulgarian Ministry of Education
- Built with XGBoost, pandas, scikit-learn, Streamlit
