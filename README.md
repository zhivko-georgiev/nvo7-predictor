# NVO 7th Grade Rankings Prediction System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)](https://nvo7-predictor.streamlit.app/)

ML-based prediction system for Bulgarian 7th grade high school admission cutoff scores (NVO - Национално външно оценяване).

**🌐 Try it now: [nvo7-predictor.streamlit.app](https://nvo7-predictor.streamlit.app/)**

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
| R1 MAE | 18.97 points | **10.60 points** |
| R1 Within 10 pts | 54.0% | 68.4% |
| R1 Within 20 pts | 69.3% | 83.9% |
| R2 MAE | 24.80 points | **14.99 points** |
| R1 Interval Coverage (±2.5σ) | 77% | ~85% |

**Reliable predictions** = profiles with ≥2 years history, volatility <20, and previous year data.

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

1. **Equipercentile Backbone**: Convert last year's cutoff to a percentile rank in last year's exam distribution, then map that rank through this year's distribution. This handles exam difficulty shifts by construction — no extrapolation needed.

2. **ML Residual Correction**: A delta model (XGBoost) predicts the year-over-year change not explained by the difficulty shift — capturing trend, mean reversion, and capacity effects.

3. **Hybrid Blending**: The final prediction blends the equipercentile base with the model residual, using a weight tuned by leave-one-year-out cross-validation (capped at 0.8 to prevent overfitting).

4. **Profile Rename Detection**: Automatically detects renamed profiles across years via string similarity + language-token matching, maintaining continuity in the training data.

5. **Confidence Intervals**: ±2.5× historical volatility, targeting ~87% empirical coverage for reliable predictions.

### Model Details

- **Algorithm**: XGBoost with shallow trees (max_depth=3) + equipercentile rank mapping
- **Regularization**: Strong L1/L2 to prevent overfitting
- **Training**: Walk-forward feature construction (no data leakage), temporal validation
- **Training data**: ~1000 samples (4 years × ~270 schools)
- **Training time**: <5 seconds

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
