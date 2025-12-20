# Data Sources and Format

This document explains how to obtain and organize the data files needed for the NVO prediction system.

## Official Data Source

All data comes from **РУО София-град** (Regional Education Department - Sofia):
- Website: [https://ruo-sofia-grad.com/](https://ruo-sofia-grad.com/)
- Section: "Прием след 7 клас" (Admission after 7th grade)

## Required Files

For each year, you need the following files:

### 1. Ranking Files (`klasirane_X_YEAR.xlsx`)

These contain the admission results for each round (1-4).

**How to obtain:**
1. Go to [ruo-sofia-grad.com](https://ruo-sofia-grad.com/)
2. Navigate to "Прием след 7 клас" → "Класиране" (Rankings)
3. Download files for each round (1st, 2nd, 3rd, 4th)

**Expected columns:**
- Училище - име (School name)
- Паралелка - име (Profile name)
- Мъже (Male) - minimum score
- Жени (Female) - minimum score
- Общо (Total) - minimum score

### 2. Exam Score Distribution (`average_grades_BEL_MAT-YEAR.xlsx`)

Contains the distribution of exam scores by gender.

**How to obtain:**
1. Go to [ruo-sofia-grad.com](https://ruo-sofia-grad.com/)
2. Navigate to "Прием след 7 клас" → "Статистика" (Statistics)
3. Download the combined BEL+MAT score distribution file

**Expected format:**
- Score ranges (0-10, 10-20, etc.)
- Count of students per range by gender

### 3. School Capacity (`schools_YEAR.xlsx`) - Optional

Contains the number of available spots per school/profile.

**How to obtain:**
1. Go to [ruo-sofia-grad.com](https://ruo-sofia-grad.com/)
2. Navigate to "Прием след 7 клас" → "Училища" (Schools)
3. Download the school list with capacity

## File Organization

Place files in the following structure:

```
files/
├── 2022/
│   ├── klasirane_1_2022.xlsx
│   ├── klasirane_2_2022.xlsx
│   ├── klasirane_3_2022.xlsx
│   ├── average_grades_BEL_MAT-2022.xlsx
│   └── schools_2022.xlsx (optional)
├── 2023/
│   ├── klasirane_1_2023.xlsx
│   ├── klasirane_2_2023.xlsx
│   ├── klasirane_3_2023.xlsx
│   ├── klasirane_4_2023.xlsx
│   ├── average_grades_BEL_MAT-2023.xlsx
│   └── schools_2023.xlsx (optional)
├── 2024/
│   └── ... (same structure)
└── 2025/
    └── ... (same structure)
```

## Data Format Notes

### Bulgarian Number Format

The Excel files use Bulgarian number format:
- Decimal separator: comma (`,`) instead of period (`.`)
- Example: `493,25` means `493.25`

The system automatically handles this conversion.

### Header Detection

The ranking files have varying header structures. The system automatically detects the header row by looking for columns containing "Мъже" (Male) and "Жени" (Female).

### Missing Data

- If a school/profile has no applicants in a round, the score may be 0 or empty
- The system treats 0 scores as missing data
- Schools with missing data in the previous year cannot be reliably predicted

## Data Quality

For best prediction accuracy:
- Include at least 3 years of historical data
- Ensure all ranking rounds (1-4) are present
- Include exam distribution files for each year

## Privacy Note

All data is publicly available from the Ministry of Education. No personal student information is included - only aggregate statistics and minimum admission scores.
