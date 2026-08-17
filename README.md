# Hospital Uncompensated Care Cost Predictor

An end-to-end machine learning project predicting uncompensated care costs 
across US hospitals using CMS Hospital Cost Report (HCRIS) data from 2019–2023.

---

## What is Uncompensated Care?

Hospitals are required to treat patients regardless of ability to pay.
Uncompensated care represents the cost of services provided without payment —
either through charity care (deliberate write-offs) or bad debt (unpaid bills).
This financial burden varies widely across hospitals and directly impacts their
ability to stay operational, particularly in rural communities.

---

## Project Goal

Build a predictive model that estimates how much uncompensated care a hospital 
will absorb based on its size, location, payer mix, and financial profile.
This enables early identification of financially at-risk hospitals.

---

## Dataset

- **Source:** CMS Hospital Provider Cost Report (HCRIS)
- **Years:** 2019–2023
- **Records:** 22,745 hospital-year observations
- **Features:** 27 raw columns → 30 after feature engineering
- **Download:** https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/Cost-Reports

---

## Project Structure

```
hospital-financial-risk-analysis/
├── notebooks/
│   ├── 01_data_loading.ipynb        # Data ingestion and cleaning
│   ├── 02_eda.ipynb                 # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb # Feature creation and transformation
│   ├── 04_modeling.ipynb            # Model training and evaluation
│   └── 05_results_summary.ipynb     # Full project summary
├── model/
│   └── random_forest_model.pkl      # Saved trained model
├── outputs/                         # All generated visualizations
├── app.py                           # Streamlit prediction app
└── requirements.txt
```

---

## Key Findings

### Target Variable
- Heavily right-skewed: mean $8.9M vs median $2.9M
- 78.7% of hospitals carry under $10M in uncompensated care
- log1p transformation applied for modeling; predictions reversed via expm1

### Geographic Patterns
- GU, FL, and NJ lead in median uncompensated care per hospital
- California lower than expected — attributable to aggressive Medicaid expansion under the ACA

### Rural vs Urban
- Urban hospitals carry 3x more in raw dollars ($5.4M vs $1.7M)
- Per-bed ratio reverses this finding: rural $43K/bed vs urban $40K/bed
- Rural hospitals carry disproportionate burden relative to their capacity

### COVID Era Trends (2019–2023)
- Uncompensated care dipped in 2020–2021 despite expectations
- Driven by reduced patient volume and federal CARES Act relief funding
- Mean surged to ~$10M by 2023 as relief ended and costs rose

### Feature Engineering
- Created `Uncompensated_Care_Per_Bed` ratio — became the #1 most important 
  feature in the final model (did not exist in raw data)
- Encoded `Rural Versus Urban` as binary `Is_Urban` feature

---

## Modeling Results

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression | 0.4871 | $5,135,344 |
| Random Forest | **0.9977** | **$362,077** |
| XGBoost | 0.9932 | $503,575 |

**Random Forest selected as final model.**

The large gap between Linear Regression and tree-based models confirms 
that hospital uncompensated care relationships are genuinely non-linear.

---

## Streamlit App

Run the prediction app locally:

```bash
pip install streamlit
streamlit run app.py
```

Enter hospital characteristics to get a predicted uncompensated care cost 
with national median context.

---

## How to Reproduce

1. Download HCRIS data (2019–2023) from the CMS link above
2. Place CSV files in the `data/` folder
3. Run notebooks 01 through 05 in order
4. Launch the Streamlit app with `streamlit run app.py`

---

## Tools and Libraries

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit
- GitHub Codespaces

---

## Application

https://hospital-financial-risk-analysis-7akcrfqt5vuwusobzucabq.streamlit.app

