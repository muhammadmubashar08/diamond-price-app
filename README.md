# 💎 Diamond Price Prediction System
 
**Course:** Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus
**Assignment 3, Task 2:** Applying, Evaluating, and Comparing Multiple ML Algorithms on a Large Real-World Dataset
 
🔗 **Live App:** https://diamond-price-app-kphqe83hwlv8lxublydyob.streamlit.app/
 
## Problem
Predict the price (USD) of a diamond from its physical attributes. This is a regression problem, similar to the Car Price Prediction System.
 
## Dataset
- **Diamonds dataset**, full version: 53,940 instances
- **Inputs (9):** carat, cut, color, clarity, depth, table, x, y, z
- **Target:** price (USD)
- 20 rows with a zero dimension (x, y or z) were removed as invalid records, leaving 53,920 rows.
## Method
- **Preprocessing:** one-hot encoding for categorical features (cut, color, clarity) and standard scaling for numeric features, all inside a scikit-learn `Pipeline` so preprocessing is fitted only on training folds (no data leakage).
- **Evaluation:** 10-Fold Cross-Validation (shuffled, `random_state=42`) on the full dataset.
- **Metric:** RMSE (lower is better). MAE and R² are also reported.
## Algorithms Compared
| # | Algorithm |
|---|-----------|
| 1 | Ridge Regression |
| 2 | ElasticNet |
| 3 | AdaBoost Regressor |
| 4 | Extra Trees Regressor |
| 5 | Histogram Gradient Boosting Regressor |
 
## Results
10-Fold Cross-Validation on the full dataset (53,920 rows after cleaning), sorted by RMSE:
 
| Rank | Algorithm | Mean RMSE | Std RMSE | Mean MAE | Mean R² | Time (s) |
|------|-----------|-----------|----------|----------|---------|----------|
| 1 | Extra Trees | 530.91 | 20.41 | 263.49 | 0.982 | 65.6 |
| 2 | HistGradientBoosting | 545.18 | 12.82 | 283.46 | 0.981 | 13.1 |
| 3 | Ridge Regression | 1137.13 | 51.12 | 735.69 | 0.919 | 1.2 |
| 4 | ElasticNet | 1145.87 | 43.67 | 746.23 | 0.917 | 14.4 |
| 5 | AdaBoost | 1518.24 | 60.22 | 1226.93 | 0.854 | 66.0 |
 
**Best model:** Extra Trees (RMSE 530.91), retrained on the full dataset and saved as `diamond_price_model.pkl`.
 
**Observations:**
- The two tree-based ensembles (Extra Trees and HistGradientBoosting) clearly beat the linear models, which suggests the price relationship with carat, clarity and color is non-linear.
- HistGradientBoosting is only about 14 RMSE behind Extra Trees, is about 5x faster, and has the most stable error across folds (lowest std).
- AdaBoost performed worst, likely because its default shallow trees underfit this data.
## Repository Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit web app |
| `diamond_price_model.pkl` | Trained model (full preprocessing pipeline + best regressor) |
| `requirements.txt` | Python dependencies |
 
## Run Locally
```bash
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows
# venv\Scripts\Activate.ps1       # PowerShell
pip install -r requirements.txt
streamlit run app.py
```
 
## Deployment
Deployed on **Streamlit Community Cloud** from this GitHub repository (main branch, `app.py`).
 
## Author
Muhammad Mubashar Murtaza 
