# 🏠 Bengaluru House Price Prediction

A machine learning web app that predicts house prices in Bengaluru, India, based on square footage, number of bedrooms, bathrooms, and location. Built end-to-end — from raw data cleaning to a deployed Streamlit interface.

<!-- Optional: add a screenshot once the app is running
![App Screenshot](assets/app_screenshot.png)
-->

**🔗 Live Demo:** [Add your deployed Streamlit link here]
**📓 Notebook:** [`ML_project_1.ipynb`](./ML_project_1.ipynb)

---

## 📌 Overview

Real estate pricing is notoriously noisy — inconsistent listings, mixed units, and outliers. This project builds a clean, reproducible pipeline that takes the raw [Bengaluru House Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) dataset (Kaggle) and turns it into a working price prediction model with a simple web interface anyone can use.

## ✨ Features

- End-to-end ML pipeline: cleaning → feature engineering → outlier removal → encoding → model selection
- Multi-model comparison (Linear Regression, Lasso, Decision Tree) via `GridSearchCV`
- Custom outlier-removal logic based on price-per-sqft and BHK pricing anomalies
- Interactive Streamlit UI for real-time predictions
- Fully reproducible: relative paths, pinned dependencies, no hardcoded environment assumptions

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib |
| Modeling | scikit-learn (Linear Regression, Lasso, Decision Tree) |
| Model Selection | GridSearchCV |
| Deployment | Streamlit |
| Serialization | joblib |

## 🧭 Methodology

1. **Data Cleaning** — drop unused columns (`society`, `area_type`, `availability`, `balcony`), handle missing values
2. **Feature Engineering** — parse `size` into a numeric `Bedroom` count; normalize `total_sqft` (handles ranges and unit strings like Sq. Meter, Acres, Guntha, etc.)
3. **Location Grouping** — bucket locations with fewer than 10 listings into `others` to reduce dimensionality
4. **Outlier Removal** — three passes: statistical outliers in price/sqft, BHK-tier pricing anomalies, implausible bathroom counts
5. **Encoding** — one-hot encode location
6. **Model Selection** — `GridSearchCV` across Linear Regression, Lasso, and Decision Tree Regressor
7. **Deployment** — final model served through a Streamlit app

## 📊 Results

| Model | Best Params | Best CV Score |
|---|---|---|
| Linear Regression | _fill in from your GridSearchCV output_ | _e.g. 0.XX_ |
| Lasso | _fill in_ | _fill in_ |
| Decision Tree Regressor | _fill in_ | _fill in_ |

> Run the notebook's model selection cell and paste the `scores` output here — this table is the strongest signal for anyone reviewing your work.

## 📂 Project Structure

```
.
├── ML_project_1.ipynb    # Full data pipeline + model training
├── app.py                 # Streamlit prediction UI
├── requirements.txt
├── README.md
├── .gitignore
├── data/                  # Bengaluru_House_Data.csv (not tracked in git)
└── model/                 # lr_reg_model.pickle (not tracked in git)
```

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Download the [dataset from Kaggle](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) and place `Bengaluru_House_Data.csv` in a `data/` folder.

**1. Train the model**
Run `ML_project_1.ipynb` top to bottom. This produces `model/lr_reg_model.pickle`.

**2. Launch the app**
```bash
streamlit run app.py



