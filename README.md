# 🏠 Bengaluru House Price Prediction

A machine learning project that predicts house prices in Bengaluru, India, based on square footage, number of bedrooms, bathrooms, and location. Includes a simple Streamlit web app so anyone can try it out.

**🔗 Live Demo:** [Add your deployed Streamlit link here]
**📓 Notebook:** [`bengaluru_house_price_prediction.ipynb`](./bengaluru_house_price_prediction.ipynb)

---

## 📌 Project Overview

This project uses the [Bengaluru House Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) dataset from Kaggle. The raw data is messy — mixed units, missing values, inconsistent entries — so most of the work here is cleaning and preparing the data before training a model to predict prices.

## 🧰 Tech Stack

- **Language:** Python 3
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib
- **Modeling:** scikit-learn (Linear Regression, Lasso, Decision Tree)
- **Model Selection:** GridSearchCV
- **Deployment:** Streamlit
- **Model Saving:** joblib

## 🧭 Steps Followed

1. **Data Cleaning** — dropped unused columns (`society`, `area_type`, `availability`, `balcony`) and removed missing values
2. **Feature Engineering** — converted `size` into a numeric `Bedroom` count, and cleaned up `total_sqft` values (some were ranges or in different units like Sq. Meter, Acres, etc.)
3. **Location Grouping** — grouped locations with fewer than 10 listings into a single `others` category
4. **Outlier Removal** — removed unrealistic listings using price-per-sqft, BHK pricing comparisons, and bathroom counts
5. **Encoding** — converted location into one-hot encoded columns
6. **Model Selection** — compared Linear Regression, Lasso, and Decision Tree using GridSearchCV
7. **Deployment** — built a Streamlit app to serve predictions from the trained model

## 📊 Results

| Model | Best Parameters | Best CV Score |
|---|---|---|
| Linear Regression | `fit_intercept: True` | 0.8615 |
| Lasso | `alpha: 1, selection: random` | 0.8295 |
| Decision Tree Regressor | `criterion: squared_error, splitter: random` | 0.8164 |

Linear Regression performed best and was used for the final model.

## 📂 Project Structure

```
.
├── bengaluru_house_price_prediction.ipynb   # Data cleaning + model training
├── app.py                                    # Streamlit prediction app
├── requirements.txt
├── README.md
├── data/                                     # Bengaluru_House_Data.csv
└── model/                                    # lr_reg_model.pickle
```

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Download the [dataset](./Bengaluru_House_Data.csv) and place `Bengaluru_House_Data.csv` in a `data/` folder.

**1. Train the model**
Run `bengaluru_house_price_prediction.ipynb` from top to bottom. This creates `model/lr_reg_model.pickle`.

**2. Run the app**
```bash
streamlit run app.py
