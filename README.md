# 🏠 Bengaluru House Price Prediction

A machine learning project that predicts house prices in Bengaluru, India, based on square footage, number of bedrooms, bathrooms, and location. Includes a simple Streamlit web app so anyone can try it out.

**🔗 Live Demo:** [Add your deployed Streamlit link here]
**📓 Notebook:** [`bengaluru_house_price_prediction.ipynb`](./bengaluru_house_price_prediction.ipynb)

---

## 📌 Project Overview

Real estate pricing in a city like Bengaluru can vary wildly from one street to the next, and public listing data tends to be messy — mixed units, missing fields, and inconsistent formatting. The goal of this project was to take a raw, unrefined housing dataset and turn it into a reliable price prediction model, following a proper data science workflow from cleaning to deployment.

This project uses the [Bengaluru House Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) dataset from Kaggle, which contains thousands of property listings scraped from real estate platforms across the city, including details like location, size, total square footage, number of bathrooms, and price.

## 🎯 Problem Statement

Given basic property details — square footage, number of bedrooms, number of bathrooms, and location — predict the market price of a house in Bengaluru (in Lakhs). The challenge wasn't really the modeling itself, but getting there: the raw data had inconsistent size formats, sqft values written as ranges or in different units, hundreds of sparsely-represented locations, and clear outliers that would have thrown off any model trained on them directly.

## 🧰 Tech Stack

- **Language:** Python 3
- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib
- **Modeling:** scikit-learn (Linear Regression, Lasso, Decision Tree)
- **Model Selection:** GridSearchCV
- **Deployment:** Streamlit
- **Model Saving:** joblib

## 🧭 Data Cleaning & Feature Engineering

The raw dataset needed significant work before it was usable for modeling:

1. **Dropping Unused Columns** — columns like `society`, `area_type`, `availability`, and `balcony` weren't relevant to price prediction and were removed, along with rows containing missing values.

2. **Parsing Bedroom Counts** — the `size` column contained inconsistent text like "2 BHK" or "3 Bedroom." This was parsed into a clean numeric `Bedroom` column.

3. **Cleaning `total_sqft`** — this field was one of the messiest in the dataset. Some entries were plain numbers, others were ranges like "2100-2850," and some used entirely different units like Sq. Meter, Sq. Yards, Acres, or Guntha. Each of these had to be detected and converted into a consistent numeric square footage value.

4. **Grouping Rare Locations** — the dataset contained over a thousand unique location names, many with just one or two listings. To keep the feature space manageable, any location with fewer than 10 listings was grouped into a single `others` category.

5. **Removing Outliers** — three separate outlier-removal passes were applied:
   - Listings with an unrealistically small square footage per bedroom
   - Statistical outliers in price-per-square-foot within each location
   - Properties priced lower than expected for their BHK tier compared to similar listings in the same location (e.g. a 3 BHK priced below the average 2 BHK nearby)
   - Listings with implausible bathroom counts relative to bedroom count

6. **Encoding** — after cleaning, the `location` column was one-hot encoded so it could be used as input to the regression models.

## 🤖 Modeling Approach

Rather than picking one algorithm and hoping for the best, three regression models were compared using `GridSearchCV` with cross-validation to find the best-performing option and its optimal hyperparameters:

- **Linear Regression**
- **Lasso Regression**
- **Decision Tree Regressor**

## 📊 Results

| Model | Best Parameters | Best CV Score |
|---|---|---|
| Linear Regression | `fit_intercept: True` | 0.8615 |
| Lasso | `alpha: 1, selection: random` | 0.8295 |
| Decision Tree Regressor | `criterion: squared_error, splitter: random` | 0.8164 |

**Linear Regression performed best**, with an R² score of ~0.86, and was selected as the final model. This means the model explains roughly 86% of the variance in house prices using just square footage, bedroom count, bathroom count, and location — a solid result given how noisy the underlying data was before cleaning.

## 🖥️ Deployment

To make the model usable beyond a notebook, it's wrapped in a Streamlit app where a user can input square footage, bathrooms, BHK, and location, and instantly get a predicted price. The app loads the trained model directly and builds the same one-hot encoded feature vector used during training, so predictions stay consistent with what the model actually learned.
