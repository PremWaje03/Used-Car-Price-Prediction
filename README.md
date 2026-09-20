# 🚗 Used Car Resale Price Prediction

A machine learning web application that predicts the resale/listed price of a used car using vehicle characteristics such as model year, mileage, fuel type, transmission, body type, brand, model, owner type, location, and selected technical specifications.

## 🌐 Live Demo

**[Open AutoValue AI](https://used-car-price-prediction-prem.streamlit.app/)**

The application is built with Streamlit and uses a trained Random Forest Regression model for price prediction.

---

## 🎯 Project Objective

The objective of this project is to develop an end-to-end supervised machine learning system for estimating used-car prices.

The project covers:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Numerical and categorical feature preprocessing
- Regression model implementation
- Model evaluation
- Random Forest hyperparameter tuning
- Model serialization using Joblib
- Interactive Streamlit deployment

---

## 🤖 Machine Learning Approach

**Problem Type:** Supervised Learning  
**Task:** Regression  
**Target Variable:** `listed_price`

### Models Implemented

1. Linear Regression
2. K-Nearest Neighbors (KNN) Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Tuned Random Forest Regression

### Evaluation Metrics

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

---

## 📊 Model Results

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---:|---:|---:|
| Linear Regression | 185,114.61 | 638,100.28 | 0.6957 |
| KNN Regression | 119,556.91 | 527,469.64 | 0.7921 |
| Decision Tree Regression | 124,372.00 | 423,779.89 | 0.8658 |
| Random Forest Regression | 96,330.80 | 381,873.75 | 0.8910 |
| Tuned Random Forest Regression | 96,062.31 | 376,565.49 | 0.8940 |

The tuned Random Forest model is used by the deployed application.

---

## 🔍 Feature Importance

The Random Forest feature-importance analysis identified several influential features, including:

- Maximum Power Delivered
- Vehicle Age
- Vehicle Width
- Wheel Base
- Maximum Torque Delivered
- Vehicle Length
- OEM / Brand
- Car Model
- Kilometers Driven
- Number of Cylinders

Feature importance describes the contribution of features within the fitted Random Forest model and should not be interpreted as proof of causation.

---

## 🏗️ Project Structure

```text
Used-Car-Price-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── requirements_streamlit.txt
├── .gitignore
├── .gitattributes
│
├── models/
│   ├── car_price_metadata.joblib
│   ├── car_price_preprocessor.joblib
│   └── car_price_random_forest.joblib
│
├── outputs/
│   ├── model_comparison_mae.png
│   ├── model_comparison_rmse.png
│   ├── model_comparison_r2.png
│   ├── random_forest_actual_vs_predicted.png
│   ├── random_forest_first_100_predictions.png
│   ├── random_forest_feature_importance.png
│   └── random_forest_feature_importance.csv
│
└── src/
    ├── predict.py
    └── preprocessing_final.py
