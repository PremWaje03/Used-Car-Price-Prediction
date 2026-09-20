import pandas as pd
import time
import numpy as np

# Use a non-GUI Matplotlib backend for VS Code/terminal execution.
# This prevents Tkinter thread/deallocator errors on Windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Machine Learning Model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Evaluation Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

PROGRAM_START_TIME = time.perf_counter()
import os
os.makedirs("../outputs", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# ============================================
# FAST EXECUTION SETTINGS
# ============================================

# Normal runs should keep this False.
# Set True only when you intentionally want to
# repeat the expensive GridSearchCV experiment.
RUN_HYPERPARAMETER_TUNING = False

# Graphs are saved directly to outputs/.
# A non-GUI Matplotlib backend is used so the script can run
# safely from the VS Code terminal without Tkinter errors.
SHOW_PLOTS = False

# Use all available CPU cores for KNN prediction.
KNN_N_JOBS = -1

# Number of trees used by the main Random Forest.
# 100 keeps the model comparable with your earlier results.
RF_N_ESTIMATORS = 100


# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("../data/cars_data_clean.csv")

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# ============================================
# 2. DISPLAY BASIC INFORMATION
# ============================================

print("\nFirst 5 Rows:")
print(df.head())

print("\nNumber of Columns:", len(df.columns))


# ============================================
# 3. COLUMNS TO REMOVE
# ============================================

columns_to_remove = [
    "usedCarSkuId",
    "loc",
    "ip",
    "images",
    "imgCount",
    "threesixty",
    "dvn",
    "discountValue",

    "variant",
    "City",

    "top_features",
    "comfort_features",
    "interior_features",
    "exterior_features",
    "safety_features",

    "Color",
    "Engine Type",

    "Front Tread",
    "Rear Tread",
    "Gross Weight",
    "Top Speed",
    "Acceleration",

    "model_type_new",
    "exterior_color",

    "Fuel Suppy System",
    "Compression Ratio",
    "Alloy Wheel Size",
    "Ground Clearance Unladen",
    "Bore",
    "Stroke"
]


# ============================================
# 4. CREATE MODELING DATASET
# ============================================

df_model = df.drop(columns=columns_to_remove)

print("\nOriginal Shape:", df.shape)
print("Modeling Shape:", df_model.shape)


# ============================================
# 5. REMOVE IDENTIFIED ANOMALOUS RECORD
# ============================================

df_model = df_model[
    ~(
        (df_model["oem"] == "ford") &
        (df_model["model"] == "ford ecosport") &
        (df_model["listed_price"] == 550000555)
    )
]

print("\nShape after anomaly removal:", df_model.shape)


# ============================================
# 6. CREATE VEHICLE AGE
# ============================================

reference_year = df_model["myear"].max() + 1

df_model["vehicle_age"] = (
    reference_year - df_model["myear"]
)

print("\nReference Year:", reference_year)

print("\nVehicle Age:")
print(df_model["vehicle_age"].head())


# ============================================
# 7. REMOVE ORIGINAL YEAR
# ============================================

df_model = df_model.drop(columns=["myear"])


# ============================================
# 8. REMOVE DUPLICATE ROWS
# ============================================

duplicates = df_model.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

df_model = df_model.drop_duplicates()

print("Shape after removing duplicates:", df_model.shape)


# ============================================
# 9. SEPARATE FEATURES AND TARGET
# ============================================

X = df_model.drop(columns=["listed_price"])

y = df_model["listed_price"]


# ============================================
# 10. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================
# 11. DISPLAY DATASET INFORMATION
# ============================================

print("\n==============================")
print("FINAL DATA INFORMATION")
print("==============================")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================
# 12. MISSING VALUES
# ============================================

missing_values = X_train.isnull().sum()

missing_values = missing_values[
    missing_values > 0
].sort_values(ascending=False)

print("\nMissing Values in Training Data:")
print(missing_values)


# ============================================
# 13. MISSING VALUE PERCENTAGE
# ============================================

missing_percentage = (
    X_train.isnull().sum() /
    len(X_train) * 100
)

missing_percentage = missing_percentage[
    missing_percentage > 0
].sort_values(ascending=False)

print("\nMissing Value Percentage in Training Data:")
print(missing_percentage)


# ============================================
# 14. DEFINE NUMERICAL FEATURES
# ============================================

numerical_features = [
    "km",
    "No of Cylinder",
    "Valves per Cylinder",
    "Length",
    "Width",
    "Height",
    "Wheel Base",
    "Kerb Weight",
    "Seats",
    "Doors",
    "Cargo Volume",
    "Max Power Delivered",
    "Max Power At",
    "Max Torque Delivered",
    "Max Torque At",
    "vehicle_age"
]


# ============================================
# 15. DEFINE CATEGORICAL FEATURES
# ============================================

categorical_features = [
    "body",
    "transmission",
    "fuel",
    "oem",
    "model",
    "owner_type",
    "state",
    "Gear Box",
    "Drive Type",
    "Steering Type",
    "Front Brake Type",
    "Rear Brake Type",
    "Tyre Type",
    "Turbo Charger",
    "Super Charger",
    "Valve Configuration"
]


# ============================================
# 16. NUMERICAL PREPROCESSING PIPELINE
# ============================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================
# 17. CATEGORICAL PREPROCESSING PIPELINE
# ============================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ============================================
# 18. COMBINE BOTH PIPELINES
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================
# 19. FIT AND TRANSFORM TRAINING DATA
# ============================================

X_train_processed = (
    preprocessor.fit_transform(X_train)
)


# ============================================
# 20. TRANSFORM TESTING DATA
# ============================================

X_test_processed = (
    preprocessor.transform(X_test)
)


# ============================================
# 21. GET FEATURE NAMES
# ============================================

feature_names = (
    preprocessor.get_feature_names_out()
)


# ============================================
# 22. DISPLAY PREPROCESSING RESULTS
# ============================================

print("\n==============================")
print("PREPROCESSING RESULTS")
print("==============================")

print(
    "Original X_train shape:",
    X_train.shape
)

print(
    "Processed X_train shape:",
    X_train_processed.shape
)

print(
    "\nOriginal X_test shape:",
    X_test.shape
)

print(
    "Processed X_test shape:",
    X_test_processed.shape
)

print(
    "\nTotal processed features:",
    len(feature_names)
)


# ============================================
# 23. DISPLAY FIRST 30 PROCESSED FEATURES
# ============================================

print("\nFirst 30 Processed Features:")

for feature in feature_names[:30]:
    print(feature)


# ============================================
# PREPROCESSING VERIFICATION
# ============================================

print("\n==============================")
print("PREPROCESSING VERIFICATION")
print("==============================")

# Check NaN values in sparse matrix
train_nan = np.isnan(X_train_processed.data).sum()
test_nan = np.isnan(X_test_processed.data).sum()

print("Training data contains NaN:", train_nan)
print("Testing data contains NaN:", test_nan)

print("Processed data type:", type(X_train_processed))

# ============================================
# 25. LINEAR REGRESSION MODEL
# ============================================

print("\n==============================")
print("TRAINING LINEAR REGRESSION")
print("==============================")

linear_model = LinearRegression()

linear_model.fit(
    X_train_processed,
    y_train
)

print("Linear Regression model trained successfully!")


# ============================================
# 26. MAKE PREDICTIONS
# ============================================

y_pred = linear_model.predict(
    X_test_processed
)


# ============================================
# 27. ACTUAL VS PREDICTED PRICES
# ============================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print("\n==============================")
print("ACTUAL VS PREDICTED PRICES")
print("==============================")

print(comparison)


# ============================================
# 28. CALCULATE EVALUATION METRICS
# ============================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================
# 29. DISPLAY MODEL RESULTS
# ============================================

print("\n==============================")
print("LINEAR REGRESSION RESULTS")
print("==============================")

print(f"MAE : ₹{mae:,.2f}")
print(f"MSE : {mse:,.2f}")
print(f"RMSE: ₹{rmse:,.2f}")
print(f"R²  : {r2:.4f}")


# ============================================
# 30. KNN REGRESSION
# ============================================

print("\n==============================")
print("TRAINING KNN REGRESSION")
print("==============================")

knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=KNN_N_JOBS)
knn_model.fit(X_train_processed, y_train)

print("KNN Regression model trained successfully!")


# ============================================
# 31. KNN PREDICTIONS
# ============================================

knn_pred = knn_model.predict(X_test_processed)


# ============================================
# 32. KNN ACTUAL VS PREDICTED PRICES
# ============================================

knn_comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": knn_pred[:10]
})

print("\n==============================")
print("KNN ACTUAL VS PREDICTED PRICES")
print("==============================")
print(knn_comparison)


# ============================================
# 33. KNN EVALUATION METRICS
# ============================================

knn_mae = mean_absolute_error(y_test, knn_pred)
knn_mse = mean_squared_error(y_test, knn_pred)
knn_rmse = knn_mse ** 0.5
knn_r2 = r2_score(y_test, knn_pred)


# ============================================
# 34. DISPLAY KNN RESULTS
# ============================================

print("\n==============================")
print("KNN REGRESSION RESULTS")
print("==============================")
print(f"MAE : ₹{knn_mae:,.2f}")
print(f"MSE : {knn_mse:,.2f}")
print(f"RMSE: ₹{knn_rmse:,.2f}")
print(f"R²  : {knn_r2:.4f}")


# ============================================
# 35. DECISION TREE REGRESSION
# ============================================

print("\n==============================")
print("TRAINING DECISION TREE REGRESSION")
print("==============================")

decision_tree_model = DecisionTreeRegressor(
    random_state=42
)

decision_tree_model.fit(
    X_train_processed,
    y_train
)

print("Decision Tree Regression model trained successfully!")


# ============================================
# 36. DECISION TREE PREDICTIONS
# ============================================

decision_tree_pred = decision_tree_model.predict(
    X_test_processed
)


# ============================================
# 37. DECISION TREE ACTUAL VS PREDICTED
# ============================================

decision_tree_comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": decision_tree_pred[:10]
})

print("\n==============================")
print("DECISION TREE ACTUAL VS PREDICTED PRICES")
print("==============================")

print(decision_tree_comparison)


# ============================================
# 38. DECISION TREE EVALUATION METRICS
# ============================================

dt_mae = mean_absolute_error(
    y_test,
    decision_tree_pred
)

dt_mse = mean_squared_error(
    y_test,
    decision_tree_pred
)

dt_rmse = dt_mse ** 0.5

dt_r2 = r2_score(
    y_test,
    decision_tree_pred
)


# ============================================
# 39. DISPLAY DECISION TREE RESULTS
# ============================================

print("\n==============================")
print("DECISION TREE REGRESSION RESULTS")
print("==============================")

print(f"MAE : ₹{dt_mae:,.2f}")
print(f"MSE : {dt_mse:,.2f}")
print(f"RMSE: ₹{dt_rmse:,.2f}")
print(f"R²  : {dt_r2:.4f}")


# ============================================
# 40. RANDOM FOREST REGRESSION
# ============================================

print("\n==============================")
print("TRAINING RANDOM FOREST REGRESSION")
print("==============================")

random_forest_model = RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train_processed,
    y_train
)

print("Random Forest Regression model trained successfully!")

# ============================================
# 45. SAVE TRAINED MODEL AND PREPROCESSOR
# ============================================

print("\n==============================")
print("SAVING MODEL ARTIFACTS")
print("==============================")

# Save the preprocessing object and the trained Random Forest.
# This allows predict.py to make predictions later without
# retraining the entire ML pipeline.
joblib.dump(
    preprocessor,
    "../models/car_price_preprocessor.joblib"
)

joblib.dump(
    random_forest_model,
    "../models/car_price_random_forest.joblib"
)

model_metadata = {
    "reference_year": int(reference_year),
    "target_column": "listed_price",
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "model_type": "RandomForestRegressor",
    "n_estimators": RF_N_ESTIMATORS,
    "random_state": 42
}

joblib.dump(
    model_metadata,
    "../models/car_price_metadata.joblib"
)

print("Preprocessor saved successfully.")
print("Random Forest model saved successfully.")
print("Model metadata saved successfully.")
print("Artifacts saved inside: models/")


# ============================================
# 41. RANDOM FOREST PREDICTIONS
# ============================================

random_forest_pred = random_forest_model.predict(
    X_test_processed
)


# ============================================
# 42. RANDOM FOREST ACTUAL VS PREDICTED
# ============================================

random_forest_comparison = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": random_forest_pred[:10]
})

print("\n==============================")
print("RANDOM FOREST ACTUAL VS PREDICTED PRICES")
print("==============================")

print(random_forest_comparison)


# ============================================
# 43. RANDOM FOREST EVALUATION METRICS
# ============================================

rf_mae = mean_absolute_error(
    y_test,
    random_forest_pred
)

rf_mse = mean_squared_error(
    y_test,
    random_forest_pred
)

rf_rmse = rf_mse ** 0.5

rf_r2 = r2_score(
    y_test,
    random_forest_pred
)


# ============================================
# 44. DISPLAY RANDOM FOREST RESULTS
# ============================================

print("\n==============================")
print("RANDOM FOREST REGRESSION RESULTS")
print("==============================")

print(f"MAE : ₹{rf_mae:,.2f}")
print(f"MSE : {rf_mse:,.2f}")
print(f"RMSE: ₹{rf_rmse:,.2f}")
print(f"R²  : {rf_r2:.4f}")


# ============================================
# 45. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================

if RUN_HYPERPARAMETER_TUNING:

    print("\n==============================")
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("==============================")

    rf_tuning_model = RandomForestRegressor(
        random_state=42,
        n_jobs=1
    )

    # Compact search space for laptop execution.
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=rf_tuning_model,
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=2
    )

    print("\nStarting GridSearchCV...")
    print("Parameter combinations: 8")
    print("Cross-validation folds: 3")
    print("Total model fits: 24")

    grid_search.fit(
        X_train_processed,
        y_train
    )

    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)

    best_cv_mae = -grid_search.best_score_

    print("\nBest Cross-Validation MAE:")
    print(f"₹{best_cv_mae:,.2f}")

    best_random_forest_model = grid_search.best_estimator_

    tuned_random_forest_pred = (
        best_random_forest_model.predict(X_test_processed)
    )

    tuned_rf_mae = mean_absolute_error(
        y_test,
        tuned_random_forest_pred
    )

    tuned_rf_mse = mean_squared_error(
        y_test,
        tuned_random_forest_pred
    )

    tuned_rf_rmse = tuned_rf_mse ** 0.5

    tuned_rf_r2 = r2_score(
        y_test,
        tuned_random_forest_pred
    )

    print("\n==============================")
    print("TUNED RANDOM FOREST RESULTS")
    print("==============================")

    print(f"MAE : ₹{tuned_rf_mae:,.2f}")
    print(f"MSE : {tuned_rf_mse:,.2f}")
    print(f"RMSE: ₹{tuned_rf_rmse:,.2f}")
    print(f"R²  : {tuned_rf_r2:.4f}")

    print("\nTuning completed.")

else:

    print("\n==============================")
    print("HYPERPARAMETER TUNING SKIPPED")
    print("==============================")

    print(
        "Using the original Random Forest for the normal fast run."
    )

    # No expensive GridSearchCV is executed during normal runs.
    best_random_forest_model = random_forest_model

    tuned_random_forest_pred = random_forest_pred

    tuned_rf_mae = rf_mae
    tuned_rf_mse = rf_mse
    tuned_rf_rmse = rf_rmse
    tuned_rf_r2 = rf_r2


# ============================================
# 46. TUNED RANDOM FOREST STATUS
# ============================================

if RUN_HYPERPARAMETER_TUNING:
    print("\nTuned Random Forest model evaluated successfully.")
else:
    print("\nTuned Random Forest step skipped during fast run.")
    print("The original Random Forest metrics will be used.")


# ============================================
# 52. FINAL MODEL COMPARISON
# ============================================

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

model_names = [
    "Linear Regression",
    "KNN Regression",
    "Decision Tree Regression",
    "Random Forest Regression"
]

model_mae = [
    mae,
    knn_mae,
    dt_mae,
    rf_mae
]

model_mse = [
    mse,
    knn_mse,
    dt_mse,
    rf_mse
]

model_rmse = [
    rmse,
    knn_rmse,
    dt_rmse,
    rf_rmse
]

model_r2 = [
    r2,
    knn_r2,
    dt_r2,
    rf_r2
]

# Add tuned model only when tuning was actually executed.
if RUN_HYPERPARAMETER_TUNING:
    model_names.append("Tuned Random Forest Regression")
    model_mae.append(tuned_rf_mae)
    model_mse.append(tuned_rf_mse)
    model_rmse.append(tuned_rf_rmse)
    model_r2.append(tuned_rf_r2)

model_comparison = pd.DataFrame({
    "Model": model_names,
    "MAE": model_mae,
    "MSE": model_mse,
    "RMSE": model_rmse,
    "R2": model_r2
})

print(model_comparison.to_string(index=False))


# ============================================
# 53. END
# ============================================

print("\n==============================")
print("PROJECT STEP COMPLETED")
print("==============================")

print("Data preprocessing completed.")
print("Linear Regression training completed.")
print("KNN Regression training completed.")
print("Decision Tree Regression training completed.")
print("Random Forest Regression training completed.")
print("Random Forest hyperparameter tuning step completed or skipped.")
print("All models evaluated successfully.")
print(f"Elapsed execution time so far: {(time.perf_counter() - PROGRAM_START_TIME) / 60:.2f} minutes.")

# ============================================
# 54. MODEL COMPARISON VISUALIZATION
# ============================================

print("\n==============================")
print("MODEL COMPARISON VISUALIZATION")
print("==============================")

selected_model_name = "Random Forest Regression"

print("\nSelected model for visualization:", selected_model_name)


def save_plot(filename):
    """Save a plot without using the Tkinter GUI backend."""
    output_path = os.path.join("../outputs", filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", output_path)


# ============================================
# 55. MAE COMPARISON
# ============================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=model_comparison,
    x="Model",
    y="MAE"
)

plt.title("Model Comparison - Mean Absolute Error")
plt.xlabel("Machine Learning Model")
plt.ylabel("MAE (₹)")
plt.xticks(rotation=20)
plt.tight_layout()

save_plot("model_comparison_mae.png")


# ============================================
# 56. RMSE COMPARISON
# ============================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=model_comparison,
    x="Model",
    y="RMSE"
)

plt.title("Model Comparison - Root Mean Squared Error")
plt.xlabel("Machine Learning Model")
plt.ylabel("RMSE (₹)")
plt.xticks(rotation=20)
plt.tight_layout()

save_plot("model_comparison_rmse.png")


# ============================================
# 57. R² COMPARISON
# ============================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=model_comparison,
    x="Model",
    y="R2"
)

plt.title("Model Comparison - R² Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("R² Score")
plt.xticks(rotation=20)
plt.tight_layout()

save_plot("model_comparison_r2.png")


# ============================================
# 58. ACTUAL VS PREDICTED - RANDOM FOREST
# ============================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    random_forest_pred,
    alpha=0.5
)

min_price = min(y_test.min(), random_forest_pred.min())
max_price = max(y_test.max(), random_forest_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.title("Random Forest - Actual vs Predicted Prices")
plt.xlabel("Actual Price (₹)")
plt.ylabel("Predicted Price (₹)")
plt.tight_layout()

save_plot("random_forest_actual_vs_predicted.png")


# ============================================
# 59. ACTUAL VS PREDICTED - FIRST 100 RECORDS
# ============================================

plt.figure(figsize=(12, 6))

sample_size = min(100, len(y_test))

plt.plot(
    range(sample_size),
    y_test.values[:sample_size],
    label="Actual Price"
)

plt.plot(
    range(sample_size),
    random_forest_pred[:sample_size],
    label="Predicted Price"
)

plt.title(
    "Random Forest - Actual vs Predicted Prices "
    "(First 100 Test Records)"
)

plt.xlabel("Test Record")
plt.ylabel("Price (₹)")
plt.legend()
plt.tight_layout()

save_plot("random_forest_first_100_predictions.png")


print("\nVisualization completed successfully.")


# ============================================
# 60. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================

print("\n==============================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("==============================")

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features:")
print(feature_importance.head(20).to_string(index=False))


# ============================================
# 61. FEATURE IMPORTANCE VISUALIZATION
# ============================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest - Top 15 Feature Importances")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

save_plot("random_forest_feature_importance.png")


# ============================================
# 62. SAVE FEATURE IMPORTANCE
# ============================================

feature_importance.to_csv(
    "../outputs/random_forest_feature_importance.csv",
    index=False
)

print(
    "\nFeature importance saved to:"
    " outputs/random_forest_feature_importance.csv"
)


# ============================================
# 63. FINAL STATUS
# ============================================

print("\n==============================")
print("VISUALIZATION AND FEATURE IMPORTANCE COMPLETED")
print("==============================")

print("\n==============================")
print("TOTAL EXECUTION TIME")
print("==============================")

print(
    f"{(time.perf_counter() - PROGRAM_START_TIME) / 60:.2f} minutes"
)

