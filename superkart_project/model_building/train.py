# for data manipulation
import pandas as pd
# for data preprocessing and pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import numpy as np
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import time

# set MLflow UI on port 5000
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Set hugginhgface repository
repo_id = "JohnsonSAimlarge/superkart-prediction"

# Download file safely from HF Hub
Xtrain_path = hf_hub_download(repo_id=repo_id, filename="Xtrain.csv", repo_type="dataset")
Xtest_path = hf_hub_download(repo_id=repo_id, filename="Xtest.csv", repo_type="dataset")
ytrain_path = hf_hub_download(repo_id=repo_id, filename="ytrain.csv", repo_type="dataset")
ytest_path = hf_hub_download(repo_id=repo_id, filename="ytest.csv", repo_type="dataset")

# Read the csv files in dataframes
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Column Categorization
# =============================
numeric_features = [
    "product_weight",
    "product_allocated_area",
    "product_mrp",
    "store_establishment_year",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "CityTier"
]

categorical_features = [
    "product_sugar_content",
    "product_type",
    "store_size",
    "store_location_city_type",
    "store_type"
]

# Define binary features if any (was referenced but not defined in original)
binary_features = []  # Add your binary columns here if applicable

# Preprocessing pipeline for columns
preprocessor = make_column_transformer(
    # For numeric column
    # Replace NaNs with median as median is robust to outliers
    # Apply Standard scaling
    (Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    # For categorical columns
    # Replace NaNs with the most frequent value; if nothing exists, mark as 'Unknown
    # Use One Hot Encoding as all the categorical features are nominal
    (Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features),

    # For columns with binary values (if any)
    # Replace NaNs with the most frequent value
    (Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]), binary_features) if binary_features else ('passthrough', [])
)

# Define base XGBoost model for REGRESSION
xgb_model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid for REGRESSION
param_grid = {
    'xgbregressor__n_estimators': [100, 150, 200],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__colsample_bytree': [0.5, 0.7, 0.9],
    'xgbregressor__colsample_bylevel': [0.5, 0.7, 0.9],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.1, 1.0, 10.0],
    'xgbregressor__reg_alpha': [0, 0.1, 1.0],
    'xgbregressor__subsample': [0.7, 0.8, 0.9],
    'xgbregressor__min_child_weight': [1, 3, 5]
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Hyperparameter tuning with regression scoring
    grid_search = GridSearchCV(
        model_pipeline, 
        param_grid, 
        cv=5, 
        n_jobs=-1,
        scoring='neg_mean_squared_error'  # Use MSE for regression
    )
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)
        # to handle the restrictions of number of requests to MLflow per minute
        time.sleep(1)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Make predictions for regression (no threshold needed)
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Calculate regression metrics
    train_mse = mean_squared_error(ytrain, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    train_r2 = r2_score(ytrain, y_pred_train)
    train_mape = mean_absolute_percentage_error(ytrain, y_pred_train)

    test_mse = mean_squared_error(ytest, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(ytest, y_pred_test)
    test_r2 = r2_score(ytest, y_pred_test)
    test_mape = mean_absolute_percentage_error(ytest, y_pred_test)

    # Logs all evaluation metrics for train and test sets into MLflow
    mlflow.log_metrics({
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "train_mape": train_mape,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_mape": test_mape
    })

    # Print metrics for visibility
    print("Training Metrics:")
    print(f"  MSE: {train_mse:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R2 Score: {train_r2:.4f}")
    print(f"  MAPE: {train_mape:.4f}")
    
    print("\nTest Metrics:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R2 Score: {test_r2:.4f}")
    print(f"  MAPE: {test_mape:.4f}")

    # Save the model locally
    model_path = "best_superkart_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "JohnsonSAimlarge/superkart-prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        # creating a new one if the space does not exist
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Upload the best model in huggingface repo
    api.upload_file(
        path_or_fileobj="best_superkart_model_v1.joblib",
        path_in_repo="best_superkart_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
