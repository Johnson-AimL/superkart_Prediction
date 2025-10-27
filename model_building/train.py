# for data manipulation
import pandas as pd
import numpy as np
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
ytrain = pd.read_csv(ytrain_path).values.ravel()  # Flatten to 1D array for regression
ytest = pd.read_csv(ytest_path).values.ravel()    # Flatten to 1D array for regression

# Print debug information
print("="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"X_train shape: {Xtrain.shape}")
print(f"X_test shape: {Xtest.shape}")
print(f"y_train shape: {ytrain.shape}")
print(f"y_test shape: {ytest.shape}")
print(f"\nColumns in dataset: {list(Xtrain.columns)}")
print(f"\nTarget statistics (Product_Store_Sales_Total):")
print(f"  Min: {ytrain.min():.2f}")
print(f"  Max: {ytrain.max():.2f}")
print(f"  Mean: {ytrain.mean():.2f}")
print(f"  Median: {np.median(ytrain):.2f}")

# Column Categorization - UPDATED WITH CORRECT COLUMN NAMES
# =============================
numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area", 
    "Product_MRP",
    "Store_Establishment_Year"
]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

# ID columns to exclude (if they exist in the data)
id_columns = ["Product_Id", "Store_Id"]

# Remove ID columns if they exist in the dataset
for col in id_columns:
    if col in Xtrain.columns:
        Xtrain = Xtrain.drop(columns=[col])
        Xtest = Xtest.drop(columns=[col])
        print(f"Dropped ID column: {col}")

# Verify all specified columns exist
missing_numeric = [col for col in numeric_features if col not in Xtrain.columns]
missing_categorical = [col for col in categorical_features if col not in Xtrain.columns]

if missing_numeric:
    print(f"\n⚠️ WARNING: Missing numeric columns: {missing_numeric}")
    numeric_features = [col for col in numeric_features if col in Xtrain.columns]
    
if missing_categorical:
    print(f"⚠️ WARNING: Missing categorical columns: {missing_categorical}")
    categorical_features = [col for col in categorical_features if col in Xtrain.columns]

print(f"\n✓ Using {len(numeric_features)} numeric features: {numeric_features}")
print(f"✓ Using {len(categorical_features)} categorical features: {categorical_features}")

# Preprocessing pipeline for columns
preprocessor = make_column_transformer(
    # For numeric columns
    # Replace NaNs with median as median is robust to outliers
    # Apply Standard scaling
    (Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),

    # For categorical columns
    # Replace NaNs with the most frequent value
    # Use One Hot Encoding as all the categorical features are nominal
    (Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_features),
    
    remainder='passthrough'  # Keep any other columns not specified
)

# Define base XGBoost model for REGRESSION
xgb_model = xgb.XGBRegressor(
    random_state=42,
    objective='reg:squarederror',  # Explicitly set for regression
    n_jobs=-1
)

# Define hyperparameter grid for REGRESSION
# Using a smaller grid for faster training initially
param_grid = {
    'xgbregressor__n_estimators': [100, 150],
    'xgbregressor__max_depth': [4, 6],
    'xgbregressor__colsample_bytree': [0.6, 0.8],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.5, 1.0, 5.0],
    'xgbregressor__subsample': [0.8, 0.9]
}

# Calculate total combinations
import itertools
total_combinations = len(list(itertools.product(*param_grid.values())))
print(f"\n🔍 Total parameter combinations to test: {total_combinations}")

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("\n" + "="*60)
print("STARTING MODEL TRAINING")
print("="*60)

with mlflow.start_run():
    # Hyperparameter tuning with regression scoring
    grid_search = GridSearchCV(
        model_pipeline, 
        param_grid, 
        cv=3,  # Using 3-fold CV for faster training
        n_jobs=-1,
        scoring='neg_mean_squared_error',  # Use MSE for regression
        verbose=1
    )
    
    print("Performing Grid Search...")
    grid_search.fit(Xtrain, ytrain)

    # Log best parameters
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV Score (RMSE): {np.sqrt(-grid_search.best_score_):.4f}")
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_rmse", np.sqrt(-grid_search.best_score_))

    # Log top 5 parameter combinations
    results = grid_search.cv_results_
    sorted_indices = np.argsort(results['mean_test_score'])[-5:]  # Top 5
    
    print("\nTop 5 parameter combinations:")
    for rank, idx in enumerate(sorted_indices[::-1], 1):
        score = -results['mean_test_score'][idx]  # Negate because it's negative MSE
        print(f"  {rank}. RMSE: {np.sqrt(score):.4f}")
        
        # Log as nested run
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][idx])
            mlflow.log_metric("cv_rmse", np.sqrt(score))
            mlflow.log_metric("cv_mse", score)
        time.sleep(0.5)  # Small delay for MLflow

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Make predictions for regression
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Calculate regression metrics
    train_mse = mean_squared_error(ytrain, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    train_r2 = r2_score(ytrain, y_pred_train)
    
    test_mse = mean_squared_error(ytest, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(ytest, y_pred_test)
    test_r2 = r2_score(ytest, y_pred_test)
    
    # Calculate MAPE only if there are no zeros in the target
    train_mape = None
    test_mape = None
    
    if not np.any(ytrain == 0):
        train_mape = mean_absolute_percentage_error(ytrain, y_pred_train)
    else:
        print("\n⚠️ Warning: Target contains zeros, skipping MAPE calculation")
    
    if not np.any(ytest == 0):
        test_mape = mean_absolute_percentage_error(ytest, y_pred_test)

    # Log all evaluation metrics
    metrics = {
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2
    }
    
    if train_mape is not None:
        metrics["train_mape"] = train_mape
    if test_mape is not None:
        metrics["test_mape"] = test_mape
        
    mlflow.log_metrics(metrics)

    # Print metrics for visibility
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print("\n📊 Training Metrics:")
    print(f"  MSE:   {train_mse:,.2f}")
    print(f"  RMSE:  {train_rmse:,.2f}")
    print(f"  MAE:   {train_mae:,.2f}")
    print(f"  R²:    {train_r2:.4f}")
    if train_mape is not None:
        print(f"  MAPE:  {train_mape:.4f}")
    
    print("\n📊 Test Metrics:")
    print(f"  MSE:   {test_mse:,.2f}")
    print(f"  RMSE:  {test_rmse:,.2f}")
    print(f"  MAE:   {test_mae:,.2f}")
    print(f"  R²:    {test_r2:.4f}")
    if test_mape is not None:
        print(f"  MAPE:  {test_mape:.4f}")

    # Prediction statistics
    print("\n📈 Prediction Statistics:")
    print(f"  Training predictions - Min: {y_pred_train.min():,.2f}, Max: {y_pred_train.max():,.2f}, Mean: {y_pred_train.mean():,.2f}")
    print(f"  Test predictions     - Min: {y_pred_test.min():,.2f}, Max: {y_pred_test.max():,.2f}, Mean: {y_pred_test.mean():,.2f}")
    print(f"  Actual values        - Min: {ytrain.min():,.2f}, Max: {ytrain.max():,.2f}, Mean: {ytrain.mean():,.2f}")

    # Save the model locally
    model_path = "best_superkart_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"\n💾 Model saved locally: {model_path}")

    # Upload to Hugging Face
    repo_id = "JohnsonSAimlarge/superkart-prediction"
    repo_type = "model"

    # Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"📦 Repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"📦 Creating new repository '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"✅ Repository created.")

    # Upload the best model to Hugging Face
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    
    print(f"\n✅ SUCCESS! Model uploaded to Hugging Face")
    print(f"   Repository: https://huggingface.co/{repo_id}")
    print(f"   Model file: {model_path}")
    print("="*60)
