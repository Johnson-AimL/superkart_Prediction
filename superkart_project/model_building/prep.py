# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ---------------------------
# Config
# ---------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
DESIGNATION_CARDINALITY_THRESHOLD = 40  # heuristic

# ---------------------------
# Helper functions
# ---------------------------
def safe_to_numeric(series):
    """Convert series to numeric, coercing errors to NaN"""
    return pd.to_numeric(series, errors='coerce')

# ---------------------------
# Main processing
# ---------------------------
def main():
    # Load data
    # Define constants for the dataset and output paths
    api = HfApi(token=os.getenv("HF_TOKEN"))
    DATASET_PATH = "hf://datasets/JohnsonSAimlarge/superkart-prediction/SuperKart.csv"
    df_raw = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")

    # Remove the first column which is marked as 'Unnamed'
    df = df_raw.iloc[:, 1:]


    # Duplicate checks
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f" - Found {duplicate_count} duplicate rows. Removing them...")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print(" - No duplicate rows found.")

    # Null value check (before cleaning)
    null_before = df.isnull().sum()
    print("Null values BEFORE cleaning:")
    print(null_before[null_before > 0])


    # Null check after cleaning
    null_after = df.isnull().sum()
    if null_after.sum() == 0:
        print(" - No null values remain after cleaning.")
    else:
        print("Null values AFTER cleaning:")
        print(null_after[null_after > 0])


    # Define target variable
    y = df['Product_Store_Sales_Total']

    # Define predictor matrix (X)
    X = df.drop(columns=['Product_Store_Sales_Total'])


    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train.to_csv("Xtrain.csv",index=False)
    X_test.to_csv("Xtest.csv",index=False)
    y_train.to_csv("ytrain.csv",index=False)
    y_test.to_csv("ytest.csv",index=False)

    # Save in huggingface dataset
    files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

    for file_path in files:
      api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="JohnsonSAimlarge/superkart-prediction",
        repo_type="dataset",
    )

    # Final summary
    print("\nSummary of preprocessing:")
    print(f" - Final encoded dataset shape: {df.shape}")
    print(f" - Train set: {X_train.shape}")
    print(f" - Test set: {X_test.shape}")

if __name__ == "__main__":
    main()
