

# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for converting text data into numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face authentication and upload
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset path from Hugging Face
DATASET_PATH =  "hf://datasets/Debugdemon/Tourism-package-prediction/tourism_package_prediction.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

# Drop unique identifier column
df.drop(columns=['CustomerID'], inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()

categorical_cols = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation'
]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))

# Define target column
target_col = 'ProdTaken'

# Split into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Save files locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# List of files to upload
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload files to Hugging Face dataset repo
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="/Debugdemon/Tourism-package-prediction/tourism-project-data",
        repo_type="dataset",
    )

print("Train-test files uploaded successfully.")
