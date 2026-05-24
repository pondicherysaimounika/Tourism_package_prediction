
# for data manipulation
import pandas as pd

# preprocessing and pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# model training and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# model serialization
import joblib

# utilities
import os

# hugging face
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# mlflow
import mlflow


# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset paths
Xtrain_path = "hf://datasets/Debugdemon/Tourism-package-prediction/Xtrain.csv"
Xtest_path =  "hf://datasets/Debugdemon/Tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/Debugdemon/Tourism-package-prediction/ytrain.csv"
ytest_path =  "hf://datasets/Debugdemon/Tourism-package-prediction/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)

ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print("Train and test datasets loaded successfully.")


# Define numeric and categorical features

numeric_features = [
    'Age',
    'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation'
]


# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 1, 10]
}

# Pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

with mlflow.start_run():

    # Grid Search
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring='accuracy'
    )

    grid_search.fit(Xtrain, ytrain)

    # Log parameter sets
    results = grid_search.cv_results_

    for i in range(len(results['params'])):

        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]

        with mlflow.start_run(nested=True):

            mlflow.log_params(param_set)
            mlflow.log_metric("mean_accuracy", mean_score)

    # Best model
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    train_accuracy = accuracy_score(ytrain, y_pred_train)
    test_accuracy = accuracy_score(ytest, y_pred_test)

    train_precision = precision_score(ytrain, y_pred_train)
    test_precision = precision_score(ytest, y_pred_test)

    train_recall = recall_score(ytrain, y_pred_train)
    test_recall = recall_score(ytest, y_pred_test)

    train_f1 = f1_score(ytrain, y_pred_train)
    test_f1 = f1_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "train_f1_score": train_f1,
        "test_f1_score": test_f1
    })

    # Save model locally
    model_path = "tourism_package_prediction_model.joblib"

    joblib.dump(best_model, model_path)

    # Log model artifact
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "Debugdemon/Tourism-package-prediction/tourism-package-prediction-model"
    repo_type = "model"

    # Check whether repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository '{repo_id}' already exists. Using it.")

    except RepositoryNotFoundError:

        print(f"Repository '{repo_id}' not found. Creating new repository...")

        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=False
        )

        print(f"Repository '{repo_id}' created.")

    # Upload model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    print("Model uploaded to Hugging Face successfully.")
