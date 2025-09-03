# -*- coding: utf-8 -*-
"""02_end_to_end_machine_learning_project.py

Complete end-to-end machine learning project for California housing data.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tarfile
import urllib.request
from packaging import version
import sklearn
from scipy import stats
from scipy.stats import randint, uniform, geom, expon, loguniform
import joblib

# Set up matplotlib configuration
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Assert Python and sklearn versions
assert sys.version_info >= (3, 7)
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

def load_housing_data():
    """Load the California housing dataset."""
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Save matplotlib figure."""
    path = Path("images") / "end_to_end_project" / f"{fig_id}.{fig_extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Handle root_mean_squared_error import for different sklearn versions
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

def main():
    """Main function to run the complete housing project."""
    print("Welcome to Machine Learning Housing Project!")
    
    # Load data
    housing = load_housing_data()
    print("Data loaded successfully!")
    print(f"Dataset shape: {housing.shape}")
    
    # Explore data
    print("\n=== Data Exploration ===")
    print(housing.head())
    print("\nData info:")
    housing.info()
    print("\nOcean proximity value counts:")
    print(housing["ocean_proximity"].value_counts())
    print("\nData description:")
    print(housing.describe())
    
    # Create test set
    from sklearn.model_selection import train_test_split
    
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    print(f"\nTraining set size: {len(strat_train_set)}")
    print(f"Test set size: {len(strat_test_set)}")
    
    # Prepare data for ML
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    
    # Data preprocessing pipeline
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Numerical pipeline
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    
    # Categorical pipeline
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )
    
    # Column transformer
    num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
                   "total_bedrooms", "population", "households", "median_income"]
    cat_attribs = ["ocean_proximity"]
    
    preprocessing = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
    
    # Full pipeline with Random Forest
    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ])
    
    # Train the model
    print("\n=== Training Model ===")
    full_pipeline.fit(housing, housing_labels)
    print("Model trained successfully!")
    
    # Evaluate on test set
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    
    final_predictions = full_pipeline.predict(X_test)
    
    final_rmse = root_mean_squared_error(y_test, final_predictions)
    print(f"\nFinal RMSE on test set: {final_rmse:.2f}")
    
    # Save the model
    joblib.dump(full_pipeline, "california_housing_model.pkl")
    print("Model saved as 'california_housing_model.pkl'")
    
    # Load and test the model
    print("\n=== Testing Saved Model ===")
    model_reloaded = joblib.load("california_housing_model.pkl")
    test_predictions = model_reloaded.predict(X_test.iloc[:5])
    print(f"Predictions on first 5 test instances: {test_predictions.round(-2)}")
    print(f"Actual values: {y_test.iloc[:5].values}")
    
    print("\n=== Project Complete ===")

if __name__ == "__main__":
    main()
