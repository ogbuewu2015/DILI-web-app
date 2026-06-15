
# preprocessing.py

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


# ============================================================
# 1️⃣ REMOVE LOW VARIANCE FEATURES
# ============================================================

def remove_low_variance(input_data, threshold=0.1):
    selector = VarianceThreshold(threshold)
    selector.fit(input_data)

    retained_features = input_data.columns[selector.get_support(indices=True)]

    # Safe selection
    reduced_data = input_data[retained_features]

    return reduced_data, retained_features, selector


# ============================================================
# 2️⃣ REMOVE CORRELATED FEATURES
# ============================================================

def remove_correlated_features(descriptors, threshold=0.9):
    correlated_matrix = descriptors.corr().abs()

    upper_triangle = correlated_matrix.where(
        np.triu(np.ones(correlated_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    descriptors_correlated_dropped = descriptors.drop(columns=to_drop, axis=1)

    return descriptors_correlated_dropped, to_drop


# ============================================================
# 3️⃣ SCALE TRAINING DATA
# ============================================================

def scale_training_data(train_data):
    # Use ALL numeric columns (not just float)
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns
    non_numeric_columns = train_data.drop(columns=numeric_columns)

    scaler = StandardScaler()

    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(train_data[numeric_columns]),
        columns=numeric_columns,
        index=train_data.index
    )

    scaled_train = pd.concat([scaled_numeric, non_numeric_columns], axis=1)

    # Preserve original column order
    scaled_train = scaled_train[train_data.columns]

    return scaled_train, scaler, numeric_columns


# ============================================================
# 4️⃣ SCALE NEW DATA
# ============================================================

def scale_new_data(new_data, scaler, numeric_columns):
    new_data = new_data.copy()

    # Ensure all expected numeric columns exist
    for col in numeric_columns:
        if col not in new_data.columns:
            new_data[col] = 0

    numeric_data = new_data[numeric_columns]

    scaled_numeric = pd.DataFrame(
        scaler.transform(numeric_data),
        columns=numeric_columns,
        index=new_data.index
    )

    non_numeric_columns = new_data.drop(columns=numeric_columns, errors='ignore')

    scaled_data = pd.concat([scaled_numeric, non_numeric_columns], axis=1)

    # Preserve column order
    scaled_data = scaled_data[new_data.columns]

    return scaled_data

