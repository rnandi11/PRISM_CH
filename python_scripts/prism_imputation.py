import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def impute_missing_features(data, feature_columns, n_neighbors=5):
    result = data.copy()

    missing_columns = [
        column for column in feature_columns
        if result[column].isna().any()
    ]

    if not missing_columns:
        return result

    # Use every model feature for neighbor-distance calculations.
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(result[feature_columns]),
        columns=feature_columns,
        index=result.index,
    )

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled = pd.DataFrame(
        imputer.fit_transform(scaled),
        columns=feature_columns,
        index=result.index,
    )

    imputed = pd.DataFrame(
        scaler.inverse_transform(imputed_scaled),
        columns=feature_columns,
        index=result.index,
    )

    # Replace only the originally missing cells.
    for column in missing_columns:
        missing_mask = result[column].isna()
        result.loc[missing_mask, column] = imputed.loc[missing_mask, column]

    if "giant_plt" in missing_columns:
        result["giant_plt"] = np.where(
            result["giant_plt"] >= 0.5, 1, 0
        )

    return result