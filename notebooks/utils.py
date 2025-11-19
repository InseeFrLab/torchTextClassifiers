import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def categorize_surface(
    df: pd.DataFrame, surface_feature_name: int, like_sirene_3: bool = True
) -> pd.DataFrame:
    """
    Categorize the surface of the activity.

    Args:
        df (pd.DataFrame): DataFrame to categorize.
        surface_feature_name (str): Name of the surface feature.
        like_sirene_3 (bool): If True, categorize like Sirene 3.

    Returns:
        pd.DataFrame: DataFrame with a new column "surf_cat".
    """
    df_copy = df.copy()
    df_copy[surface_feature_name] = df_copy[surface_feature_name].replace("nan", np.nan)
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(float)
    # Check surface feature exists
    if surface_feature_name not in df.columns:
        raise ValueError(f"Surface feature {surface_feature_name} not found in DataFrame.")
    # Check surface feature is a float variable
    if not (pd.api.types.is_float_dtype(df_copy[surface_feature_name])):
        raise ValueError(f"Surface feature {surface_feature_name} must be a float variable.")

    if like_sirene_3:
        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy[surface_feature_name],
            bins=[0, 120, 400, 2500, np.inf],
            labels=["1", "2", "3", "4"],
        ).astype(str)
    else:
        # Log transform the surface
        df_copy["surf_log"] = np.log(df[surface_feature_name])

        # Categorize the surface
        df_copy["surf_cat"] = pd.cut(
            df_copy.surf_log,
            bins=[0, 3, 4, 5, 12],
            labels=["1", "2", "3", "4"],
        ).astype(str)

    df_copy[surface_feature_name] = df_copy["surf_cat"].replace("nan", "0")
    df_copy[surface_feature_name] = df_copy[surface_feature_name].astype(int)
    df_copy = df_copy.drop(columns=["surf_log", "surf_cat"], errors="ignore")
    return df_copy


def clean_and_tokenize_df(
    df,
    categorical_features=["CJ", "NAT", "TYP", "CRT"],
    text_feature="libelle_processed",
    label_col="apet_finale",
):
    df.fillna("nan", inplace=True)
    les = []
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        les.append(le)

    df = categorize_surface(df, "SRF", like_sirene_3=True)
    df = df[[text_feature, "CJ", "NAT", "TYP", "SRF", "CRT", label_col]]

    return df, les
