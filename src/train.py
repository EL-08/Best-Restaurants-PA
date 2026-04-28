"""
train.py
Trains and evaluates the Random Forest model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def build_features(df): # Engineer ML features from raw restaurant columns.
    feat = pd.DataFrame()

    feat["stars"] = df["stars"].clip(1, 5)
    feat["review_count"] = np.log1p(df["review_count"])
    feat["price"] = df["price"].fillna(2).clip(1, 4)
    feat["stars_per_dollar"] = feat["stars"] / feat["price"]
    feat["weighted_stars"] = feat["stars"] * np.log1p(df["review_count"]) / 5
    feat["city_rank"] = df.groupby("city")["stars"].rank(pct=True)
    feat["stars_vs_cat_mean"] = df["stars"] - df.groupby("category")["stars"].transform("mean")

    max_rev = df.groupby("city")["review_count"].transform("max")
    feat["relative_buzz"] = df["review_count"] / (max_rev + 1)

    return feat.fillna(0)

def make_target(df): # Build the Price-vs-Quality score from 0 to 100.
    
    raw = (
        df["stars"] * 18
        + np.log1p(df["review_count"]) * 1.5
        - (df["price"].fillna(2) - 1) * 4
        + (df["stars"] - df.groupby("category")["stars"].transform("mean")) * 3
    )

    return ((raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 100).clip(0, 100)

def train_model(df): # Train the model and return the trained objects/results.
    X = build_features(df)
    y = make_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train_sc, y_train)
    predictions = model.predict(X_test_sc).clip(0, 100)

    metrics = {
        "r2": round(r2_score(y_test, predictions), 4),
        "mae": round(mean_absolute_error(y_test, predictions), 4),
        "rmse": round(mean_squared_error(y_test, predictions) ** 0.5, 4),
    }

    X_sc = scaler.fit_transform(X)
    model.fit(X_sc, y)
    cv_r2 = cross_val_score(model, X_sc, y, cv=3, scoring="r2").mean()

    df = df.copy()
    df["pq_score"] = model.predict(X_sc).clip(0, 100)

    feature_importance = pd.DataFrame({
        "Feature": list(X.columns),
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df, model, scaler, round(cv_r2, 4), feature_importance, metrics
