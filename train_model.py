"""
train_model.py
Optional script for checking the model pipeline from the command line.
"""

from src.model_loader import load_model_and_data


def main():
    df, model, scaler, cv_r2, feature_importance, metrics = load_model_and_data()

    print("Model training complete")
    print(f"Rows: {len(df)}")
    print(f"Cross-validation R2: {cv_r2}")
    print(f"Test metrics: {metrics}")
    print("\nTop feature importances:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
