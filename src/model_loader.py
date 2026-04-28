"""
model_loader.py
Main model-loading function used by the Flask application.
"""

from src.data_generation import generate_data
from src.train import train_model


def load_model_and_data():
    """Generate data, train the model, and return everything the UI needs."""
    df = generate_data()
    return train_model(df)
