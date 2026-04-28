"""
data_generation.py
Generates synthetic PA restaurant data for the demo project.
"""

import numpy as np
import pandas as pd

def generate_data(n=800, seed=42):
    """Generate synthetic Pennsylvania restaurant data."""
    rng = np.random.default_rng(seed)

    pa_cities = [
        "Philadelphia", "Pittsburgh", "Allentown", "Erie", "Reading",
        "Scranton", "Bethlehem", "Lancaster", "Harrisburg", "York",
        "Altoona", "Wilkes-Barre", "Chester", "State College", "Easton",
    ]

    categories = [
        "Italian", "American", "Chinese", "Mexican", "Japanese",
        "Thai", "Indian", "Pizza", "Burgers", "Seafood",
        "Breakfast", "Bakery", "BBQ", "Vegan", "Mediterranean",
    ]

    city_weights = np.array([30, 25, 8, 7, 6, 5, 5, 4, 4, 3, 2, 2, 2, 2, 2], dtype=float)
    city_weights = city_weights / city_weights.sum()

    city_arr = rng.choice(pa_cities, size=n, p=city_weights)
    category_arr = rng.choice(categories, size=n)
    price_arr = rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.45, 0.22, 0.08])
    stars_arr = np.clip(rng.normal(3.8, 0.6, n) + (price_arr - 2.5) * 0.1, 1.0, 5.0).round(1)
    review_arr = np.clip(rng.lognormal(4.5, 1.2, n).astype(int), 1, 5000)

    suffixes = ["Kitchen", "Grill", "Bistro", "Eatery", "House", "Corner", "Table", "Place"]
    names = [f"{city_arr[i]} {category_arr[i]} {rng.choice(suffixes)}" for i in range(n)]

    return pd.DataFrame({
        "name": names,
        "city": city_arr,
        "category": category_arr,
        "stars": stars_arr,
        "review_count": review_arr,
        "price": price_arr,
    })
