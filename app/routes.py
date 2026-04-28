"""
routes.py
Flask routes for displaying and filtering restaurant rankings.
"""

from flask import Blueprint, render_template, request

from src.model_loader import load_model_and_data

main = Blueprint("main", __name__)

df, model, scaler, cv_r2, feature_importance, metrics = load_model_and_data()


@main.route("/", methods=["GET"])
def index():
    cities = sorted(df["city"].unique())
    categories = sorted(df["category"].unique())
    top_restaurants = df.sort_values("pq_score", ascending=False).head(10)

    return render_template(
        "index.html",
        cities=cities,
        categories=categories,
        top_restaurants=top_restaurants,
        cv_r2=cv_r2,
        metrics=metrics,
    )


@main.route("/results", methods=["POST"])
def results():
    selected_city = request.form.get("city", "All")
    selected_category = request.form.get("category", "All")
    max_price = int(request.form.get("max_price", 4))

    filtered = df.copy()

    if selected_city != "All":
        filtered = filtered[filtered["city"] == selected_city]

    if selected_category != "All":
        filtered = filtered[filtered["category"] == selected_category]

    filtered = filtered[filtered["price"] <= max_price]
    filtered = filtered.sort_values("pq_score", ascending=False).head(10)

    return render_template(
        "results.html",
        restaurants=filtered,
        selected_city=selected_city,
        selected_category=selected_category,
        max_price=max_price,
        feature_importance=feature_importance.head(6),
        cv_r2=cv_r2,
        metrics=metrics,
    )
