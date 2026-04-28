# Best in PA Restaurants

This project is a web-based machine learning application for the **Best in PA** theme. It ranks Pennsylvania restaurants using a predicted Price-vs-Quality score based on rating, review count, price, city ranking, category comparison, and local review activity.

## Project Objective

The goal is to help users find strong restaurant options in Pennsylvania by combining multiple restaurant factors into one ranking score. Instead of ranking only by star rating, the app also considers price and review activity to estimate overall value.

## Features

- Filter restaurants by city, category, and maximum price
- Display top ranked matching restaurants
- Show model evaluation metrics
- Show most important model features
- Run locally through a Flask web application

## Structure

```text
best_PA_restaurants/
├── app/
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── results.html
│   ├── __init__.py
│   └── routes.py
├── data/
│   └── sample_restaurants.csv
├── src/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── model_loader.py
│   └── train.py
├── Report.md
├── run.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the web app:

```bash
python run.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## Optional Model Check

You can also run the model pipeline directly:

```bash
python train_model.py
```

This prints the row count, model metrics, and feature importance values.

## Model Summary

The project uses a `RandomForestRegressor` to predict a Price-vs-Quality score from engineered restaurant features.

Main features:

- star rating
- review count
- price level
- stars per dollar
- weighted stars
- city ranking
- stars compared to category average
- relative review buzz within the city
