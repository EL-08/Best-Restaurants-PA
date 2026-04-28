# Best in PA Restaurants
## Capstone Project Report

---

## 1. Description of the Project
The application is a full-stack machine learning web application that helps users discover high-value restaurants across Pennsylvania. The project's core objective is to move beyond the limitations of a simple star-rating system by computing a composite Price-vs-Quality (PQ) score that accounts for multiple restaurant characteristics simultaneously.

Built with Flask, it exposes a browser-based interface where users can filter restaurants by city, cuisine category, and maximum price tier. After submitting the form, the app queries a trained Random Forest model and returns the top-ranked restaurants along with model evaluation metrics and feature importances. A secondary command-line entry point (train_model.py) allows the full pipeline — data generation, feature engineering, training, and evaluation — to be run and inspected without the web server.

Key objectives:
- Train a supervised regression model to predict a meaningful restaurant quality score from structured features.
- Engineer informative features that capture value relative to price, popularity relative to city peers, and quality relative to cuisine category.
- Deliver the model as a usable Flask web application with filtering, ranking, and model-transparency views.

---

## 2. Significance of the Project
Restaurant discovery is a ubiquitous, high-frequency decision problem. Existing tools (Google Maps, Yelp, TripAdvisor) predominantly surface results by raw star rating or sponsored placement. This approach has well-documented drawbacks: star inflation across categories, recency bias in review aggregates, and no adjustment for price level. A restaurant with 4.2 stars at \$15 per person may offer better value than one with 4.5 stars at \$60 per person, but naive sorting will always favor the latter.

Novelty and meaningfulness:
- Price-adjusted ranking: The PQ score explicitly penalizes higher price tiers and rewards value, so budget-conscious users receive results that reflect their constraints.
- Contextual comparison: Features such as `stars_vs_cat_mean` and `city_rank` measure a restaurant's quality *relative* to its competitive context rather than on an absolute scale, which is a stronger signal of genuine quality.
- Transparent ML: The results page exposes feature importances and cross-validation metrics, making the ranking interpretable rather than a black box — an increasingly valued property in recommendation systems.
- Local focus: Targeting Pennsylvania specifically makes the tool practically useful for students, residents, and visitors in the Commonwealth, aligning with a real regional need.

---

## 3. Data Collection
The project uses synthetically generated Pennsylvania restaurant data produced by `src/data_generation.py`. Synthetic data was chosen to ensure a reproducible, self-contained demonstration pipeline that can be run without API keys or external data access. The generator uses a seeded NumPy random number generator (`seed=42`) to produce fully deterministic results.

The synthetic data is designed to mirror the schema and statistical properties of data that would be collected from sources such as the Yelp Fusion API or Google Places API. The city distribution follows approximate real-world population weights, and star ratings are drawn from a normal distribution centered at 3.8 with a slight positive skew for higher price levels, reflecting real-world reviewer behavior.

### Dataset Quantity
 
| Property | Value |
|---|---|
| Total records | 800 restaurants |
| Cities covered | 15 Pennsylvania cities |
| Cuisine categories | 15 distinct categories |
| Features in raw data | 6 columns |
| Features after engineering | 8 model-ready features |

### Metadata
 
| Column | Type | Description | Range / Values |
|---|---|---|---|
| `name` | string | Synthetic restaurant name | e.g., "Philadelphia Italian Kitchen" |
| `city` | string | Pennsylvania city | 15 cities including Philadelphia, Pittsburgh, Lancaster |
| `category` | string | Cuisine type | Italian, American, Chinese, Mexican, Japanese, Thai, Indian, Pizza, Burgers, Seafood, Breakfast, Bakery, BBQ, Vegan, Mediterranean |
| `stars` | float | Average star rating | 1.0 – 5.0 (clipped normal distribution) |
| `review_count` | int | Number of user reviews | 1 – 5,000 (log-normal distribution) |
| `price` | int | Price tier | 1 ($) to 4 ($$$$); P(1)=0.25, P(2)=0.45, P(3)=0.22, P(4)=0.08 |

### Sample Data
 
| Name | City | Category | Stars | Reviews | Price |
|---|---|---|---:|---:|---:|
| Philadelphia Italian Kitchen | Philadelphia | Italian | 4.7 | 1320 | 2 ($$) |
| Pittsburgh BBQ Grill | Pittsburgh | BBQ | 4.5 | 980 | 2 ($$) |
| Lancaster Breakfast Table | Lancaster | Breakfast | 4.6 | 430 | 1 ($) |
| Harrisburg Vegan Bistro | Harrisburg | Vegan | 4.4 | 370 | 2 ($$) |
| State College Pizza House | State College | Pizza | 4.3 | 620 | 1 ($) |
| Allentown Mexican Corner | Allentown | Mexican | 4.2 | 500 | 2 ($$) |
| Erie Seafood Place | Erie | Seafood | 4.5 | 260 | 3 ($$$) |
| Reading Bakery Eatery | Reading | Bakery | 4.1 | 320 | 1 ($) |
| Scranton American Grill | Scranton | American | 4.0 | 450 | 2 ($$) |
| Bethlehem Thai Kitchen | Bethlehem | Thai | 4.6 | 290 | 2 ($$) |
 
---

## 4. Data Processing and Feature Engineering
All preprocessing and feature engineering is handled in `src/train.py` within the `build_features()` function.
 
### Preprocessing Steps
- **Stars clipping:** Raw star values are clipped to the range [1, 5] to eliminate any out-of-range anomalies that could arise from data collection errors.
 
- **Review count log-scaling:** Review counts follow a highly skewed log-normal distribution (a restaurant may have 5 reviews or 5,000). Applying `log1p` (log(1 + x)) compresses this dynamic range, preventing a handful of extremely popular venues from dominating features that use review count.
 
- **Price imputation and clipping:** Missing price values are filled with 2 (the most common tier, representing $$) and then clipped to [1, 4].

### Engineered Features
Eight features are constructed from the six raw columns:
 
| Feature | Formula | Rationale |
|---|---|---|
| `stars` | `clip(stars, 1, 5)` | Cleaned raw rating signal |
| `review_count` | `log1p(review_count)` | Log-scaled popularity |
| `price` | `fillna(2).clip(1, 4)` | Cleaned price tier |
| `stars_per_dollar` | `stars / price` | Direct value-for-money ratio |
| `weighted_stars` | `stars × log1p(review_count) / 5` | Rating credibility-weighted by volume |
| `city_rank` | Percentile rank of stars within city | Relative quality among city peers |
| `stars_vs_cat_mean` | `stars − mean(stars) by category` | Quality above/below cuisine average |
| `relative_buzz` | `review_count / max(review_count) in city` | Review activity relative to the city's busiest restaurant |
 
### Target Variable Construction
The Price-vs-Quality target score is defined deterministically (not labeled by humans) using the formula:
 
```
raw = stars × 18 + log1p(review_count) × 1.5 − (price − 1) × 4 + stars_vs_cat_mean × 3
pq_score = min-max normalize(raw) × 100, clipped to [0, 100]
```
 
This formula encodes the product's design philosophy: stars are the dominant signal (weight 18), review volume adds credibility (weight 1.5), higher price reduces the score (penalty of 4 per tier above baseline), and above-average quality for the cuisine type receives a bonus (weight 3). Min-max normalization ensures all scores are expressed on a 0–100 scale for interpretability.
 
### Train/Test Split
Data is split 80/20 (640 training, 160 test) using `train_test_split` with `random_state=42`. Features are standardized via `StandardScaler` fit on the training set only to prevent data leakage.
 
---
 
## 5. Model Development

### Model Architecture
The project uses a **Random Forest Regressor** from scikit-learn with the following hyperparameters:
 
| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 80 | Sufficient tree count for stable variance reduction without excess compute |
| `max_depth` | 8 | Limits overfitting while allowing moderately complex splits |
| `min_samples_leaf` | 3 | Prevents overfitting on small leaf nodes |
| `random_state` | 42 | Reproducibility |
 
**Input:** An 8-dimensional feature vector per restaurant (see Feature Engineering section).
 
**Output:** A single continuous PQ score in the range [0, 100].
 
Random Forest was selected because it is robust to feature scale differences (though StandardScaler is still applied), handles nonlinear interactions between features naturally, and directly provides feature importances as a transparency mechanism.
 
### Test Results
The model was evaluated on the held-out 20% test set and via 3-fold cross-validation on the full dataset:
 
| Metric | Value | Interpretation |
|---|---|---|
| Test R² | **0.9875** | The model explains 98.75% of variance in PQ scores on unseen data |
| Test MAE | **1.2683** | Average absolute prediction error of ~1.27 points on a 0–100 scale |
| Test RMSE | **1.8450** | Root mean squared error, penalizing larger misses |
| Cross-validation R² (3-fold) | **0.9858** | Consistent with test R²; no meaningful overfitting detected |
 
### Feature Importances
 
| Feature | Importance |
|---|---|
| `stars` | 0.4888 |
| `stars_vs_cat_mean` | 0.3026 |
| `stars_per_dollar` | 0.1084 |
| `city_rank` | 0.0822 |
| `weighted_stars` | 0.0124 |
| `relative_buzz` | 0.0027 |
| `review_count` | 0.0024 |
| `price` | 0.0005 |
 
### Discussion of Results
The model achieves very high R² values (≥ 0.985) because the target variable is a deterministic function of the features — there is no irreducible noise since both input features and target scores are derived from the same raw columns. This is by design: for a synthetic dataset, it confirms the model correctly learns the scoring function. On real-world data, where human review behavior introduces genuine noise, R² values would be lower and MAE/RMSE would be more informative benchmarks.
 
The dominance of `stars` (49%) and `stars_vs_cat_mean` (30%) in feature importances is consistent with the target formula, which weights stars heavily. The presence of `stars_per_dollar` (11%) and `city_rank` (8%) confirms the model captures the value-adjusted and context-relative signals. Low importance for raw `review_count` and `price` indicates these contribute primarily through derived features rather than directly.
 
---
 
## 6. Showcase
The trained model is integrated into a Flask web application with two routes:
 
### Homepage (`GET /`)
On startup, `routes.py` calls `load_model_and_data()`, which generates the 800-restaurant dataset, engineers features, trains the Random Forest, and stores the trained model and scored DataFrame in module-level variables. The homepage renders:
- A search form with dropdowns for **City** (15 options + "All"), **Category** (15 options + "All"), and **Max Price** ($–$$$$).
- A **Model Summary** card showing cross-validation R², test R², MAE, and RMSE.
- A **Current Top Restaurants** table showing the 10 highest-scoring restaurants overall.

### Results Page (`POST /results`)
When the user submits the form, the app filters the scored DataFrame by the selected city, category, and maximum price, then returns the top 10 matches ranked by `pq_score`. The results page displays:
- A ranked table of up to 10 matching restaurants with columns: Rank, Name, City, Category, Stars, Reviews, Price (displayed as $ symbols), and PQ Score (formatted to 2 decimal places).
- A **Most Important Features** table showing the top 6 features and their importances.
- A **Model Metrics** card with all four evaluation metrics.
- A "Back to Search" link for iterative exploration.

### Example Interaction
A user visiting the app selects **Philadelphia**, **Italian**, and **$$** (max price 2). The app instantly returns the top 10 Philadelphia Italian restaurants with PQ scores ranked from highest to lowest, enabling the user to identify the best value Italian dining in the city without manually comparing dozens of listings.
 
---
 
## 7. Discussion and Conclusions

### Summary of Findings
The project demonstrates that a multi-factor, price-adjusted ranking score can be engineered and learned by a machine learning model and delivered through a web interface. The Random Forest achieves near-perfect reconstruction of the target formula (R² > 0.98), validating the feature engineering design. The most predictive features — raw stars and category-relative stars — confirm that rating quality is the dominant driver of value perception, while price adjustment and city context provide meaningful secondary signals.
 
### Project Limitations and Issues:
- **Synthetic data dependency:** The current implementation uses procedurally generated data. Real restaurant data from the Yelp Fusion API or Google Places API would introduce genuine noise, geographic clustering, and temporal variation that synthetic data cannot replicate. The model should be retrained on real data before any production deployment.
- **Target leakage risk:** Because the PQ score is computed from the same columns used as model inputs, the model's task is function approximation rather than generalization to unobserved outcomes. On real data, where the "true" quality of a restaurant is a latent variable not fully captured by any formula, the model's task would be substantially harder.
- **Static model:** The model is trained once at application startup. In a production setting, models should be retrained periodically as new reviews arrive to reflect current restaurant quality.
- **No user personalization:** Users receive the same ranked output for a given filter combination. Incorporating user preference signals (cuisine history, budget patterns, location) would allow personalized recommendations.
 
### Application of Course Learning
This project integrates multiple course concepts end-to-end: supervised regression (Random Forest), feature engineering (log transforms, relative rankings, ratio features), model evaluation (train/test split, cross-validation, R²/MAE/RMSE), and software deployment (Flask routing, Jinja2 templating, modular Python package structure). The transition from a Jupyter notebook exploratory workflow to a production-ready Flask application reflects the software engineering dimension of applied machine learning.
 
---
 
## 8. AI Usage
AI tools (specifically large language model assistants) were used throughout the project in the following ways:
- **Feature engineering ideation:** AI was consulted during the feature design phase to brainstorm candidate features for the price-vs-quality domain. Suggestions such as `stars_vs_cat_mean` and `relative_buzz` emerged from this dialogue and were subsequently implemented and validated by the team.
- **Extent of involvement:** AI served as a development accelerator and sounding board. The project architecture decisions, feature formula design, hyperparameter choices, and final code review were performed by the human team. No AI-generated code was accepted without review and testing.

## Conclusion
The project delivers a complete, runnable ML application with clear separation of concerns: data generation (`src/data_generation.py`), model training and evaluation (`src/train.py`), model loading for the web layer (`src/model_loader.py`), Flask routing (`app/routes.py`), and HTML templates. The codebase includes module-level docstrings and inline comments throughout. The Flask app loads the model at startup, ensuring zero per-request training latency.
 
For the final submission, the primary recommended improvement is replacing synthetic data with a real Pennsylvania restaurant dataset (e.g., from the Yelp Open Dataset or Yelp Fusion API), which would make the model's generalization performance the central evaluation story rather than function approximation fidelity.

## Contributions
- Michael: Creating the beta and final main code. Writing the report.
- Eric: Creating the UI, the github page, the sample inputs, and the MD's. Reorganizing the main code.
