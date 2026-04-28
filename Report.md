# Best in PA Restaurants Project Report

## 1. Description of the Project

This project builds a web-based machine learning application that ranks Pennsylvania restaurants using a Price-vs-Quality score. The goal is to help users identify strong restaurant options based on ratings, review activity, price, and category-based comparisons.

## 2. Significance of the Project

Restaurant ratings alone do not always show the best value. A restaurant with slightly lower ratings but a much lower price may be a better option for some users. This application combines multiple factors into one ranking score so users can make a more informed decision.

## 3. Data Collection

The current version uses synthetic Pennsylvania restaurant data for development. The dataset includes restaurant name, city, category, star rating, review count, and price level. For final submission, this section should be updated with the real data source, collection method, dataset size, and metadata.

| Column | Description |
|---|---|
| name | Restaurant name |
| city | Pennsylvania city |
| category | Restaurant type |
| stars | Rating from 1 to 5 |
| review_count | Number of reviews |
| price | Price level from 1 to 4 |

Sample data:

| name | city | category | stars | review_count | price |
|---|---|---|---:|---:|---:|
| Philadelphia Italian Kitchen | Philadelphia | Italian | 4.7 | 1320 | 2 |
| Pittsburgh BBQ Grill | Pittsburgh | BBQ | 4.5 | 980 | 2 |
| Lancaster Breakfast Table | Lancaster | Breakfast | 4.6 | 430 | 1 |

## 4. Data Processing and Feature Engineering

The raw dataset is transformed into model-ready features. Review count is log-scaled to reduce the effect of extreme values. Price is clipped between 1 and 4. Additional features include stars per dollar, weighted stars, city rank, category comparison, and relative buzz within a city.

## 5. Model Development

The model uses a Random Forest Regressor. The input is a feature vector built from the restaurant information. The output is a predicted Price-vs-Quality score from 0 to 100. The model is evaluated using R², MAE, RMSE, and 3-fold cross-validation.

## 6. Showcase

The web app allows the user to choose a city, restaurant category, and maximum price level. After submitting the form, the app displays the top matching restaurants ranked by predicted Price-vs-Quality score. The results page also shows the model's most important features.

## 7. Discussion and Conclusions

The project shows how machine learning can be used to rank restaurants based on multiple factors instead of only star ratings. The main limitation is that the current version uses synthetic data, so the model should be retrained on real restaurant data before final use. Overall, the project connects data preprocessing, feature engineering, model training, evaluation, and web application development into one complete system.

## 8. AI Usage

AI tools were used to help structure the project and generate the web application layout. 
AI was used as a development assistant, but the project structure, testing, and final content were reviewed and adjusted by the team.
