# Multi-Class Exoplanet Classification Pipeline

## Overview
This project implements a comprehensive machine learning pipeline for **classifying exoplanets** based on Kepler mission data. The pipeline includes **data cleaning, feature engineering, feature selection, polynomial feature interactions, and model training** using multiple state-of-the-art algorithms. The final model can predict the disposition of exoplanets (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`) with high accuracy(95%).

## Features
- Handles missing and infinite values
- Extensive **feature engineering** based on orbital mechanics, stellar properties, transit properties, insolation, and habitability
- Feature selection using **ANOVA F-test** and **Mutual Information**
- Polynomial feature interactions for top features
- Trains multiple models:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
  - AdaBoost
  - Baseline Random Forest
- Evaluates models using **accuracy**, **classification report**, and **confusion matrix**
- Saves trained models and preprocessing objects for deployment

## Requirements
- Python 3.8+
- Libraries:
  pandas, numpy, scikit-learn, matplotlib, seaborn,
  xgboost, lightgbm, catboost, pickle
