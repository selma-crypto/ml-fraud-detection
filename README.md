## Fraud Detection System — Machine Learning Engineer Project
End-to-end fraud detection system built during the Jedha Full-Stack Data Scientist Bootcamp. Système de détection de fraude de bout en bout développé dans le cadre du bootcamp Jedha – Full Stack Data Scientist.

###Project Overview
This project aims to detect fraudulent online payment transactions using Machine Learning. It addresses a real-world constraint: highly imbalanced data and the need to minimize false positives while maximizing fraud detection.

The final solution is based on an optimized XGBoost model and deployed as an interactive web application.

Ce projet vise à détecter les transactions frauduleuses en ligne à l’aide du Machine Learning, dans un contexte réaliste de données fortement déséquilibrées.

La solution finale repose sur un modèle XGBoost optimisé et déployé sous forme d’application interactive.

Jedha Bootcamp — Skills Covered
This project is part of the Jedha Full-Stack Data Scientist Bootcamp, covering:

Python for Data Science
Data Cleaning & Exploratory Data Analysis (EDA)
Feature Engineering
Supervised Machine Learning
Handling imbalanced datasets
Model evaluation (Recall, Precision, F1, PR-AUC)
Hyperparameter optimization (Optuna)
Model interpretability (SHAP)
Model deployment (Docker, Hugging Face Spaces)
Building ML-powered web applications
Ce projet s’inscrit dans la formation Jedha Full Stack Data Scientist, incluant :

Python pour la Data Science
Analyse exploratoire des données (EDA)
Feature Engineering
Machine Learning supervisé
Gestion des données déséquilibrées
Évaluation des modèles (Recall, Precision, F1, PR-AUC)
Optimisation d’hyperparamètres (Optuna)
Interprétabilité des modèles (SHAP)
Déploiement de modèles (Docker, Hugging Face)
Développement d’applications ML
Dataset & Business Problem
Dataset: E-commerce transactions (Kaggle)
~300,000 transactions
Fraud rate ≈ 2%
Main challenge: Detect fraud efficiently while avoiding excessive false positives that would block legitimate customers.

Machine Learning Pipeline
Data preprocessing & cleaning
Exploratory Data Analysis
Feature Engineering (temporal, behavioral, security, geo features)
Baseline models (Logistic Regression, Random Forest)
Model selection: XGBoost
Hyperparameter tuning with Optuna
Threshold optimization
Model explainability with SHAP
Deployment
Model Performance
The final XGBoost model provides the best trade-off between fraud detection and false positives.

Key metrics:

High recall on fraud class
Stable performance between train and test
Optimized decision threshold
Application & Deployment
The model is deployed as an interactive web application.

Features:

Single transaction analysis (JSON)
Batch analysis (CSV)
Fraud probability scoring
Feature importance explanation
Live demo: https://patrickcharda-detectionfraude.hf.space

Repository Structure
app.py # Web application
Dockerfile # Containerized deployment
requirements.txt # Dependencies
assets/ # Test files, notebooks, models
README.md
Future Improvements
Model monitoring in production
Drift detection
Dynamic thresholding
Integration of unsupervised anomaly detection
MLOps pipeline (CI/CD, retraining)
Author
Machine Learning Engineer Junior
Jedha Full-Stack Data Scientist Bootcamp
