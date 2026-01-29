## Fraud Detection System — Machine Learning Engineer Project

### How to Run

1) Clone the repository
```bash
git clone https://github.com/selma-crypto/ml-fraud-detection.git
cd ml-fraud-detection


End-to-end fraud detection system built during a Full-Stack Data Scientist training project.
Système de détection de fraude de bout en bout développé dans le cadre d’un projet de formation Data Science.

### Project Overview

This project aims to detect fraudulent online payment transactions using Machine Learning. It addresses a real-world constraint: highly imbalanced data and the need to minimize false positives while maximizing fraud detection.

The final solution is based on an optimized XGBoost model and deployed as an interactive web application.

Ce projet vise à détecter les transactions frauduleuses en ligne à l’aide du Machine Learning, dans un contexte réaliste de données fortement déséquilibrées.

La solution finale repose sur un modèle XGBoost optimisé et est déployée sous forme d’application web interactive.

### Project Highlights

- End-to-end Machine Learning pipeline
- Fraud detection on imbalanced financial data
- Model optimization using Optuna
- Evaluation with Recall, Precision, F1-score and PR-AUC
- Model interpretability using SHAP
- Deployment with Docker and web interface

### Skills Covered

This project demonstrates key Machine Learning and Data Science skills:

- Python for Data Science
- Data Cleaning & Exploratory Data Analysis (EDA)
- Feature Engineering
- Supervised Machine Learning
- Handling imbalanced datasets
- Model evaluation (Recall, Precision, F1, PR-AUC)
- Hyperparameter optimization (Optuna)
- Model interpretability (SHAP)
- Model deployment (Docker)
- Building ML-powered web applications

Ce projet met en avant les compétences suivantes :

- Python pour la Data Science
- Nettoyage et analyse exploratoire des données (EDA)
- Feature Engineering
- Machine Learning supervisé
- Gestion des données déséquilibrées
- Évaluation des modèles (Recall, Precision, F1, PR-AUC)
- Optimisation d’hyperparamètres (Optuna)
- Interprétabilité des modèles (SHAP)
- Déploiement de modèles (Docker)
- Développement d’applications ML

### Dataset & Business Problem

Dataset: E-commerce transactions dataset (Kaggle)
Volume: ~300,000 transactions
Fraud rate: ≈ 2%

Main challenge: Detect fraudulent transactions efficiently while avoiding excessive false positives that would block legitimate customers.

Jeu de données de transactions e-commerce issu de Kaggle.
Environ 300 000 transactions, avec un taux de fraude proche de 2 %.

Défi principal : détecter efficacement les fraudes tout en évitant trop de faux positifs qui bloqueraient des clients légitimes.

### Machine Learning Pipeline

- Data preprocessing & cleaning
- Exploratory Data Analysis
- Feature Engineering (behavioral, temporal, transaction patterns)
- Baseline models (Logistic Regression, Random Forest)
- Model selection: XGBoost
- Hyperparameter tuning with Optuna
- Threshold optimization
- Model explainability with SHAP

### Model Performance

The final XGBoost model provides the best trade-off between fraud detection and false positives.

Key metrics include:

- High recall on fraud class
- Stable performance between training and test sets
- Optimized decision threshold
- Le modèle XGBoost final offre le meilleur compromis entre détection des fraudes et limitation des faux positifs.

Principaux résultats :

- Recall élevé sur la classe fraude
- Performances stables entre train et test
- Seuil de décision optimisé

### Application & Deployment

The model is deployed as an interactive web application.

Main features:

- Single transaction analysis (JSON input)
- Batch analysis (CSV file)
- Fraud probability scoring
- Feature importance explanation

L’application permet :

- L’analyse d’une transaction individuelle
- L’analyse par lot de transactions
- L’affichage d’un score de probabilité de fraude
- L’explication des prédictions du modèle

### Repository Structure

app/
Web application and deployment files

notebooks/
EDA, modeling and experimentation

src/
Training and prediction scripts

models/
Saved trained models

requirements.txt
Project dependencies

README.md
Project documentation

### How to Run
1. git clone …
2. pip install -r requirements.txt
3. python src/train.py
4. python src/predict.py
5. streamlit run app/app.py

### Future Improvements

- Model monitoring in production
- Data drift detection
- Dynamic thresholding
- Integration of unsupervised anomaly detection
- Full MLOps pipeline (CI/CD, automated retraining)

### Author

Machine Learning Engineer / Data Science Portfolio Project
Fraud Detection System
