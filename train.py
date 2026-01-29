from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, classification_report
from xgboost import XGBClassifier

from feature_engineering import build_features, FEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an XGBoost fraud model (plug & play).")
    p.add_argument("--data-path", required=True, help="Chemin du CSV d'entraînement (ex: data/transactions.csv)")
    p.add_argument("--target", default="is_fraud", help="Nom de la colonne cible (default: is_fraud)")
    p.add_argument("--model-out", default="models/fraud_xgb_model.pkl", help="Chemin de sortie de l'artefact .pkl")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5, help="Seuil stocké dans l'artefact (default: 0.5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if args.target not in df.columns:
        raise ValueError(f"Colonne cible '{args.target}' introuvable dans {data_path.name}.")

    y = df[args.target].astype(int)
    X_raw = df.drop(columns=[args.target])

    # Feature engineering (sans fuite)
    X = build_features(X_raw, config=FEConfig())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Déséquilibre : scale_pos_weight
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=450,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=args.random_state
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print(classification_report(y_test, (y_prob >= args.threshold).astype(int), digits=4))

    artifact = {
        "model": model,
        "feature_columns": list(X.columns),
        "target": args.target,
        "default_threshold": float(args.threshold),
        "fe_config": FEConfig().__dict__,
        "dataset_columns": list(df.columns),
    }

    joblib.dump(artifact, model_out)
    print(f"Artefact sauvegardé -> {model_out.as_posix()}")


if __name__ == "__main__":
    main()
