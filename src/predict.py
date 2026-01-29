from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import build_features, FEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score transactions with trained fraud model.")
    p.add_argument("--model-path", default="models/fraud_xgb_model.pkl", help="Chemin vers l'artefact .pkl")
    p.add_argument("--input-path", required=True, help="CSV à scorer (ex: data/new_transactions.csv)")
    p.add_argument("--output-path", default="predictions.csv", help="Fichier de sortie")
    p.add_argument("--threshold", type=float, default=None, help="Seuil (sinon utilise celui stocké dans l'artefact)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    artifact = joblib.load(model_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    default_threshold = artifact.get("default_threshold", 0.5)

    threshold = default_threshold if args.threshold is None else float(args.threshold)

    df = pd.read_csv(args.input_path)

    X = build_features(df, config=FEConfig())

    # Alignement strict des colonnes (comme en entraînement)
    # Colonnes manquantes -> erreur explicite (plus safe)
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError(
            "Certaines colonnes attendues par le modèle manquent après feature engineering:\n"
            + "\n".join(missing[:50])
            + ("\n..." if len(missing) > 50 else "")
        )

    # Colonnes en trop -> on ignore
    X = X[feature_columns]

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_proba"] = proba
    out["fraud_pred"] = pred

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)

    print(f"Fichier généré: {args.output_path}")
    print(f"Seuil utilisé: {threshold}")


if __name__ == "__main__":
    main()
