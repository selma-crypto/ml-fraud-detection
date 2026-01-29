from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import build_features, FEConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score transactions with a trained fraud model (plug & play).")
    p.add_argument("--model-path", default="models/fraud_xgb_model.pkl")
    p.add_argument("--input-path", required=True, help="CSV à scorer (sans is_fraud)")
    p.add_argument("--output-path", default="predictions.csv")
    p.add_argument("--threshold", type=float, default=None, help="Seuil (sinon utilise celui de l'artefact)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    artifact = joblib.load(args.model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    default_threshold = float(artifact.get("default_threshold", 0.5))
    threshold = default_threshold if args.threshold is None else float(args.threshold)

    df = pd.read_csv(args.input_path)

    # Si jamais l'utilisateur passe un CSV qui contient encore la target, on la retire
    if "is_fraud" in df.columns:
        df = df.drop(columns=["is_fraud"])

    X = build_features(df, config=FEConfig())

    # Plug & play : on aligne les colonnes sur l'entraînement
    # - Colonnes manquantes -> ajoutées à 0
    # - Colonnes en trop -> ignorées
    X = X.reindex(columns=feature_columns, fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_proba"] = proba
    out["fraud_pred"] = pred

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"OK -> {output_path.as_posix()} (threshold={threshold})")


if __name__ == "__main__":
    main()
