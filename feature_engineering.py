from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


# Mapping des fuseaux horaires IANA par pays (tel que dans ton notebook)
TZ_MAP: Dict[str, str] = {
    "FR": "Europe/Paris",
    "US": "America/New_York",
    "TR": "Europe/Istanbul",
    "PL": "Europe/Warsaw",
    "ES": "Europe/Madrid",
    "IT": "Europe/Rome",
    "RO": "Europe/Bucharest",
    "GB": "Europe/London",
    "NL": "Europe/Amsterdam",
    "DE": "Europe/Berlin",
}


@dataclass(frozen=True)
class FEConfig:
    """
    Configuration légère du Feature Engineering.
    Tu peux ajuster sans toucher au code (et sans 'en faire trop').
    """
    new_account_days: int = 30
    rolling_windows: tuple = ("24h", "7d", "30d")
    categorical_cols: tuple = ("country", "bin_country", "channel", "merchant_category")
    drop_cols: tuple = ("transaction_id",)


def _ensure_datetime_utc(df: pd.DataFrame, col: str = "transaction_time") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Colonne datetime '{col}' introuvable.")
    # utc=True gère les timestamps finissant par 'Z'
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def _add_local_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit transaction_time UTC -> local selon le pays, puis extrait hour/day.
    Note: on fait une version 'simple' (apply) comme ton notebook.
    """
    out = df.copy()

    def get_local_feature(row, feature_type: str):
        country_code = row.get("country", None)
        tz_name = TZ_MAP.get(country_code, "UTC")
        dt = row["transaction_time"]
        if pd.isna(dt):
            return np.nan
        local_dt = dt.tz_convert(tz_name)
        if feature_type == "hour":
            return local_dt.hour
        if feature_type == "dayofweek":
            return local_dt.dayofweek
        if feature_type == "day":
            return local_dt.day
        return np.nan

    # Si country manque, on reste en UTC via TZ_MAP.get(...,"UTC")
    out["hour_local"] = out.apply(lambda r: get_local_feature(r, "hour"), axis=1)
    out["dayofweek_local"] = out.apply(lambda r: get_local_feature(r, "dayofweek"), axis=1)
    out["day_local"] = out.apply(lambda r: get_local_feature(r, "day"), axis=1)
    return out


def _rolling_count_by_user(df: pd.DataFrame, window: str) -> pd.Series:
    """
    Rolling count des transactions passées (exclut la transaction actuelle).
    On utilise closed='left' pour exclure la transaction courante.
    """
    # nécessite transaction_time trié + timezone ok
    g = df.set_index("transaction_time")["amount"]
    return g.rolling(window, closed="left").count().values


def build_features(df: pd.DataFrame, config: Optional[FEConfig] = None) -> pd.DataFrame:
    """
    Feature engineering "prêt scripts" (train/predict).
    Important:
    - Ne dépend PAS de la target is_fraud (pas de fuite, et score possible en prod).
    - Tri par user_id + transaction_time avant features temporelles.
    """
    cfg = config or FEConfig()

    out = df.copy()

    # 1) Datetime UTC
    out = _ensure_datetime_utc(out, "transaction_time")

    # 2) Local time features
    out = _add_local_time_features(out)

    # 3) Tri obligatoire avant features temporelles
    if "user_id" in out.columns:
        out = out.sort_values(["user_id", "transaction_time"]).reset_index(drop=True)
    else:
        # fallback: tri uniquement temporel
        out = out.sort_values(["transaction_time"]).reset_index(drop=True)

    # 4) Features comportementales simples
    if "hour_local" in out.columns:
        out["is_night"] = ((out["hour_local"] >= 22) | (out["hour_local"] <= 5)).astype(int)

    if "user_id" in out.columns and "amount" in out.columns:
        out["avg_amount_user_past"] = (
            out.groupby("user_id")["amount"]
               .expanding()
               .mean()
               .shift(1)
               .reset_index(level=0, drop=True)
        )
        out["amount_diff_user_avg"] = out["amount"] - out["avg_amount_user_past"]

        out["amount_delta_prev"] = out.groupby("user_id")["amount"].diff().fillna(0)

    if "account_age_days" in out.columns:
        out["is_new_account"] = (out["account_age_days"] < cfg.new_account_days).astype(int)

    # Score sécurité (avs/cvv)
    if "avs_match" in out.columns and "cvv_result" in out.columns:
        out["security_mismatch_score"] = ((out["avs_match"] == 0).astype(int) + (out["cvv_result"] == 0).astype(int))

    # Mismatch pays / BIN
    if "country" in out.columns and "bin_country" in out.columns:
        out["country_bin_mismatch"] = (out["country"] != out["bin_country"]).astype(int)

    # Distance / amount
    if "shipping_distance_km" in out.columns and "amount" in out.columns:
        out["distance_amount_ratio"] = out["shipping_distance_km"] / (out["amount"] + 1)

    # Changement de canal
    if "user_id" in out.columns and "channel" in out.columns:
        out["channel_changed"] = (out["channel"] != out.groupby("user_id")["channel"].shift()).astype(int)

    # Temps depuis la dernière transaction
    if "user_id" in out.columns and "transaction_time" in out.columns:
        out["time_since_last"] = (
            out.groupby("user_id")["transaction_time"]
               .diff()
               .dt.total_seconds()
               .fillna(0)
        )

    # 5) Rolling windows (nombre de tx passées)
    # Pour rester robuste, on calcule uniquement si user_id + amount existent.
    if "user_id" in out.columns and "amount" in out.columns and "transaction_time" in out.columns:
        # On s'assure que transaction_time est bien datetime (sans perdre le tz)
        out["transaction_time"] = pd.to_datetime(out["transaction_time"], utc=True, errors="coerce")

        for win in cfg.rolling_windows:
            col_name = f"tx_last_{win}"
            out[col_name] = (
                out.groupby("user_id", group_keys=False)[["transaction_time", "amount"]]
                   .apply(lambda g: pd.Series(_rolling_count_by_user(g, win), index=g.index))
            )
        # NaN -> 0
        roll_cols = [f"tx_last_{w}" for w in cfg.rolling_windows]
        out[roll_cols] = out[roll_cols].fillna(0)

    # 6) Nettoyage colonnes non-modèle
    for c in cfg.drop_cols:
        if c in out.columns:
            out = out.drop(columns=[c])

    # On évite d'inclure une colonne datetime brute dans le modèle
    # (si tu veux, tu peux garder hour/dayofweek/day)
    if "transaction_time" in out.columns:
        out = out.drop(columns=["transaction_time"])

    # 7) One-hot encoding des catégories principales
    cat_cols = [c for c in cfg.categorical_cols if c in out.columns]
    if cat_cols:
        out = pd.get_dummies(out, columns=cat_cols, drop_first=False)

    # 8) Sécurités numériques
    out = out.replace([np.inf, -np.inf], 0)

    # Les NaN restants: on met 0 (simple, efficace pour XGBoost)
    out = out.fillna(0)

    return out
