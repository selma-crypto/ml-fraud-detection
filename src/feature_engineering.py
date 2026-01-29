from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Mapping IANA par pays (si un code pays n'est pas listé -> UTC)
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
    # Seuil "nouveau compte"
    new_account_days: int = 30

    # Fenêtres rolling (nb de transactions passées)
    rolling_windows: tuple = ("24h", "7d", "30d")

    # Colonnes catégorielles à one-hot
    categorical_cols: tuple = ("country", "bin_country", "channel", "merchant_category")

    # Colonnes à drop (identifiants)
    drop_cols: tuple = ("transaction_id",)


def _ensure_datetime_utc(df: pd.DataFrame, col: str = "transaction_time") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Colonne datetime '{col}' introuvable.")
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


def _add_local_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit transaction_time UTC -> local selon le pays, puis extrait hour/day.
    Version simple (apply) : OK sur ~300k lignes, mais évite de l'utiliser sur des dizaines de millions.
    """
    out = df.copy()

    def get_local_feature(row, feature_type: str):
        tz_name = TZ_MAP.get(row.get("country", None), "UTC")
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

    out["hour_local"] = out.apply(lambda r: get_local_feature(r, "hour"), axis=1)
    out["dayofweek_local"] = out.apply(lambda r: get_local_feature(r, "dayofweek"), axis=1)
    out["day_local"] = out.apply(lambda r: get_local_feature(r, "day"), axis=1)
    return out


def _rolling_tx_count(g: pd.DataFrame, window: str) -> pd.Series:
    """
    Rolling count de transactions passées (exclut la transaction courante).
    g doit être trié et indexé par transaction_time.
    """
    s = g.set_index("transaction_time")["amount"]
    return s.rolling(window, closed="left").count()


def build_features(df: pd.DataFrame, config: Optional[FEConfig] = None) -> pd.DataFrame:
    """
    Feature engineering "plug & play" pour ce dataset (transactions.csv).
    - Ne dépend PAS de la cible is_fraud.
    - Fonctionne aussi bien pour train (df sans target) que pour predict (df sans target).
    """
    cfg = config or FEConfig()
    out = df.copy()

    # 1) Datetime
    out = _ensure_datetime_utc(out, "transaction_time")

    # 2) Temps local (hour/day)
    out = _add_local_time_features(out)

    # 3) Tri (obligatoire)
    if "user_id" in out.columns:
        out = out.sort_values(["user_id", "transaction_time"]).reset_index(drop=True)
    else:
        out = out.sort_values(["transaction_time"]).reset_index(drop=True)

    # 4) Features "métier" simples
    out["is_night"] = ((out["hour_local"] >= 22) | (out["hour_local"] <= 5)).astype(int)

    # Historique montant user (past only)
    out["avg_amount_user_past"] = (
        out.groupby("user_id")["amount"]
           .expanding()
           .mean()
           .shift(1)
           .reset_index(level=0, drop=True)
    )
    out["amount_diff_user_avg_past"] = out["amount"] - out["avg_amount_user_past"]

    # Delta montant vs transaction précédente
    out["amount_delta_prev"] = out.groupby("user_id")["amount"].diff().fillna(0)

    # Nouveau compte
    if "account_age_days" in out.columns:
        out["is_new_account"] = (out["account_age_days"] < cfg.new_account_days).astype(int)

    # Score sécurité (AVS/CVV)
    out["security_mismatch_score"] = ((out["avs_match"] == 0).astype(int) + (out["cvv_result"] == 0).astype(int))

    # BIN mismatch
    out["country_bin_mismatch"] = (out["country"] != out["bin_country"]).astype(int)

    # Distance / amount
    out["distance_amount_ratio"] = out["shipping_distance_km"] / (out["amount"] + 1)

    # Changement de canal
    out["channel_changed"] = (out["channel"] != out.groupby("user_id")["channel"].shift()).astype(int)

    # Temps depuis dernière transaction (sec)
    out["time_since_last"] = (
        out.groupby("user_id")["transaction_time"]
           .diff()
           .dt.total_seconds()
           .fillna(0)
    )

    # 5) Rolling counts par fenêtre
    for win in cfg.rolling_windows:
        col_name = f"tx_last_{win}"
        out[col_name] = (
            out.groupby("user_id", group_keys=False)[["transaction_time", "amount"]]
               .apply(lambda g: _rolling_tx_count(g, win).reindex(g.index).values)
        )
    roll_cols = [f"tx_last_{w}" for w in cfg.rolling_windows]
    out[roll_cols] = out[roll_cols].fillna(0)

    # 6) Drop colonnes inutiles
    out = out.drop(columns=[c for c in cfg.drop_cols if c in out.columns], errors="ignore")

    # On retire la datetime brute (on garde hour/day)
    out = out.drop(columns=["transaction_time"], errors="ignore")

    # 7) One-hot
    cat_cols = [c for c in cfg.categorical_cols if c in out.columns]
    out = pd.get_dummies(out, columns=cat_cols, drop_first=False)

    # 8) Hygiène
    out = out.replace([np.inf, -np.inf], 0).fillna(0)

    return out
