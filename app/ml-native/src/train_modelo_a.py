"""
Entrenamiento del Modelo A — Afinidad de mozo (regresión de propina_rate).

El modelo predice el propina_rate esperado dado el perfil del comensal
y el mozo asignado. En inferencia se itera sobre los 8 mozos posibles
para encontrar el que maximiza el propina_rate esperado.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from src.feature_engineering import load_processed, build_features
from src.evaluate import evaluate_modelo_a

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "models/modelo_a_mozo.joblib"
FEATURE_NAMES_PATH = "models/feature_names_a.json"


def aggregate_mesa_scores(per_comensal_scores: list[dict]) -> list[dict]:
    """
    Promedia los propina_rate predichos por mozo a través de todos los comensales de la mesa.

    Args:
        per_comensal_scores: lista de dicts {id_mozo: propina_rate_predicho}
                             un dict por comensal.

    Returns:
        Lista ordenada de {'id_mozo': int, 'propina_rate_esperado': float, 'rank': int}
    """
    all_mozos = list(per_comensal_scores[0].keys())
    avg_scores = {
        mozo: float(np.mean([scores[mozo] for scores in per_comensal_scores]))
        for mozo in all_mozos
    }
    ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id_mozo": int(mozo), "propina_rate_esperado": round(score, 4), "rank": i + 1}
        for i, (mozo, score) in enumerate(ranked)
    ]


def train(raw_csv: str = "data/raw/reservas.csv") -> dict:
    """
    Entrena el Modelo A desde cero.

    Flujo:
    1. Carga CSV crudo y aplica feature engineering.
    2. Agrega id_mozo a X (feature clave del modelo).
    3. Split 80/20 estratificado por id_mozo.
    4. Entrena XGBRegressor con early stopping.
    5. Evalúa y persiste modelo + feature names.

    Returns:
        dict con métricas: rmse, mae, pearson.
    """
    os.makedirs("models", exist_ok=True)

    logger.info("Cargando dataset crudo desde %s", raw_csv)
    df_raw = pd.read_csv(raw_csv)

    X_base, targets = build_features(df_raw)

    idx = targets["propina_rate"].index
    X_a = X_base.loc[idx].copy()
    X_a["id_mozo"] = df_raw.loc[idx, "id_mozo"].values
    y = targets["propina_rate"]

    feature_names = list(X_a.columns)
    logger.info("Features Modelo A (%d): %s", len(feature_names), feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_a, y, test_size=0.20, random_state=42
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        early_stopping_rounds=20,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Entrenando XGBRegressor — train=%d, test=%d", len(X_train), len(X_test))
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    metrics = evaluate_modelo_a(model, X_test, y_test.values)

    joblib.dump(model, MODEL_PATH)
    logger.info("Modelo guardado en %s", MODEL_PATH)

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info("Feature names guardados en %s", FEATURE_NAMES_PATH)

    print(f"\n{'='*55}")
    print(f"Modelo A entrenado — {model.best_iteration} árboles efectivos")
    print(f"  RMSE     : {metrics['rmse']:.4f}")
    print(f"  MAE      : {metrics['mae']:.4f}")
    print(f"  Pearson  : {metrics['pearson']:.4f}")
    print(f"  Modelo   : {MODEL_PATH}")
    print(f"{'='*55}\n")

    return metrics


if __name__ == "__main__":
    metrics = train()
