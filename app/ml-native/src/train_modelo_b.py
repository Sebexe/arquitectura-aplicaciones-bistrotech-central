"""
Entrenamiento de los Modelos B — Recomendación de menú (4 modelos independientes).

Modelos: entrada (1-8), principal (9-20), postre (21-25), bebida (26-30).
Cada modelo es un XGBClassifier multiclase con softmax.
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import load_processed, build_features
from src.evaluate import evaluate_modelo_b

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CURSOS = ["entrada", "principal", "postre", "bebida"]


def train_curso(
    X_full: pd.DataFrame,
    targets: dict,
    curso: str,
    random_state: int = 42,
) -> dict:
    """
    Entrena un modelo B para un curso específico.

    Args:
        X_full: Features completas (sin leakage).
        targets: dict de targets con índices filtrados.
        curso: 'entrada' | 'principal' | 'postre' | 'bebida'
        random_state: semilla de reproducibilidad.

    Returns:
        dict con métricas hit_rate_k y f1_macro.
    """
    os.makedirs("models", exist_ok=True)

    target_col = f"id_{curso}"
    y_raw = targets[target_col]
    X_curso = X_full.loc[y_raw.index].copy()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw.values.astype(int))
    n_classes = len(le.classes_)

    logger.info(
        "Curso '%s': %d registros, %d clases únicas",
        curso,
        len(X_curso),
        n_classes,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_curso, y_encoded, test_size=0.20, random_state=random_state
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        early_stopping_rounds=20,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    metrics = evaluate_modelo_b(model, X_test, y_test, label_encoder=le, k=3)

    model_path = f"models/modelo_b_{curso}.joblib"
    le_path = f"models/label_encoder_{curso}.joblib"
    joblib.dump(model, model_path)
    joblib.dump(le, le_path)
    logger.info("Guardado: %s y %s", model_path, le_path)

    print(f"  [{curso}] Hit Rate@3: {metrics['hit_rate_k']:.4f} | F1 macro: {metrics['f1_macro']:.4f}")
    return metrics


def train_all(raw_csv: str = "data/raw/reservas.csv") -> dict:
    """
    Entrena los 4 modelos B (entrada, principal, postre, bebida).

    Returns:
        dict {curso: metrics_dict}
    """
    logger.info("Cargando dataset crudo desde %s", raw_csv)
    df_raw = pd.read_csv(raw_csv)
    X, targets = build_features(df_raw)

    all_metrics = {}
    hit_rates = []

    print(f"\n{'='*55}")
    print("Entrenando Modelos B (recomendación de menú)")
    print(f"{'='*55}")

    for curso in CURSOS:
        metrics = train_curso(X, targets, curso)
        all_metrics[curso] = metrics
        hit_rates.append(metrics["hit_rate_k"])

    avg_hit_rate = float(np.mean(hit_rates))
    print(f"\nHit Rate promedio: {avg_hit_rate:.2%}")
    print(f"{'='*55}\n")

    return all_metrics


if __name__ == "__main__":
    all_metrics = train_all()
