"""
Funciones de evaluación de modelos y comparación de versiones para BistroTech.
"""
import logging
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def evaluate_modelo_a(model, X_test, y_test) -> dict:
    """
    Evalúa el Modelo A (afinidad de mozo).

    Args:
        model: XGBRegressor entrenado.
        X_test: Features de evaluación (debe incluir id_mozo).
        y_test: propina_rate reales.

    Returns:
        dict con claves 'rmse', 'mae', 'pearson'.
    """
    preds = model.predict(X_test)
    residuals = y_test - preds
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    pearson, _ = stats.pearsonr(y_test, preds)
    metrics = {"rmse": round(rmse, 6), "mae": round(mae, 6), "pearson": round(float(pearson), 6)}
    logger.info("Modelo A — RMSE: %.4f | MAE: %.4f | Pearson: %.4f", rmse, mae, pearson)
    return metrics


def evaluate_modelo_b(model, X_test, y_test, label_encoder=None, k: int = 3) -> dict:
    """
    Evalúa un Modelo B (recomendación de plato).

    Args:
        model: XGBClassifier entrenado.
        X_test: Features de evaluación.
        y_test: IDs de plato reales (codificados o crudos).
        label_encoder: LabelEncoder usado durante el entrenamiento.
        k: Número de top-k predicciones para Hit Rate.

    Returns:
        dict con claves 'hit_rate_k', 'f1_macro'.
    """
    proba = model.predict_proba(X_test)
    top_k_indices = np.argsort(proba, axis=1)[:, -k:]

    hits = 0
    y_pred = []
    for i, true_label in enumerate(y_test):
        top_k_labels = top_k_indices[i]
        y_pred.append(int(np.argmax(proba[i])))
        if true_label in top_k_labels:
            hits += 1

    hit_rate = hits / len(y_test)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    metrics = {"hit_rate_k": round(hit_rate, 6), "f1_macro": round(float(f1), 6)}
    logger.info("Modelo B — Hit Rate@%d: %.4f | F1 macro: %.4f", k, hit_rate, f1)
    return metrics


def compare_versions(
    metrics_new: dict,
    metrics_current: dict,
    improvement_threshold: float = 0.05,
) -> bool:
    """
    Compara dos versiones de modelo y decide si la nueva es mejor.

    Para métricas donde menor es mejor (rmse, mae): mejora = (current - new) / current > threshold.
    Para métricas donde mayor es mejor (pearson, hit_rate_k, f1_macro): mejora = (new - current) / current > threshold.

    Args:
        metrics_new: Métricas del modelo candidato.
        metrics_current: Métricas del modelo en producción.
        improvement_threshold: Fracción mínima de mejora (default 5%).

    Returns:
        True si el modelo nuevo mejora en TODAS las métricas por encima del umbral.
    """
    lower_is_better = {"rmse", "mae"}
    all_improved = True

    for metric, new_val in metrics_new.items():
        if metric not in metrics_current:
            logger.warning("Métrica '%s' ausente en modelo actual; se omite.", metric)
            continue
        current_val = metrics_current[metric]
        if current_val == 0:
            logger.warning("Métrica '%s' es 0 en producción; se omite comparación.", metric)
            continue

        if metric in lower_is_better:
            improvement = (current_val - new_val) / abs(current_val)
        else:
            improvement = (new_val - current_val) / abs(current_val)

        improved = improvement > improvement_threshold
        direction = "↓" if metric in lower_is_better else "↑"
        status = "✅" if improved else "❌"
        logger.info(
            "%s %s: nuevo=%.4f actual=%.4f mejora=%.2f%% (umbral=%.0f%%) %s",
            status,
            metric,
            new_val,
            current_val,
            improvement * 100,
            improvement_threshold * 100,
            direction,
        )
        if not improved:
            all_improved = False

    logger.info("Decisión de deploy: %s", "APROBAR" if all_improved else "RECHAZAR")
    return all_improved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metrics_current = {"rmse": 0.05, "mae": 0.04, "pearson": 0.70}
    metrics_new = {"rmse": 0.046, "mae": 0.037, "pearson": 0.74}
    result = compare_versions(metrics_new, metrics_current)
    print(f"Deploy aprobado: {result}")
