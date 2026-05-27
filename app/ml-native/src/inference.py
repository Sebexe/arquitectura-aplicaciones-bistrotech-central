"""
Funciones de inferencia para BistroTech.

Expone predict(contexto_mesa) que devuelve el formato exacto de la API
definido en el README: mozos_recomendados + recomendaciones_por_comensal.
"""
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import get_inference_features, load_processed

logger = logging.getLogger(__name__)

CURSO_DISH_RANGE: dict[str, range] = {
    "entrada":    range(1, 9),
    "principal":  range(9, 21),
    "postre":     range(21, 26),
    "bebida":     range(26, 31),
}

MODEL_VERSION_FILE = "models/version.txt"
_models: dict | None = None


def load_models() -> dict:
    """
    Carga todos los modelos en memoria (patrón singleton).

    Returns:
        dict con claves:
          'modelo_a', 'feature_names_a',
          'modelo_b_entrada/principal/postre/bebida',
          'label_encoder_entrada/principal/postre/bebida',
          'version'
    """
    global _models
    if _models is not None:
        return _models

    models: dict = {}

    try:
        models["modelo_a"] = joblib.load("models/modelo_a_mozo.joblib")
        with open("models/feature_names_a.json") as f:
            models["feature_names_a"] = json.load(f)
        logger.info("Modelo A cargado.")
    except FileNotFoundError as e:
        logger.warning("Modelo A no encontrado: %s", e)

    for curso in ["entrada", "principal", "postre", "bebida"]:
        try:
            models[f"modelo_b_{curso}"] = joblib.load(f"models/modelo_b_{curso}.joblib")
            models[f"label_encoder_{curso}"] = joblib.load(
                f"models/label_encoder_{curso}.joblib"
            )
            logger.info("Modelo B '%s' cargado.", curso)
        except FileNotFoundError as e:
            logger.warning("Modelo B '%s' no encontrado: %s", curso, e)

    if os.path.exists(MODEL_VERSION_FILE):
        with open(MODEL_VERSION_FILE) as f:
            models["version"] = f.read().strip()
    else:
        models["version"] = "v1.0"

    _models = models
    return models


def predict_mozo(comensales_features: list[dict]) -> list[dict]:
    """
    Predice el propina_rate esperado por mozo para cada comensal y promedia por mesa.

    Para cada uno de los 8 mozos disponibles, construye el vector de features
    con id_mozo y predice propina_rate. Promedia los scores individuales y
    devuelve el ranking de mozos.

    Args:
        comensales_features: lista de dicts con las features de cada comensal
                             (formato de entrada de la API). Incluye contexto de mesa.

    Returns:
        Lista ordenada de {'id_mozo': int, 'propina_rate_esperado': float, 'rank': int}
    """
    models = load_models()
    if "modelo_a" not in models:
        logger.warning("Modelo A no disponible; retornando ranking aleatorio.")
        mozos = list(range(1, 9))
        np.random.shuffle(mozos)
        return [{"id_mozo": m, "propina_rate_esperado": 0.0, "rank": i + 1} for i, m in enumerate(mozos)]

    model_a = models["modelo_a"]
    feature_names_a = models["feature_names_a"]

    contexto_base = comensales_features[0]["_contexto_mesa"]
    X_base = get_inference_features(contexto_base)

    per_comensal_scores: list[dict] = []

    for i, _ in enumerate(contexto_base["comensales"]):
        x_row = X_base.iloc[[i]].copy()
        scores_by_mozo: dict = {}
        for mozo_id in range(1, 9):
            x_with_mozo = x_row.copy()
            x_with_mozo["id_mozo"] = mozo_id
            x_with_mozo = x_with_mozo.reindex(columns=feature_names_a, fill_value=0)
            score = float(model_a.predict(x_with_mozo)[0])
            score = max(0.0, min(1.0, score))
            scores_by_mozo[mozo_id] = score
        per_comensal_scores.append(scores_by_mozo)

    from src.train_modelo_a import aggregate_mesa_scores
    return aggregate_mesa_scores(per_comensal_scores)


def predict_menu(comensal_features: dict) -> dict:
    """
    Devuelve top 3 recomendaciones para cada curso (entrada/principal/postre/bebida).

    Args:
        comensal_features: fila de X (pd.Series o dict de features transformadas).

    Returns:
        dict {curso: [{'id_plato': int, 'score': float, 'rank': int}]}
    """
    models = load_models()
    result: dict = {}

    for curso, dish_range in CURSO_DISH_RANGE.items():
        key_model = f"modelo_b_{curso}"
        key_le = f"label_encoder_{curso}"

        if key_model not in models:
            result[curso] = []
            continue

        model_b = models[key_model]
        le = models[key_le]

        if isinstance(comensal_features, dict):
            X_row = pd.DataFrame([comensal_features])
        else:
            X_row = comensal_features.to_frame().T if isinstance(comensal_features, pd.Series) else comensal_features

        proba = model_b.predict_proba(X_row)[0]

        scored: list[tuple] = []
        for dish_id in dish_range:
            if dish_id in le.classes_:
                encoded_idx = int(np.where(le.classes_ == dish_id)[0][0])
                score = float(proba[encoded_idx])
                scored.append((dish_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top3 = scored[:3]

        result[curso] = [
            {"id_plato": int(did), "score": round(score, 4), "rank": rank + 1}
            for rank, (did, score) in enumerate(top3)
        ]

    return result


def predict(contexto_mesa: dict) -> dict:
    """
    Orquestador principal. Recibe el JSON de la API y devuelve la respuesta completa.

    Output exacto del README:
    {
      "id_mesa": int,
      "mozos_recomendados": [...],
      "recomendaciones_por_comensal": [...],
      "modelo_version": str,
      "latencia_ms": int
    }

    Args:
        contexto_mesa: dict con formato de input de la API (README).

    Returns:
        dict con el output completo de la API.
    """
    t0 = time.time()

    load_processed()

    comensales = contexto_mesa["comensales"]
    comensales_with_ctx = [{"_contexto_mesa": contexto_mesa} for _ in comensales]
    mozos_recomendados = predict_mozo(comensales_with_ctx)

    X_all = get_inference_features(contexto_mesa)
    recomendaciones: list[dict] = []

    for i, comensal in enumerate(comensales):
        x_row = X_all.iloc[[i]]
        menu = predict_menu(x_row)
        recomendaciones.append(
            {
                "id_persona_en_mesa": comensal["id_persona_en_mesa"],
                "entrada": menu.get("entrada", []),
                "principal": menu.get("principal", []),
                "postre": menu.get("postre", []),
                "bebida": menu.get("bebida", []),
            }
        )

    models = load_models()
    latencia_ms = int((time.time() - t0) * 1000)

    return {
        "id_mesa": contexto_mesa["id_mesa"],
        "mozos_recomendados": mozos_recomendados,
        "recomendaciones_por_comensal": recomendaciones,
        "modelo_version": models.get("version", "v1.0"),
        "latencia_ms": latencia_ms,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    caso1 = {
        "id_mesa": 42,
        "comensales": [
            {
                "id_persona_en_mesa": 1,
                "franja_etaria_persona": "adulto",
                "cant_acompañantes": 3,
                "motivo_visita": "negocios",

                "es_repetidor": True,
                "visitas_previas": 5,
                "ticket_promedio_historico": 3200.0,
                "orden_de_pedido": 1,
            }
        ],
        "dia_semana": 1,
        "franja_horaria": "mediodia",
    }

    caso2 = {
        "id_mesa": 15,
        "comensales": [
            {
                "id_persona_en_mesa": i,
                "franja_etaria_persona": ["joven", "joven", "adulto", "adulto"][i - 1],
                "cant_acompañantes": 3,
                "motivo_visita": "cumpleaños",

                "es_repetidor": i > 2,
                "visitas_previas": (i - 2) * 3 if i > 2 else 0,
                "ticket_promedio_historico": 2500.0 if i > 2 else None,
                "orden_de_pedido": i,
            }
            for i in range(1, 5)
        ],
        "dia_semana": 6,
        "franja_horaria": "noche",
    }

    caso3 = {
        "id_mesa": 8,
        "comensales": [
            {
                "id_persona_en_mesa": 1,
                "franja_etaria_persona": "adulto",
                "cant_acompañantes": 1,
                "motivo_visita": "date",

                "es_repetidor": False,
                "visitas_previas": 0,
                "ticket_promedio_historico": None,
                "orden_de_pedido": 1,
            },
            {
                "id_persona_en_mesa": 2,
                "franja_etaria_persona": "adulto",
                "cant_acompañantes": 1,
                "motivo_visita": "date",

                "es_repetidor": True,
                "visitas_previas": 8,
                "ticket_promedio_historico": 4500.0,
                "orden_de_pedido": 2,
            },
        ],
        "dia_semana": 5,
        "franja_horaria": "noche",
    }

    for i, caso in enumerate([caso1, caso2, caso3], 1):
        print(f"\n{'='*60}")
        print(f"Caso {i} — Mesa {caso['id_mesa']}")
        resultado = predict(caso)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
