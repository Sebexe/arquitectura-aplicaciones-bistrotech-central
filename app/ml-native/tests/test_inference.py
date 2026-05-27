"""
Tests de inferencia para BistroTech.

Ejecutar: pytest tests/test_inference.py -v

Requiere que los modelos estén entrenados (run_pipeline.py o train_*.py).
Si no existen, los tests se marcan como skip.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import pytest
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.inference import CURSO_DISH_RANGE, predict_menu, predict
from src.feature_engineering import build_features, get_inference_features, load_processed, save_processed
from src.generate_dataset import generate


MODELS_AVAILABLE = (
    os.path.exists("models/modelo_a_mozo.joblib")
    and os.path.exists("models/modelo_b_entrada.joblib")
    and os.path.exists("data/processed/preprocessor.joblib")
)


@pytest.fixture(scope="module")
def minimal_pipeline(tmp_path_factory):
    """
    Genera un dataset mínimo, entrena modelos ligeros y persiste para los tests.
    Esto permite correr los tests de inferencia sin depender del pipeline completo.
    """
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    tmp = tmp_path_factory.mktemp("models_test")
    original_cwd = os.getcwd()
    os.chdir(str(tmp))

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    df = generate(n_records=300, seed=42)
    df.to_csv("data/raw/reservas.csv", index=False)
    X, targets = build_features(df)
    save_processed(X, targets)

    # Modelo A mínimo
    idx = targets["propina_rate"].index
    X_a = X.loc[idx].copy()
    X_a["id_mozo"] = df.loc[idx, "id_mozo"].values
    y_a = targets["propina_rate"]

    model_a = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model_a.fit(X_a, y_a)
    joblib.dump(model_a, "models/modelo_a_mozo.joblib")
    feature_names_a = list(X_a.columns)
    with open("models/feature_names_a.json", "w") as f:
        json.dump(feature_names_a, f)

    # Modelos B mínimos
    for curso, col in [("entrada", "id_entrada"), ("principal", "id_principal"),
                       ("postre", "id_postre"), ("bebida", "id_bebida")]:
        y_raw = targets[col]
        X_c = X.loc[y_raw.index]
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw.values.astype(int))
        model_b = XGBClassifier(
            n_estimators=5, max_depth=2, objective="multi:softprob",
            num_class=len(le.classes_), random_state=42,
        )
        model_b.fit(X_c, y_enc)
        joblib.dump(model_b, f"models/modelo_b_{curso}.joblib")
        joblib.dump(le, f"models/label_encoder_{curso}.joblib")

    yield str(tmp)

    os.chdir(original_cwd)


def _make_context(mesa_id: int = 1) -> dict:
    return {
        "id_mesa": mesa_id,
        "comensales": [
            {
                "id_persona_en_mesa": 1,
                "franja_etaria_persona": "adulto",
                "cant_acompañantes": 1,
                "motivo_visita": "casual",
                "es_repetidor": True,
                "visitas_previas": 3,
                "ticket_promedio_historico": 2000.0,
                "orden_de_pedido": 1,
            }
        ],
        "dia_semana": 2,
        "franja_horaria": "noche",
    }


def test_predict_output_format(minimal_pipeline):
    """
    predict() debe devolver exactamente el formato de la API definido en el README.

    Campos requeridos en el output:
      - id_mesa (int)
      - mozos_recomendados (list de dicts con id_mozo, propina_rate_esperado, rank)
      - recomendaciones_por_comensal (list de dicts con id_persona_en_mesa,
        entrada, principal, postre, bebida)
      - modelo_version (str)
      - latencia_ms (int)
    """
    os.chdir(minimal_pipeline)
    from src.inference import _models as _m
    import src.inference as inf_mod
    inf_mod._models = None

    contexto = _make_context(mesa_id=42)
    resultado = predict(contexto)

    assert isinstance(resultado, dict), "El output debe ser un dict."
    assert resultado["id_mesa"] == 42, "id_mesa debe coincidir con el input."

    assert "mozos_recomendados" in resultado, "Falta 'mozos_recomendados'."
    assert isinstance(resultado["mozos_recomendados"], list), "'mozos_recomendados' debe ser list."
    assert len(resultado["mozos_recomendados"]) > 0, "Debe haber al menos un mozo recomendado."

    for mozo in resultado["mozos_recomendados"]:
        assert "id_mozo" in mozo, f"Falta 'id_mozo' en {mozo}"
        assert "propina_rate_esperado" in mozo, f"Falta 'propina_rate_esperado' en {mozo}"
        assert "rank" in mozo, f"Falta 'rank' en {mozo}"
        assert isinstance(mozo["id_mozo"], int), "'id_mozo' debe ser int."
        assert 1 <= mozo["id_mozo"] <= 8, f"id_mozo fuera de rango: {mozo['id_mozo']}"
        assert 0.0 <= mozo["propina_rate_esperado"] <= 1.0

    ranks = [m["rank"] for m in resultado["mozos_recomendados"]]
    assert ranks == sorted(ranks), "Los mozos deben estar ordenados por rank."

    assert "recomendaciones_por_comensal" in resultado, "Falta 'recomendaciones_por_comensal'."
    assert len(resultado["recomendaciones_por_comensal"]) == len(contexto["comensales"])

    for rec in resultado["recomendaciones_por_comensal"]:
        assert "id_persona_en_mesa" in rec
        for curso in ["entrada", "principal", "postre", "bebida"]:
            assert curso in rec, f"Falta curso '{curso}' en recomendaciones."
            for item in rec[curso]:
                assert "id_plato" in item, f"Falta 'id_plato' en {item}"
                assert "score" in item, f"Falta 'score' en {item}"
                assert "rank" in item, f"Falta 'rank' en {item}"

    assert "modelo_version" in resultado, "Falta 'modelo_version'."
    assert "latencia_ms" in resultado, "Falta 'latencia_ms'."
    assert isinstance(resultado["latencia_ms"], int), "'latencia_ms' debe ser int."


def test_recommended_dishes_in_correct_course_range(minimal_pipeline):
    """
    Cada plato recomendado debe pertenecer al rango de IDs de su curso.

    Verifica la integridad del output: entradas 1-8, principales 9-20,
    postres 21-25, bebidas 26-30.
    """
    os.chdir(minimal_pipeline)
    import src.inference as inf_mod
    inf_mod._models = None

    load_processed(path="data/processed/")

    contexto = _make_context(mesa_id=10)
    X_inf = get_inference_features(contexto)

    menu = predict_menu(X_inf.iloc[[0]])

    for curso, recomendaciones in menu.items():
        valid_ids = set(CURSO_DISH_RANGE[curso])
        for rec in recomendaciones:
            assert rec["id_plato"] in valid_ids, (
                f"Plato #{rec['id_plato']} recomendado en '{curso}' "
                f"pero su ID no pertenece al rango {min(valid_ids)}-{max(valid_ids)}."
            )
