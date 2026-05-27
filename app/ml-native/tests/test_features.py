"""
Tests de feature engineering para BistroTech.

Ejecutar: pytest tests/test_features.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.feature_engineering import (
    FEEDBACK_COLS,
    build_features,
    get_inference_features,
    load_processed,
    save_processed,
)
from src.generate_dataset import generate


@pytest.fixture(scope="module")
def small_df():
    """Dataset pequeño para tests rápidos."""
    return generate(n_records=500, seed=7)


@pytest.fixture(scope="module")
def built_features(small_df, tmp_path_factory):
    """Construye features y las persiste en un directorio temporal."""
    X, targets = build_features(small_df)
    path = str(tmp_path_factory.mktemp("processed"))
    save_processed(X, targets, path=path)
    return X, targets, path


def test_no_leakage(small_df):
    """
    X no debe contener ninguna columna de feedback post-servicio.

    Las columnas de feedback (proporcion_dejada_*, like_*, hora_retiro_plato,
    score_satisfaccion_*, monto_propina) solo están disponibles después del servicio
    y no deben ser features del modelo.
    """
    X, targets = build_features(small_df)

    cols_x = set(X.columns)
    leakage = cols_x & FEEDBACK_COLS
    assert leakage == set(), (
        f"Data leakage: columnas de feedback encontradas en X: {leakage}"
    )

    raw_cols_to_exclude = {"id_registro", "id_mesa", "id_cliente", "fecha_hora"}
    for raw_col in raw_cols_to_exclude & set(X.columns):
        assert False, f"Columna de identidad '{raw_col}' no debería estar en X."


def test_inference_features_shape(built_features):
    """
    get_inference_features debe devolver exactamente las mismas columnas que X de entrenamiento.

    Esto garantiza que el vector de features en inferencia es compatible
    con el modelo entrenado sin necesidad de reindexado manual.
    """
    X, targets, path = built_features
    load_processed(path=path)

    contexto = {
        "id_mesa": 99,
        "comensales": [
            {
                "id_persona_en_mesa": 1,
                "franja_etaria_persona": "adulto",
                "cant_acompañantes": 2,
                "motivo_visita": "casual",
                "restriccion_alimentaria": "ninguna",
                "es_repetidor": True,
                "visitas_previas": 3,
                "ticket_promedio_historico": 2500.0,
                "orden_de_pedido": 1,
            },
            {
                "id_persona_en_mesa": 2,
                "franja_etaria_persona": "joven",
                "cant_acompañantes": 2,
                "motivo_visita": "casual",
                "restriccion_alimentaria": "vegetariano",
                "es_repetidor": False,
                "visitas_previas": 0,
                "ticket_promedio_historico": None,
                "orden_de_pedido": 2,
            },
        ],
        "dia_semana": 3,
        "franja_horaria": "noche",
    }

    X_inf = get_inference_features(contexto)

    assert list(X_inf.columns) == list(X.columns), (
        f"Columnas de inferencia no coinciden con entrenamiento.\n"
        f"Esperadas: {list(X.columns)}\nObtenidas: {list(X_inf.columns)}"
    )
    assert X_inf.shape[0] == len(contexto["comensales"]), (
        f"Número de filas esperado: {len(contexto['comensales'])}, obtenido: {X_inf.shape[0]}"
    )
    assert not X_inf.isnull().any().any(), "X de inferencia contiene NaN inesperados."


def test_cold_start_imputation(built_features):
    """
    ticket_promedio_historico nulo debe imputarse con la media del segmento.

    El segmento se define por (franja_etaria_persona, franja_horaria, motivo_visita).
    Un cliente sin historial previo (es_repetidor=False, ticket=None) debe recibir
    el valor del segmento y NO generar un NaN o cero en las features.
    """
    X, targets, path = built_features
    load_processed(path=path)

    contexto_cold = {
        "id_mesa": 1,
        "comensales": [
            {
                "id_persona_en_mesa": 1,
                "franja_etaria_persona": "joven",
                "cant_acompañantes": 0,
                "motivo_visita": "turista",
                "restriccion_alimentaria": "ninguna",
                "es_repetidor": False,
                "visitas_previas": 0,
                "ticket_promedio_historico": None,
                "orden_de_pedido": 1,
            }
        ],
        "dia_semana": 0,
        "franja_horaria": "noche",
    }

    X_inf = get_inference_features(contexto_cold)

    ticket_val = float(X_inf["ticket_promedio_historico"].iloc[0])
    assert not np.isnan(ticket_val), (
        "ticket_promedio_historico debe imputarse cuando es null, no quedar NaN."
    )
    assert 0.0 <= ticket_val <= 1.0, (
        f"ticket_promedio_historico imputado debe estar escalado en [0,1], valor={ticket_val}"
    )

    contexto_known = {
        "id_mesa": 2,
        "comensales": [
            {
                "id_persona_en_mesa": 1,
                "franja_etaria_persona": "joven",
                "cant_acompañantes": 0,
                "motivo_visita": "turista",
                "restriccion_alimentaria": "ninguna",
                "es_repetidor": True,
                "visitas_previas": 5,
                "ticket_promedio_historico": 1000.0,
                "orden_de_pedido": 1,
            }
        ],
        "dia_semana": 0,
        "franja_horaria": "noche",
    }
    X_known = get_inference_features(contexto_known)
    assert not np.isnan(float(X_known["ticket_promedio_historico"].iloc[0])), (
        "ticket_promedio_historico conocido no debe generar NaN."
    )
