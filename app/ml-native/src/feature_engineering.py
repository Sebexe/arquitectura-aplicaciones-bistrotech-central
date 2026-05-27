"""
Feature engineering para BistroTech.

X no contiene ninguna columna de feedback (anti-leakage).
id_mozo no está en X porque no es conocido al momento de la inferencia
para Modelo A (es lo que se recomienda); train_modelo_a lo agrega internamente.
"""
import json
import os
import logging
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

FEEDBACK_COLS = {
    "proporcion_dejada_entrada",
    "proporcion_dejada_principal",
    "proporcion_dejada_postre",
    "like_mozo",
    "like_entrada",
    "like_principal",
    "like_postre",
    "like_bebida",
    "hora_retiro_plato",
    "score_satisfaccion_entrada",
    "score_satisfaccion_principal",
    "score_satisfaccion_postre",
    "monto_propina",
}

CAT_FEATURES = {
    "franja_horaria": ["mediodia", "noche", "tarde"],
    "franja_etaria_persona": ["adulto", "joven", "senior"],
    "motivo_visita": ["casual", "cumpleaños", "date", "negocios", "turista"],
}

_preprocessor: dict | None = None


def _get_preprocessor() -> dict:
    if _preprocessor is None:
        raise RuntimeError(
            "Preprocessor no inicializado. Llamá build_features() o load_processed() primero."
        )
    return _preprocessor


def _set_preprocessor(p: dict) -> None:
    global _preprocessor
    _preprocessor = p


def _fit_preprocessor(df: pd.DataFrame) -> dict:
    from sklearn.preprocessing import MinMaxScaler

    segment_means = (
        df.groupby(["franja_etaria_persona", "franja_horaria", "motivo_visita"])[
            "ticket_promedio_historico"
        ]
        .mean()
        .to_dict()
    )
    global_mean_ticket = float(df["ticket_promedio_historico"].mean())

    ticket_filled = df["ticket_promedio_historico"].fillna(global_mean_ticket)
    scaler_ticket = MinMaxScaler()
    scaler_ticket.fit(ticket_filled.values.reshape(-1, 1).astype(float))

    scaler_cant = MinMaxScaler()
    scaler_cant.fit(df["cant_acompañantes"].values.reshape(-1, 1).astype(float))

    scaler_visitas = MinMaxScaler()
    scaler_visitas.fit(np.log1p(df["visitas_previas"].values).reshape(-1, 1).astype(float))

    scaler_orden = MinMaxScaler()
    scaler_orden.fit(df["orden_de_pedido"].values.reshape(-1, 1).astype(float))

    return {
        "segment_means": segment_means,
        "global_mean_ticket": global_mean_ticket,
        "scalers": {
            "ticket_promedio_historico": scaler_ticket,
            "cant_acompañantes": scaler_cant,
            "visitas_previas": scaler_visitas,
            "orden_de_pedido": scaler_orden,
        },
    }


def _impute_ticket(ticket_series: pd.Series, df_ctx: pd.DataFrame, preprocessor: dict) -> pd.Series:
    """Imputa ticket_promedio_historico nulo usando media del segmento."""
    ticket = ticket_series.copy()
    null_mask = ticket.isna()
    for idx in ticket[null_mask].index:
        key = (
            df_ctx.loc[idx, "franja_etaria_persona"],
            df_ctx.loc[idx, "franja_horaria"],
            df_ctx.loc[idx, "motivo_visita"],
        )
        ticket.loc[idx] = preprocessor["segment_means"].get(
            key, preprocessor["global_mean_ticket"]
        )
    return ticket


def _apply_transforms(df: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    result["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    result["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)

    for col, categories in CAT_FEATURES.items():
        for cat in sorted(categories):
            result[f"{col}_{cat}"] = (df[col] == cat).astype(int)

    result["viene_solo"] = df["viene_solo"].astype(int)
    result["es_repetidor"] = df["es_repetidor"].astype(int)

    scalers = preprocessor["scalers"]

    result["cant_acompañantes"] = scalers["cant_acompañantes"].transform(
        df["cant_acompañantes"].values.reshape(-1, 1).astype(float)
    ).flatten()

    result["visitas_previas"] = scalers["visitas_previas"].transform(
        np.log1p(df["visitas_previas"].values).reshape(-1, 1).astype(float)
    ).flatten()

    ticket_imputed = _impute_ticket(df["ticket_promedio_historico"], df, preprocessor)
    result["ticket_promedio_historico"] = scalers["ticket_promedio_historico"].transform(
        ticket_imputed.values.reshape(-1, 1).astype(float)
    ).flatten()

    result["orden_de_pedido"] = scalers["orden_de_pedido"].transform(
        df["orden_de_pedido"].values.reshape(-1, 1).astype(float)
    ).flatten()

    return result


def build_features(df_raw: pd.DataFrame) -> tuple:
    """
    Transforma el DataFrame crudo en features listas para el modelo y targets separados.

    Anti-leakage: X no contiene ninguna columna de feedback (proporcion_dejada_*,
    like_*, hora_retiro_plato, score_satisfaccion_*, monto_propina).

    Cada target está indexado sobre las filas donde el feedback correspondiente
    no es null. Para usar con un target: X.loc[targets['propina_rate'].index].

    Args:
        df_raw: DataFrame con el esquema de la tabla `registros`.

    Returns:
        tuple(X: pd.DataFrame, targets: dict)
        targets keys: 'propina_rate', 'id_entrada', 'id_principal', 'id_postre', 'id_bebida'
    """
    preprocessor = _fit_preprocessor(df_raw)
    _set_preprocessor(preprocessor)

    X = _apply_transforms(df_raw, preprocessor)

    leakage = set(X.columns) & FEEDBACK_COLS
    if leakage:
        raise ValueError(f"Data leakage detectado en X: {leakage}")

    preprocessor["feature_names"] = list(X.columns)

    targets = {}

    mask_propina = df_raw["propina_rate"].notna()
    targets["propina_rate"] = df_raw.loc[mask_propina, "propina_rate"].copy()

    mask_entrada = df_raw["score_satisfaccion_entrada"].notna()
    targets["id_entrada"] = df_raw.loc[mask_entrada, "id_entrada"].copy()

    mask_principal = df_raw["score_satisfaccion_principal"].notna()
    targets["id_principal"] = df_raw.loc[mask_principal, "id_principal"].copy()

    mask_postre = df_raw["score_satisfaccion_postre"].notna()
    targets["id_postre"] = df_raw.loc[mask_postre, "id_postre"].copy()

    mask_bebida = df_raw["id_bebida"].notna() & df_raw["like_bebida"].notna()
    targets["id_bebida"] = df_raw.loc[mask_bebida, "id_bebida"].copy()

    logger.info(
        "Features construidas: X=%s | propina=%d | entrada=%d | principal=%d | postre=%d | bebida=%d",
        X.shape,
        len(targets["propina_rate"]),
        len(targets["id_entrada"]),
        len(targets["id_principal"]),
        len(targets["id_postre"]),
        len(targets["id_bebida"]),
    )
    return X, targets


def get_inference_features(contexto: dict) -> pd.DataFrame:
    """
    Convierte el JSON de input de la API en un DataFrame con las mismas columnas que X.

    Aplica imputación de ticket_promedio_historico por segmento si es null.
    Requiere que load_processed() o build_features() hayan sido llamados previamente.

    Args:
        contexto: dict con formato del README (id_mesa, comensales, dia_semana, franja_horaria).

    Returns:
        pd.DataFrame con una fila por comensal, columnas = feature_names del entrenamiento.
    """
    preprocessor = _get_preprocessor()
    rows = []
    dia_semana = contexto["dia_semana"]
    franja_horaria = contexto["franja_horaria"]

    for c in contexto["comensales"]:
        ticket = c.get("ticket_promedio_historico")
        if ticket is None:
            key = (c["franja_etaria_persona"], franja_horaria, c["motivo_visita"])
            ticket = preprocessor["segment_means"].get(key, preprocessor["global_mean_ticket"])

        rows.append(
            {
                "dia_semana": dia_semana,
                "franja_horaria": franja_horaria,
                "franja_etaria_persona": c["franja_etaria_persona"],
                "cant_acompañantes": c["cant_acompañantes"],
                "viene_solo": c["cant_acompañantes"] == 0,
                "es_repetidor": c["es_repetidor"],
                "visitas_previas": c.get("visitas_previas", 0),
                "ticket_promedio_historico": ticket,
                "motivo_visita": c["motivo_visita"],
                "orden_de_pedido": c.get("orden_de_pedido", 1),
            }
        )

    df_input = pd.DataFrame(rows)
    df_input["viene_solo"] = df_input["viene_solo"].astype(bool)
    df_input["es_repetidor"] = df_input["es_repetidor"].astype(bool)

    scalers = preprocessor["scalers"]

    result = pd.DataFrame()
    result["dia_semana_sin"] = np.sin(2 * np.pi * df_input["dia_semana"] / 7)
    result["dia_semana_cos"] = np.cos(2 * np.pi * df_input["dia_semana"] / 7)

    for col, categories in CAT_FEATURES.items():
        for cat in sorted(categories):
            result[f"{col}_{cat}"] = (df_input[col] == cat).astype(int)

    result["viene_solo"] = df_input["viene_solo"].astype(int)
    result["es_repetidor"] = df_input["es_repetidor"].astype(int)

    result["cant_acompañantes"] = scalers["cant_acompañantes"].transform(
        df_input["cant_acompañantes"].values.reshape(-1, 1).astype(float)
    ).flatten()

    result["visitas_previas"] = scalers["visitas_previas"].transform(
        np.log1p(df_input["visitas_previas"].values).reshape(-1, 1).astype(float)
    ).flatten()

    result["ticket_promedio_historico"] = scalers["ticket_promedio_historico"].transform(
        df_input["ticket_promedio_historico"].values.reshape(-1, 1).astype(float)
    ).flatten()

    result["orden_de_pedido"] = scalers["orden_de_pedido"].transform(
        df_input["orden_de_pedido"].values.reshape(-1, 1).astype(float)
    ).flatten()

    expected_cols = preprocessor["feature_names"]
    for col in expected_cols:
        if col not in result.columns:
            result[col] = 0

    return result[expected_cols].reset_index(drop=True)


def _preprocessor_to_json(preprocessor: dict) -> dict:
    """Convierte el preprocessor (con objetos sklearn) a un dict JSON-serializable."""
    scalers_json = {}
    for name, scaler in preprocessor["scalers"].items():
        scalers_json[name] = {
            "data_min_": scaler.data_min_.tolist(),
            "data_range_": scaler.data_range_.tolist(),
        }
    segment_means_json = {str(k): v for k, v in preprocessor["segment_means"].items()}
    return {
        "segment_means": segment_means_json,
        "global_mean_ticket": preprocessor["global_mean_ticket"],
        "feature_names": preprocessor["feature_names"],
        "scalers": scalers_json,
    }


def save_processed(X: pd.DataFrame, targets: dict, path: str = "data/processed/") -> None:
    """
    Persiste X, targets y el estado del preprocesador usando joblib y JSON.

    Args:
        X: DataFrame de features (sin leakage).
        targets: dict con Series de targets indexadas.
        path: directorio de salida.
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(X, os.path.join(path, "X.joblib"))
    joblib.dump(targets, os.path.join(path, "targets.joblib"))
    preprocessor = _get_preprocessor()
    joblib.dump(preprocessor, os.path.join(path, "preprocessor.joblib"))
    preprocessor_json = _preprocessor_to_json(preprocessor)
    with open(os.path.join(path, "preprocessor.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessor_json, f, ensure_ascii=False)
    logger.info("Datos procesados guardados en %s", path)


def load_processed(path: str = "data/processed/") -> tuple:
    """
    Carga X, targets y restaura el estado del preprocesador para inferencia.

    Args:
        path: directorio con los archivos procesados.

    Returns:
        tuple(X: pd.DataFrame, targets: dict)
    """
    X = joblib.load(os.path.join(path, "X.joblib"))
    targets = joblib.load(os.path.join(path, "targets.joblib"))
    preprocessor = joblib.load(os.path.join(path, "preprocessor.joblib"))
    _set_preprocessor(preprocessor)
    logger.info("Datos procesados cargados desde %s (X=%s)", path, X.shape)
    return X, targets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.generate_dataset import generate

    df = generate(1000)
    X, targets = build_features(df)
    print(f"X shape: {X.shape}")
    print(f"Columnas X: {list(X.columns)}")
    for k, v in targets.items():
        print(f"  target '{k}': {len(v)} registros")
    save_processed(X, targets)
    X2, targets2 = load_processed()
    print(f"Load OK — X shape: {X2.shape}")
