"""
Simulador de dataset para BistroTech.
Genera 10.000 registros respetando el esquema de la tabla `registros`.
"""
import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROP_MAP = {"nada": 0.0, "poco": 0.2, "mitad": 0.5, "mayoria": 0.8, "todo": 1.0}


def _gen_proporcion(like: bool) -> str:
    if like:
        return np.random.choice(
            ["nada", "poco", "mitad", "mayoria", "todo"],
            p=[0.60, 0.30, 0.07, 0.02, 0.01],
        )
    return np.random.choice(
        ["nada", "poco", "mitad", "mayoria", "todo"],
        p=[0.00, 0.00, 0.40, 0.40, 0.20],
    )


def _calc_score(like, proporcion: str, tiempo_normalizado: float):
    if like is None or proporcion is None:
        return None
    prop_val = PROP_MAP[proporcion]
    return round(0.5 * (1 - prop_val) + 0.3 * float(like) + 0.2 * tiempo_normalizado, 4)


def generate(n_records: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Genera el dataset simulado de visitas al restaurante."""
    np.random.seed(seed)
    records = []
    id_registro = 1
    id_mesa = 1
    id_cliente = 1
    start_date = pd.Timestamp("2024-01-01")

    while len(records) < n_records:
        mesa_size = int(np.random.randint(1, 9))

        offset_days = int(np.random.randint(0, 365))
        fecha_base = start_date + pd.Timedelta(days=offset_days)
        dia_semana = fecha_base.dayofweek

        franja_horaria = np.random.choice(
            ["mediodia", "tarde", "noche"], p=[0.40, 0.15, 0.45]
        )
        hora_inicio = {"mediodia": 12, "tarde": 16, "noche": 20}[franja_horaria]
        hora_base = fecha_base + pd.Timedelta(
            hours=hora_inicio, minutes=int(np.random.randint(0, 60))
        )

        id_mozo_mesa = int(np.random.randint(1, 9))

        for persona in range(1, mesa_size + 1):
            if len(records) >= n_records:
                break

            franja_etaria = np.random.choice(
                ["joven", "adulto", "senior"], p=[0.40, 0.45, 0.15]
            )

            if franja_horaria == "mediodia":
                motivo = np.random.choice(
                    ["negocios", "casual", "turista"], p=[0.50, 0.40, 0.10]
                )
            elif franja_horaria == "tarde":
                motivo = np.random.choice(
                    ["casual", "date", "turista", "cumpleaños"], p=[0.45, 0.25, 0.20, 0.10]
                )
            else:
                motivo = np.random.choice(
                    ["date", "cumpleaños", "casual", "turista"], p=[0.30, 0.25, 0.35, 0.10]
                )

            restriccion = np.random.choice(
                ["ninguna", "vegetariano", "vegano", "celiaco", "kosher"],
                p=[0.70, 0.15, 0.08, 0.05, 0.02],
            )

            es_repetidor = bool(np.random.random() < 0.60)
            if es_repetidor:
                visitas_previas = int(np.random.randint(1, 21))
                ticket_promedio_historico = round(float(np.random.uniform(800, 8000)), 2)
            else:
                visitas_previas = 0
                ticket_promedio_historico = (
                    None
                    if np.random.random() < 0.30
                    else round(float(np.random.uniform(500, 3000)), 2)
                )

            cliente_id = id_cliente if np.random.random() < 0.80 else None

            id_entrada = (
                None if np.random.random() < 0.20 else int(np.random.randint(1, 9))
            )
            id_principal = int(np.random.randint(9, 21))
            id_postre = (
                None if np.random.random() < 0.30 else int(np.random.randint(21, 26))
            )
            id_bebida = int(np.random.randint(26, 31))

            like_mozo = bool(np.random.random() < 0.65)
            like_entrada = bool(np.random.random() < 0.70) if id_entrada is not None else None
            like_principal = bool(np.random.random() < 0.70)
            like_postre = bool(np.random.random() < 0.70) if id_postre is not None else None
            like_bebida = bool(np.random.random() < 0.70)

            prop_entrada = _gen_proporcion(like_entrada) if id_entrada is not None else None
            prop_principal = _gen_proporcion(like_principal)
            prop_postre = _gen_proporcion(like_postre) if id_postre is not None else None

            hora_entrega = hora_base + pd.Timedelta(minutes=int(np.random.randint(10, 35)))
            tiempo_consumo_min = int(np.random.randint(15, 46))
            hora_retiro = hora_entrega + pd.Timedelta(minutes=tiempo_consumo_min)
            tiempo_normalizado = round(1.0 - tiempo_consumo_min / 45.0, 4)

            if like_mozo:
                propina_rate = round(float(np.random.uniform(0.10, 0.25)), 4)
            else:
                propina_rate = round(float(np.random.uniform(0.00, 0.08)), 4)

            total_cuenta = round(float(np.random.uniform(2000, 8000)), 2)
            monto_propina = round(propina_rate * total_cuenta, 2)

            records.append(
                {
                    "id_registro": id_registro,
                    "id_mesa": id_mesa,
                    "id_cliente": cliente_id,
                    "id_persona_en_mesa": persona,
                    "fecha_hora": hora_base,
                    "dia_semana": dia_semana,
                    "franja_horaria": franja_horaria,
                    "franja_etaria_persona": franja_etaria,
                    "cant_acompañantes": mesa_size - 1,
                    "viene_solo": bool(mesa_size == 1),
                    "es_repetidor": es_repetidor,
                    "visitas_previas": visitas_previas,
                    "ticket_promedio_historico": ticket_promedio_historico,
                    "motivo_visita": motivo,
                    "restriccion_alimentaria": restriccion,
                    "orden_de_pedido": persona,
                    "id_mozo": id_mozo_mesa,
                    "id_entrada": id_entrada,
                    "id_principal": id_principal,
                    "id_postre": id_postre,
                    "id_bebida": id_bebida,
                    "hora_entrega_plato": hora_entrega,
                    "hora_retiro_plato": hora_retiro,
                    "proporcion_dejada_entrada": prop_entrada,
                    "proporcion_dejada_principal": prop_principal,
                    "proporcion_dejada_postre": prop_postre,
                    "monto_propina": monto_propina,
                    "propina_rate": propina_rate,
                    "score_satisfaccion_entrada": _calc_score(
                        like_entrada, prop_entrada, tiempo_normalizado
                    ),
                    "score_satisfaccion_principal": _calc_score(
                        like_principal, prop_principal, tiempo_normalizado
                    ),
                    "score_satisfaccion_postre": _calc_score(
                        like_postre, prop_postre, tiempo_normalizado
                    ),
                    "like_mozo": like_mozo,
                    "like_entrada": like_entrada,
                    "like_principal": like_principal,
                    "like_postre": like_postre,
                    "like_bebida": like_bebida,
                }
            )

            id_registro += 1
            id_cliente += 1

        id_mesa += 1

    df = pd.DataFrame(records[:n_records])
    logger.info("Dataset generado: %d registros, %d mesas", len(df), df["id_mesa"].nunique())
    return df


def print_stats(df: pd.DataFrame) -> None:
    """Imprime estadísticas del dataset generado."""
    print(f"\n{'='*55}")
    print(f"Total registros       : {len(df)}")
    print(f"Mesas únicas          : {df['id_mesa'].nunique()}")
    print(f"Clientes únicos (id)  : {df['id_cliente'].nunique()}")
    print("\nDistribución franjas etarias:")
    dist = df["franja_etaria_persona"].value_counts(normalize=True).round(3)
    for k, v in dist.items():
        print(f"  {k:<10}: {v:.1%}")
    print("\nPropina rate promedio por mozo:")
    by_mozo = df.groupby("id_mozo")["propina_rate"].mean().round(4)
    for mozo, rate in by_mozo.items():
        print(f"  Mozo {mozo}: {rate:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate(10_000)
    output_path = "data/raw/reservas.csv"
    df.to_csv(output_path, index=False)
    logger.info("Guardado en %s", output_path)
    print_stats(df)
