"""
clean_data.py

Etapa 3: limpieza del dataset.
Entrada: data/processed/adult_selected.csv
Salida:  data/processed/adult_clean.csv

Incluye:
- Recuento de faltantes reales (NaN) antes/después.
- Recuento de faltantes semánticos antes/después (p.ej. '?', vacío, 'nan').
- Tipificación numérica segura.
- Limpieza e imputación:
  * categóricas: 'Unknown'
  * numéricas: mediana
- Tratamiento de extremos:
  * winsorización al 99.5% en capital_gain y capital_loss
- Tablas de trazabilidad en reports/tables/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dirs


@dataclass(frozen=True)
class WinsorCaps:
    """
    Contenedor para límites de winsorización.

    Attributes
    ----------
    col:
        Nombre de la columna.
    cap:
        Valor máximo permitido (percentil).
    n_capped:
        Cantidad de observaciones recortadas.
    """

    col: str
    cap: float
    n_capped: int


CAT_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "income",
]

NUM_COLS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla de faltantes reales (NaN) por columna.
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100
    out = pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    )
    return out.sort_values("missing_count", ascending=False)


def _semantic_missing_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Tabla de faltantes semánticos por columna (incluye '?', vacío, 'nan', 'None').

    Nota: No requiere que el valor sea NaN; detecta strings equivalentes a faltante.
    """
    out_rows = []
    n = len(df)
    for col in cols:
        s = df[col].astype("string").str.strip()
        sem = s.isna() | s.isin(["?", "", "nan", "None"])
        cnt = int(sem.sum())
        pct = float(cnt / n) * 100.0 if n else 0.0
        out_rows.append({"col": col, "missing_count": cnt, "missing_pct": pct})
    out = pd.DataFrame(out_rows).set_index("col")
    return out.sort_values("missing_count", ascending=False)


def _clean_cat(series: pd.Series) -> pd.Series:
    """
    Limpia una serie categórica y resuelve faltantes.

    Convierte: '?', '', 'nan' (string) y NA reales a NA y luego a 'Unknown'.
    """
    s = series.astype("string").str.strip()
    s = s.replace({"?": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
    s = s.fillna("Unknown")
    return s


def _impute_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Imputa variables numéricas con la mediana.
    """
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        med = float(out[col].median())
        out[col] = out[col].fillna(med)
    return out


def _winsorize(
    df: pd.DataFrame, col: str, q: float = 0.995
) -> Tuple[pd.DataFrame, WinsorCaps]:
    """
    Winsoriza superiormente una columna.
    """
    out = df.copy()
    cap = float(out[col].quantile(q))
    n_capped = int((out[col] > cap).sum())
    out[col] = np.minimum(out[col], cap)
    return out, WinsorCaps(col=col, cap=cap, n_capped=n_capped)


def main() -> Path:
    """
    Ejecuta la limpieza y escribe adult_clean.csv.

    Returns
    -------
    Path
        Ruta del CSV limpio.
    """
    _, processed_dir, _, tables_dir = ensure_dirs()
    in_csv = processed_dir / "adult_selected.csv"
    out_csv = processed_dir / "adult_clean.csv"

    df = pd.read_csv(in_csv)

    # Tablas antes (reales y semánticos)
    missing_before = _missing_table(df)
    missing_before.to_csv(tables_dir / "missing_before.csv")

    semantic_before = _semantic_missing_table(df, [c for c in CAT_COLS if c in df])
    semantic_before.to_csv(tables_dir / "semantic_missing_before.csv")

    # Limpieza categóricas
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = _clean_cat(df[col])

    # Numéricas
    df = _impute_numeric(df, NUM_COLS)

    # Winsor
    caps: List[WinsorCaps] = []
    for col in ["capital_gain", "capital_loss"]:
        df, cap_meta = _winsorize(df, col=col, q=0.995)
        caps.append(cap_meta)

    caps_df = pd.DataFrame(
        [{"col": c.col, "cap": c.cap, "n_capped": c.n_capped} for c in caps]
    )
    caps_df.to_csv(tables_dir / "winsor_caps.csv", index=False)

    # Tablas después
    missing_after = _missing_table(df)
    missing_after.to_csv(tables_dir / "missing_after.csv")

    semantic_after = _semantic_missing_table(df, [c for c in CAT_COLS if c in df])
    semantic_after.to_csv(tables_dir / "semantic_missing_after.csv")

    df.to_csv(out_csv, index=False)
    print(f"[clean_data] Wrote: {out_csv} ({len(df)} rows)")
    return out_csv


if __name__ == "__main__":
    main()
