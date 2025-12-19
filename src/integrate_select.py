"""
integrate_select.py

Etapa 2: integración/selección.
Entrada: data/raw/adult_raw.csv
Salida:  data/processed/adult_selected.csv

La selección es explícita para trazabilidad. En este dataset se mantiene
la mayoría de variables estándar del Adult Income.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_dirs

from .utils import project_root

def main() -> Path:
    """
    Ejecuta la integración/selección y escribe adult_selected.csv.

    Returns
    -------
    Path
        Ruta del CSV seleccionado.
    """
    _, processed_dir, _, _ = ensure_dirs()
    in_csv = project_root() / "data" / "raw" / "adult_raw.csv"
    out_csv = processed_dir / "adult_selected.csv"

    df = pd.read_csv(in_csv)

    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    df = df[cols].copy()

    df.to_csv(out_csv, index=False)
    print(f"[integrate_select] Wrote: {out_csv} ({len(df)} rows)")
    return out_csv

if __name__ == "__main__":
    main()
