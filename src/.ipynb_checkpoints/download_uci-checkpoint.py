\
"""
download_uci.py

Genera el CSV crudo del dataset Adult (train+test) en:
- data/raw/adult_raw.csv

Estrategia:
1) Si existe data/raw/adult.zip, lo usa (offline).
2) Si no existe, intenta descargar desde UCI (online).
3) Si falla la descarga, intenta usar data/raw/adult.data y data/raw/adult.test
   si existen.

El objetivo es que el pipeline sea reproducible sin depender de internet.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from .utils import ensure_dirs, project_root

UCI_BASE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
)
UCI_TRAIN = UCI_BASE + "adult.data"
UCI_TEST = UCI_BASE + "adult.test"

COLUMNS = [
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


def _read_uci_text(url: str) -> str:
    """
    Lee un recurso de texto remoto (UCI) y lo decodifica.

    Parameters
    ----------
    url:
        URL del recurso.

    Returns
    -------
    str
        Contenido decodificado.
    """
    with urlopen(url, timeout=30) as resp:  # nosec B310
        data = resp.read()
    return data.decode("utf-8", errors="replace")


def _parse_adult_text(text: str, is_test: bool) -> pd.DataFrame:
    """
    Parsea el texto del dataset Adult en un DataFrame.

    Parameters
    ----------
    text:
        Contenido del archivo (adult.data o adult.test).
    is_test:
        True si corresponde a adult.test (posee líneas con '.' final).

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas estándar.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("|"):
            continue
        if is_test and line.endswith("."):
            line = line[:-1]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(COLUMNS):
            continue
        rows.append(parts)
    return pd.DataFrame(rows, columns=COLUMNS)


def _load_from_zip(zip_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga adult.data y adult.test desde un ZIP local.

    Parameters
    ----------
    zip_path:
        Ruta al ZIP.

    Returns
    -------
    tuple(DataFrame, DataFrame)
        DataFrames train y test.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        train_name = None
        test_name = None
        for name in zf.namelist():
            low = name.lower()
            if low.endswith("adult.data"):
                train_name = name
            elif low.endswith("adult.test"):
                test_name = name

        if not train_name or not test_name:
            raise FileNotFoundError(
                "adult.zip no contiene adult.data y adult.test."
            )

        train_text = zf.read(train_name).decode("utf-8", errors="replace")
        test_text = zf.read(test_name).decode("utf-8", errors="replace")

    return _parse_adult_text(train_text, False), _parse_adult_text(
        test_text, True
    )


def _load_local_files(
    train_path: Path, test_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga archivos adult.data y adult.test locales.

    Parameters
    ----------
    train_path:
        Ruta a adult.data.
    test_path:
        Ruta a adult.test.

    Returns
    -------
    tuple(DataFrame, DataFrame)
        DataFrames train y test.
    """
    train_text = train_path.read_text(encoding="utf-8", errors="replace")
    test_text = test_path.read_text(encoding="utf-8", errors="replace")
    return _parse_adult_text(train_text, False), _parse_adult_text(
        test_text, True
    )


def main() -> Path:
    """
    Punto de entrada: genera data/raw/adult_raw.csv.

    Returns
    -------
    Path
        Ruta del CSV generado.
    """
    raw_dir, _, _, _ = ensure_dirs()
    out_csv = raw_dir / "adult_raw.csv"

    zip_path = raw_dir / "adult.zip"
    train_path = raw_dir / "adult.data"
    test_path = raw_dir / "adult.test"

    if zip_path.exists():
        df_train, df_test = _load_from_zip(zip_path)
    else:
        try:
            df_train = _parse_adult_text(_read_uci_text(UCI_TRAIN), False)
            df_test = _parse_adult_text(_read_uci_text(UCI_TEST), True)
        except (URLError, TimeoutError, OSError):
            if train_path.exists() and test_path.exists():
                df_train, df_test = _load_local_files(train_path, test_path)
            else:
                raise

    df = pd.concat([df_train, df_test], ignore_index=True)

    num_cols = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_csv(out_csv, index=False)
    print(f"[download_uci] Wrote: {out_csv} ({len(df)} rows)")
    return out_csv

if __name__ == "__main__":
    main()
