"""
utils.py

Utilidades comunes para el proyecto PRACT2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def project_root() -> Path:
    """
    Devuelve la ruta raíz del proyecto (dos niveles arriba de este archivo).

    Returns
    -------
    Path
        Ruta raíz del proyecto.
    """
    return Path(__file__).resolve().parents[1]


def load_project_config() -> Dict[str, Any]:
    """
    Carga el archivo de configuración del proyecto.

    Returns
    -------
    dict
        Configuración (members, repo_url, video_url, parámetros).
    """
    cfg_path = project_root() / "config" / "project.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def ensure_dirs() -> Tuple[Path, Path, Path, Path]:
    """
    Asegura la existencia de directorios principales.

    Returns
    -------
    tuple(Path, Path, Path, Path)
        Rutas: raw_dir, processed_dir, reports_dir, tables_dir.
    """
    root = project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    reports_dir = root / "reports"
    tables_dir = reports_dir / "tables"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, reports_dir, tables_dir
