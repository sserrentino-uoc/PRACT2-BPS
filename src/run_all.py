"""
run_all.py

Orquestador del pipeline PRACT2.

Ejecuta en orden:
1) Descarga/carga (adult.zip preferido)
2) Integración/selección
3) Limpieza
4) Análisis
5) Reporte markdown
6) Memoria PDF
"""

from __future__ import annotations

from pathlib import Path

from . import analyze_data, build_report, clean_data, download_uci, integrate_select
from . import make_memoria_pdf


def main() -> Path:
    """
    Ejecuta el pipeline end-to-end.

    Returns
    -------
    Path
        Ruta del PDF final.
    """
    download_uci.main()
    integrate_select.main()
    clean_data.main()
    analyze_data.main()
    build_report.main()
    pdf = make_memoria_pdf.main()
    print(f"[run_all] Done. PDF: {pdf}")
    return pdf


if __name__ == "__main__":
    main()
