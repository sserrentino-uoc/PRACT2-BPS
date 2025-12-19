# PRACT2 — Adult Income (Python)

## Integrantes
- NOMBRE APELLIDO (Integrante 1) — iniciales: AA
- NOMBRE APELLIDO (Integrante 2) — iniciales: BB

Repositorio: PENDIENTE

Vídeo (Google Drive UOC): PENDIENTE

## Estructura
- `src/`: código fuente (pipeline end-to-end)
- `data/raw/`: datos fuente (`adult.zip`) y CSV crudo generado
- `data/processed/`: dataset seleccionado y dataset limpio final
- `reports/`: resumen, tablas, figuras, reporte y memoria PDF
- `config/project.json`: metadatos del proyecto (nombres, links, parámetros)

## Requisitos (Python)
Instalar dependencias:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Ejecutar el pipeline completo:

```bash
python -m src.run_all
```

## Artefactos generados
- `data/raw/adult_raw.csv` (datos originales integrados train+test)
- `data/processed/adult_selected.csv` (selección)
- `data/processed/adult_clean.csv` (datos finales analizados)
- `reports/summary.json` (resumen numérico para trazabilidad)
- `reports/report.md` (reporte con observaciones/conclusiones)
- `reports/memoria.pdf` (memoria final en PDF)
