"""
make_memoria_pdf.py

Convierte reports/report.md a un PDF simple (memoria) usando reportlab.

Nota: no es un renderizador Markdown completo. Interpreta:
- títulos (#, ##, ###) como encabezados
- tablas Markdown como texto monoespaciado
- párrafos como texto normal

Salida:
- reports/memoria.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .utils import ensure_dirs


def _wrap_line(text: str, max_width: float, font: str, size: int) -> List[str]:
    """
    Divide una línea en múltiples líneas según ancho máximo.

    Parameters
    ----------
    text:
        Texto a envolver.
    max_width:
        Ancho máximo en puntos.
    font:
        Fuente.
    size:
        Tamaño.

    Returns
    -------
    list[str]
        Líneas envueltas.
    """
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def main() -> Path:
    """
    Genera reports/memoria.pdf desde reports/report.md.

    Returns
    -------
    Path
        Ruta del PDF generado.
    """
    _, _, reports_dir, _ = ensure_dirs()
    md_path = reports_dir / "report.md"
    pdf_path = reports_dir / "memoria.pdf"

    lines = md_path.read_text(encoding="utf-8").splitlines()

    canvas = Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin_x = 2.0 * cm
    margin_y = 2.0 * cm
    y = height - margin_y

    def new_page() -> None:
        nonlocal y
        canvas.showPage()
        y = height - margin_y

    def draw_text(text: str, font: str, size: int, leading: int) -> None:
        nonlocal y
        canvas.setFont(font, size)
        max_w = width - 2 * margin_x
        for wrapped in _wrap_line(text, max_w, font, size):
            if y < margin_y + leading:
                new_page()
                canvas.setFont(font, size)
            canvas.drawString(margin_x, y, wrapped)
            y -= leading

    in_table = False
    for line in lines:
        if line.strip().startswith("|") and "|" in line.strip()[1:]:
            in_table = True
        elif in_table and not line.strip():
            in_table = False

        if line.startswith("# "):
            draw_text(line[2:].strip(), "Helvetica-Bold", 16, 20)
            y -= 6
            continue
        if line.startswith("## "):
            draw_text(line[3:].strip(), "Helvetica-Bold", 13, 18)
            y -= 4
            continue
        if line.startswith("### "):
            draw_text(line[4:].strip(), "Helvetica-Bold", 11, 16)
            y -= 2
            continue

        if in_table:
            draw_text(line, "Courier", 8, 10)
            continue

        if not line.strip():
            y -= 10
            continue

        draw_text(line, "Helvetica", 10, 14)

    canvas.save()
    print(f"[make_memoria_pdf] Wrote: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    main()
