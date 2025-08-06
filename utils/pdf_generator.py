from fpdf import FPDF
import os
from datetime import datetime

def generate_pdf(data):
    text = data.get("análisis", "No se encontró información de análisis.")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)

    output_path = os.path.join("static", "informe_licitacion.pdf")
    pdf.output(output_path)

    return output_path
