from weasyprint import HTML, CSS

def html_to_pdf_bytes(html: str) -> bytes:
    pdf = HTML(string=html).write_pdf(stylesheets=[CSS(string="""
      @page { size: A4; margin: 18mm; }
      header { position: running(page-header); }
      footer { position: running(page-footer); font-size: 10px; color: #666; }
      .h1 { font-size: 18px; font-weight: 700; margin: 0 0 8px; }
      .h2 { font-size: 14px; font-weight: 600; margin: 14px 0 6px; }
      table { width: 100%; border-collapse: collapse; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 12px; }
      th { background: #f9fafb; text-align: left; }
    """)])
    return pdf
