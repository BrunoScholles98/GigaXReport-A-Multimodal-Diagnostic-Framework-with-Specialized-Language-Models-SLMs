import os
import re
from io import BytesIO

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import stringWidth

def generate_pdf_report(combined_pil_image, result_text, logo_path, pdf_logo_filename):
    """
    Generates a professional PDF report with analysis results.

    Args:
        combined_pil_image (PIL.Image.Image): The combined analysis image.
        result_text (str): The text result from the language model.
        logo_path (str): The base path to the static folder for the logo.
        pdf_logo_filename (str): The filename of the logo to use in the PDF.

    Returns:
        bytes: The generated PDF report as bytes.
    """
    buf_pdf = BytesIO()
    c = canvas.Canvas(buf_pdf, pagesize=letter)
    w, h = letter

    # Add company logo to PDF
    logo_p = os.path.join(logo_path, pdf_logo_filename)
    disp_h = 0
    if os.path.exists(logo_p):
        img_l = Image.open(logo_p)
        lw, lh = img_l.size
        dw = 150
        dh = int(dw * (lh / lw))
        c.drawInlineImage(img_l, 50, h - dh - 20, width=dw, height=dh)
        disp_h = dh

    # Add the combined analysis image
    pw, ph = combined_pil_image.size
    pw_pdf = 450
    ph_pdf = int(pw_pdf * (ph / pw))
    y0 = h - (disp_h + 40) - ph_pdf
    c.drawInlineImage(combined_pil_image, 50, y0, width=pw_pdf, height=ph_pdf)

    # Register fonts for text formatting
    try:
        # This might fail if the font is not in a standard location.
        # ReportLab will fall back to a default font.
        pdfmetrics.registerFont(TTFont('Helvetica-Bold', 'Helvetica-Bold.ttf'))
    except Exception:
        # Font registration is not critical; log or pass if needed
        pass

    # Process and format the MedGemma response for PDF
    clean = re.sub(r'(?<!\*)\*(?!\*)', '-', result_text)
    paras = clean.split('\n')

    def segs(line):
        """Split text into segments for bold formatting"""
        parts = re.split(r'(\*\*[^*]+\*\*)', line)
        out = []
        for p in parts:
            if p.startswith('**') and p.endswith('**'):
                out.append((p[2:-2], True))
            else:
                out.append((p, False))
        return out

    # Add formatted text to PDF
    txt = c.beginText(50, y0 - 20)
    txt.setLeading(12)
    for para in paras:
        if not para.strip():
            txt.textLine('')
            continue
        curr_w = 0
        for seg, bold in segs(para):
            font = 'Helvetica-Bold' if bold else 'Helvetica'
            for tok in re.split(r'(\s+)', seg):
                if not tok:
                    continue
                # Use a default font size of 8
                tw = stringWidth(tok, font, 8)
                if curr_w + tw > 500 and tok.strip():
                    txt.textLine('')
                    curr_w = 0
                txt.setFont(font, 8)
                txt.textOut(tok)
                curr_w += tw
        txt.textLine('')
        
    c.drawText(txt)
    c.showPage()
    c.save()

    buf_pdf.seek(0)
    return buf_pdf.read() 