import unittest
from io import BytesIO

from fpdf import FPDF

from backend.ingestion.pdf import extract_pdf_text_from_bytes


class TestPDFCleaning(unittest.TestCase):
    def test_extract_pdf_text_removes_noise(self):
        pdf = FPDF()
        for idx in range(1, 3):
            pdf.add_page()
            pdf.set_font("helvetica", size=12)
            pdf.cell(w=0, h=10, text="Company Confidential", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(w=0, h=10, text=f"Page {idx} content line", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(w=0, h=10, text=f"- Bullet {idx}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(w=0, h=10, text=f"Page {idx} of 2", new_x="LMARGIN", new_y="NEXT")
        buffer = BytesIO()
        pdf.output(buffer)
        payload = buffer.getvalue()

        cleaned = extract_pdf_text_from_bytes(payload)

        self.assertNotIn("Company Confidential", cleaned)
        self.assertNotIn("Page 1 of 2", cleaned)
        self.assertIn("Page 1 content line", cleaned)
        self.assertNotIn("- Bullet 1", cleaned)


if __name__ == "__main__":
    unittest.main()
