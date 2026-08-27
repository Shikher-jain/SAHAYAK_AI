import unittest
import tempfile
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from backend.ingestion.pdf import extract_pdf_text

class TestPDFIngestion(unittest.TestCase):
    def setUp(self):
        self.test_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, text="Hello PDF World!", ln=True)
        pdf.output(self.test_pdf.name)

    def test_extract_pdf_text(self):
        text = extract_pdf_text(Path(self.test_pdf.name))
        self.assertIn("Hello PDF World", text)

    def tearDown(self):
        import os
        if os.path.exists(self.test_pdf.name):
            os.remove(self.test_pdf.name)

if __name__ == "__main__":
    unittest.main()
