import unittest
from backend.ingestion.pdf import PDFIngestor
import tempfile

class TestPDFIngestor(unittest.TestCase):
    def setUp(self):
        self.ingestor = PDFIngestor()
        self.test_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        # Create a simple PDF
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Hello PDF World!", ln=True)
        pdf.output(self.test_pdf.name)

    def test_ingest_pdf(self):
        self.ingestor.ingest_pdf(self.test_pdf.name, metadata={"id": "test_pdf"})
        # No assertion: just check no error

    def tearDown(self):
        import os
        if os.path.exists(self.test_pdf.name):
            os.remove(self.test_pdf.name)

if __name__ == "__main__":
    unittest.main()
