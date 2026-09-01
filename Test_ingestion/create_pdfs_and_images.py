"""Generates the PDF and remaining image samples missing from the first script."""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont
import random

BASE = Path(r"D:\shikher sih\SAHAYAK_AI\Test_ingestion\documents")

for folder in ["pdf", "images"]:
    (BASE / folder).mkdir(parents=True, exist_ok=True)

# --- text_pdf.pdf — real text layer, single column ---
c = canvas.Canvas(str(BASE / "pdf" / "text_pdf.pdf"), pagesize=letter)
c.setFont("Helvetica-Bold", 16)
c.drawString(72, 720, "Introduction to Machine Learning")
c.setFont("Helvetica", 11)
lines = [
    "Machine learning is a subset of artificial intelligence that enables",
    "systems to learn patterns from data without being explicitly programmed.",
    "",
    "There are three main types of machine learning: supervised learning,",
    "unsupervised learning, and reinforcement learning. Supervised learning",
    "uses labeled data to train models, while unsupervised learning finds",
    "hidden patterns in unlabeled data.",
    "",
    "Common applications include image recognition, natural language",
    "processing, recommendation systems, and predictive analytics.",
]
y = 690
for line in lines:
    c.drawString(72, y, line)
    y -= 18
c.save()
print("Created pdf/text_pdf.pdf")

# --- column_topic_pdf.pdf — two-column layout ---
c = canvas.Canvas(str(BASE / "pdf" / "column_topic_pdf.pdf"), pagesize=letter)
c.setFont("Helvetica-Bold", 16)
c.drawCentredString(300, 740, "Data Structures Overview")

left_col = [
    ("Arrays", ["An array is a collection of", "elements stored at contiguous", "memory locations. Access is", "O(1) by index."]),
    ("Linked Lists", ["A linked list is a linear data", "structure where elements point", "to the next node. Insertion is", "O(1) at the head."]),
]
right_col = [
    ("Stacks", ["A stack follows LIFO order.", "Push and pop operations both", "run in O(1) time complexity."]),
    ("Queues", ["A queue follows FIFO order.", "Enqueue and dequeue both run", "in O(1) time complexity."]),
]

def draw_column(c, x, sections, y_start):
    y = y_start
    for title, body_lines in sections:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, title)
        y -= 16
        c.setFont("Helvetica", 9)
        for line in body_lines:
            c.drawString(x, y, line)
            y -= 13
        y -= 10

draw_column(c, 60, left_col, 700)
draw_column(c, 320, right_col, 700)
c.save()
print("Created pdf/column_topic_pdf.pdf")

# --- Fonts for image-based content (Windows paths) ---
try:
    font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 40)
    font_body = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 26)
except Exception:
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

# --- scan_pdf.pdf — image-only page, no text layer (forces OCR) ---
img = Image.new("RGB", (1240, 1754), "white")
draw = ImageDraw.Draw(img)
draw.text((80, 100), "Scanned Document Sample", font=font_title, fill="black")
body = [
    "This page has no real text layer - it is a rendered image,",
    "simulating a scanned document. A PDF text extractor",
    "(pdfplumber/PyMuPDF) should return empty text for this page,",
    "which should trigger the OCR fallback tier in the ingestion",
    "pipeline (Tesseract).",
    "",
    "Photosynthesis is the process by which plants convert light",
    "energy into chemical energy stored in glucose.",
]
y = 220
for line in body:
    draw.text((80, y), line, font=font_body, fill="black")
    y += 44
img.save(str(BASE / "pdf" / "scan_pdf.pdf"), "PDF", resolution=150.0)
print("Created pdf/scan_pdf.pdf")

# --- image_pdf.pdf — second image-only page, different content ---
img2 = Image.new("RGB", (1240, 1754), "white")
draw2 = ImageDraw.Draw(img2)
draw2.text((80, 100), "Embedded Image Content", font=font_title, fill="black")
body2 = [
    "This simulates a PDF that is primarily an embedded image/diagram",
    "rather than a text page.",
    "",
    "Water boils at 100 degrees Celsius at standard atmospheric",
    "pressure. This value decreases at higher altitudes due to",
    "lower atmospheric pressure.",
]
y = 220
for line in body2:
    draw2.text((80, y), line, font=font_body, fill="black")
    y += 44
img2.save(str(BASE / "pdf" / "image_pdf.pdf"), "PDF", resolution=150.0)
print("Created pdf/image_pdf.pdf")

# --- images/sample.png — clean, OCR-friendly ---
img3 = Image.new("RGB", (900, 400), "white")
draw3 = ImageDraw.Draw(img3)
draw3.text((40, 40), "Clean Sample Image for OCR", font=font_title, fill="black")
draw3.text((40, 120), "The mitochondria is the powerhouse of the cell.", font=font_body, fill="black")
draw3.text((40, 170), "This is a clean, high-contrast test image.", font=font_body, fill="black")
img3.save(str(BASE / "images" / "sample.png"))
print("Created images/sample.png")

# --- images/scanned.jpg — noisier scan-style image ---
img4 = Image.new("RGB", (900, 400), (245, 245, 240))
draw4 = ImageDraw.Draw(img4)
draw4.text((40, 40), "Lower Quality Scan Simulation", font=font_title, fill=(30, 30, 30))
draw4.text((40, 120), "Gravity causes objects to accelerate toward Earth", font=font_body, fill=(40, 40, 40))
draw4.text((40, 170), "at approximately 9.8 meters per second squared.", font=font_body, fill=(40, 40, 40))
pixels = img4.load()
for _ in range(15000):
    x, y = random.randint(0, 899), random.randint(0, 399)
    pixels[x, y] = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
img4.save(str(BASE / "images" / "scanned.jpg"), quality=70)
print("Created images/scanned.jpg")

print("\nAll PDF + image samples created.")