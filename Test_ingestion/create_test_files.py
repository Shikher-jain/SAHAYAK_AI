from pathlib import Path
import csv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont

BASE = Path(r"D:\shikher sih\Test_ingestion\documents")

for folder in ["images", "csv", "code", "text", "links"]:
    (BASE / folder).mkdir(parents=True, exist_ok=True)

# handwritten.jpg
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\ariali.ttf", 30)
except:
    font = ImageFont.load_default()

img = Image.new("RGB", (900, 300), (250, 248, 240))
draw = ImageDraw.Draw(img)
draw.text((40, 40), "This is a handwriting-STYLE approximation", font=font, fill=(20,20,60))
draw.text((40, 100), "using an italic font, not real handwriting.", font=font, fill=(20,20,60))
draw.text((40, 160), "Newton's second law: Force equals mass times acceleration.", font=font, fill=(20,20,60))
img.save(BASE / "images" / "handwritten.jpg", quality=75)

# CSV
rows = [
    [101, "Aditi Sharma", "Mathematics", 88, "A"],
    [102, "Rohan Verma", "Physics", 76, "B"],
    [103, "Priya Nair", "Chemistry", 92, "A+"],
    [104, "Karan Mehta", "Mathematics", 65, "C"],
    [105, "Sneha Iyer", "Biology", 81, "A"],
    [106, "Arjun Singh", "Physics", 58, "D"],
    [107, "Divya Reddy", "Chemistry", 95, "A+"],
]
with open(BASE / "csv" / "csv_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["student_id", "name", "subject", "score", "grade"])
    writer.writerows(rows)

# Python
(BASE / "code" / "sample.py").write_text('''"""Sample Python module for code-ingestion testing."""

def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

class BankAccount:
    """A minimal bank account with deposit/withdraw operations."""

    def __init__(self, owner: str, balance: float = 0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount

if __name__ == "__main__":
    print(f"Fib(10) = {fibonacci(10)}")
    acc = BankAccount("Shikher", 1000)
    acc.deposit(500)
    print(f"Balance: {acc.balance}")
''', encoding="utf-8")

# JavaScript
(BASE / "code" / "sample.js").write_text('''// Sample JavaScript module for code-ingestion testing

function isPrime(num) {
  if (num < 2) return false;
  for (let i = 2; i <= Math.sqrt(num); i++) {
    if (num % i === 0) return false;
  }
  return true;
}

class TodoList {
  constructor() {
    this.items = [];
  }

  add(task) {
    this.items.push({ task, done: false });
  }

  complete(index) {
    if (this.items[index]) {
      this.items[index].done = true;
    }
  }
}

const list = new TodoList();
list.add("Write ingestion tests");
list.add("Deploy to Render");
console.log(list.items);
''', encoding="utf-8")

# C++
(BASE / "code" / "sample.cpp").write_text('''// Sample C++ file for code-ingestion testing
#include <iostream>
#include <vector>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

class Rectangle {
public:
    double width, height;
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const { return width * height; }
};

int main() {
    std::cout << "Factorial of 5: " << factorial(5) << std::endl;
    Rectangle r(4.0, 5.0);
    std::cout << "Area: " << r.area() << std::endl;
    return 0;
}
''', encoding="utf-8")

# Text
(BASE / "text" / "sample.txt").write_text("""Climate Change and Renewable Energy

Climate change refers to long-term shifts in temperatures and weather
patterns, primarily driven by human activities, especially the burning
of fossil fuels since the industrial revolution.

Renewable energy sources such as solar, wind, and hydroelectric power
offer a path toward reducing greenhouse gas emissions. Solar panels
convert sunlight directly into electricity using the photovoltaic
effect, while wind turbines harness kinetic energy from moving air.

Governments and organizations worldwide are increasingly investing in
renewable infrastructure to mitigate the effects of climate change and
transition toward a more sustainable energy future.
""", encoding="utf-8")

# Links
(BASE / "links" / "links.txt").write_text("""# Test URLs for /ingest/url
https://en.wikipedia.org/wiki/Artificial_intelligence
https://docs.python.org/3/tutorial/introduction.html

# Replace this with a real YouTube video URL before YouTube ingestion testing
https://www.youtube.com/watch?v=REPLACE_WITH_REAL_VIDEO_ID
""", encoding="utf-8")

print("All test files created successfully!")

# --- PDF samples ---
(BASE / "pdf").mkdir(parents=True, exist_ok=True)

# text_pdf.pdf — real text layer
c = canvas.Canvas(str(BASE / "pdf" / "text_pdf.pdf"), pagesize=letter)
c.setFont("Helvetica-Bold", 16)
c.drawString(72, 720, "Introduction to Machine Learning")
c.setFont("Helvetica", 11)
lines = [
    "Machine learning is a subset of artificial intelligence that enables",
    "systems to learn patterns from data without being explicitly programmed.",
    "",
    "There are three main types: supervised, unsupervised, and reinforcement",
    "learning. Supervised learning uses labeled data to train models.",
]
y = 690
for line in lines:
    c.drawString(72, y, line)
    y -= 18
c.save()

# column_topic_pdf.pdf — two-column layout
c = canvas.Canvas(str(BASE / "pdf" / "column_topic_pdf.pdf"), pagesize=letter)
c.setFont("Helvetica-Bold", 16)
c.drawCentredString(300, 740, "Data Structures Overview")
c.setFont("Helvetica-Bold", 12)
c.drawString(60, 700, "Arrays")
c.setFont("Helvetica", 9)
c.drawString(60, 684, "Contiguous memory, O(1) access by index.")
c.setFont("Helvetica-Bold", 12)
c.drawString(320, 700, "Stacks")
c.setFont("Helvetica", 9)
c.drawString(320, 684, "LIFO order. Push/pop are O(1).")
c.save()

# scan_pdf.pdf — image-only page (NO text layer, forces OCR)
img = Image.new("RGB", (1240, 1754), "white")
draw = ImageDraw.Draw(img)
draw.text((80, 100), "Scanned Document Sample", font=font, fill="black")
draw.text((80, 220), "This page has no real text layer - image only.", font=font, fill="black")
draw.text((80, 270), "Photosynthesis converts light energy into glucose.", font=font, fill="black")
img.save(str(BASE / "pdf" / "scan_pdf.pdf"), "PDF", resolution=150.0)

# image_pdf.pdf — second image-only page, different content
img2 = Image.new("RGB", (1240, 1754), "white")
draw2 = ImageDraw.Draw(img2)
draw2.text((80, 100), "Embedded Image Content", font=font, fill="black")
draw2.text((80, 220), "Water boils at 100 degrees Celsius at sea level.", font=font, fill="black")
img2.save(str(BASE / "pdf" / "image_pdf.pdf"), "PDF", resolution=150.0)

# --- Missing images ---
img3 = Image.new("RGB", (900, 400), "white")
draw3 = ImageDraw.Draw(img3)
draw3.text((40, 40), "Clean Sample Image for OCR", font=font, fill="black")
draw3.text((40, 120), "The mitochondria is the powerhouse of the cell.", font=font, fill="black")
img3.save(BASE / "images" / "sample.png")

import random
img4 = Image.new("RGB", (900, 400), (245, 245, 240))
draw4 = ImageDraw.Draw(img4)
draw4.text((40, 40), "Lower Quality Scan Simulation", font=font, fill=(30, 30, 30))
draw4.text((40, 120), "Gravity accelerates objects at 9.8 m/s^2.", font=font, fill=(40, 40, 40))
pixels = img4.load()
for _ in range(15000):
    x, y = random.randint(0, 899), random.randint(0, 399)
    pixels[x, y] = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
img4.save(BASE / "images" / "scanned.jpg", quality=70)

print("PDF + missing image samples created.")