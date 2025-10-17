import os
import random
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# ---------------------------
# CONFIGURATION
# ---------------------------
OUTPUT_DIR = r"D:\Assignments\Assignment2\python_pdfs"
IMAGE_PATH = r"D:\Assignments\Assignment2\sample_image.jpg"   # optional image path
NUM_PDFS = 5000
IMAGE_PROBABILITY = 0.3   # 30% of PDFs will have an image

# Page distribution
SHORT_DOC_PROB = 0.5  # 50% = 2–3 pages, 50% = 4–7 pages

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# PDF GENERATION FUNCTION
# ---------------------------
def generate_random_text():
    words = " ".join(
        "".join(random.choices("abcdefghijklmnopqrstuvwxyz ", k=800)).split()
    )
    return words.capitalize() + "."

def create_pdf(filename, include_image=False, pages=2):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    for page_num in range(pages):
        text = generate_random_text()
        text_obj = c.beginText(50, height - 100)
        text_obj.setFont("Helvetica", 12)

        for line in range(20):
            text_obj.textLine(generate_random_text())

        c.drawText(text_obj)

        if include_image and page_num == 0 and os.path.exists(IMAGE_PATH):
            img = ImageReader(IMAGE_PATH)
            img_width = 200
            img_height = 150
            c.drawImage(img, 50, height - 300, img_width, img_height)

        c.showPage()

    c.save()

# ---------------------------
# MAIN LOOP
# ---------------------------
for i in range(1, NUM_PDFS + 1):
    title = f"PythonPDF_{i}.pdf"
    file_path = os.path.join(OUTPUT_DIR, title)

    include_image = random.random() < IMAGE_PROBABILITY
    short_doc = random.random() < SHORT_DOC_PROB
    pages = random.choice([2, 3]) if short_doc else random.choice([4, 5, 6, 7])

    create_pdf(file_path, include_image, pages)
    print(f"✅ Created: {file_path} ({pages} pages{' + image' if include_image else ''})")

print("\n🎉 All PDFs generated successfully!")
