import os
import random
import string
import subprocess
import time
import shutil

# ---------------------------
# CONFIGURATION
# ---------------------------
OUTPUT_DIR = r"D:\Assignments\Assignment2\latex_pdfs"
IMAGE_PATH = r"D:\Assignments\Assignment2\sample_image.jpg"  # local image
START_INDEX = 1
END_INDEX = 5000

IMAGE_PROBABILITY = 0.3  # 30% PDFs have images
SHORT_DOC_PROB = 0.5     # 50% PDFs are 2-3 pages

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def random_text(paragraphs=3):
    text = ""
    for _ in range(paragraphs):
        words = ''.join(random.choices(string.ascii_lowercase + ' ', k=400))
        text += " ".join(words.split()) + ".\n\n"
    return text

def generate_latex_content(pages=2, include_image=False):
    """Generate LaTeX document content with optional local image."""
    content = [
        r"\documentclass[12pt]{article}",
        r"\usepackage{graphicx}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}"
    ]
    
    for _ in range(pages * 3):  # approx 3 paragraphs per page
        content.append(random_text(1))
    
    if include_image:
        # Use relative path for LaTeX
        content.append(r"\begin{center}")
        content.append(rf"\includegraphics[width=0.5\textwidth]{{{IMAGE_PATH.replace(os.sep, '/')}}}")
        content.append(r"\end{center}")
    
    content.append(r"\end{document}")
    return "\n".join(content)

# ---------------------------
# MAIN LOOP
# ---------------------------
for i in range(START_INDEX, END_INDEX + 1):
    try:
        short_doc = random.random() < SHORT_DOC_PROB
        pages = random.choice([2, 3]) if short_doc else random.choice([4, 5, 6])
        include_image = random.random() < IMAGE_PROBABILITY

        tex_filename = os.path.join(OUTPUT_DIR, f"LatexDoc_{i}.tex")
        pdf_filename = os.path.join(OUTPUT_DIR, f"LatexDoc_{i}.pdf")

        # Write LaTeX file
        with open(tex_filename, "w", encoding="utf-8") as f:
            f.write(generate_latex_content(pages, include_image))

        # Compile PDF using pdflatex
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            cwd=OUTPUT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Delete extra files (.tex, .aux, .log)
        for ext in [".tex", ".aux", ".log"]:
            file_to_delete = os.path.join(OUTPUT_DIR, f"LatexDoc_{i}{ext}")
            if os.path.exists(file_to_delete):
                os.remove(file_to_delete)

        print(f"✅ Generated: LatexDoc_{i}.pdf ({pages} pages, image={include_image})")
        time.sleep(0.5)  # avoid overload

    except Exception as e:
        print(f"❌ Error in file {i}: {e}")
        time.sleep(5)
