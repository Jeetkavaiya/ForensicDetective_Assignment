🧩 Project Setup Guide — PDF Classification Forensics

This document explains how to set up the environment, generate synthetic PDF datasets, and train classification models based solely on PDF metadata.

📦 1. Prerequisites

Before starting, ensure you have the following installed:

Python ≥ 3.9

pip (Python package manager)

Git

wkhtmltopdf — required by pdfkit for HTML-to-PDF conversion

Download for Windows

After installing, add its path (e.g. C:\Program Files\wkhtmltopdf\bin) to your system PATH.

MiKTeX or TeX Live — required for LaTeX PDF generation

Download MiKTeX

Make sure pdflatex works from Command Prompt.

(Optional) Google Cloud account with Google Docs API enabled (for Google Docs PDF generation)

⚙️ 2. Repository Setup
# Clone this repository
git clone https://github.com/gandharnole/ForensicsDetective_pdf_classification.git
cd ForensicsDetective_pdf_classification

# Install dependencies
pip install -r requirements.txt

📂 3. Directory Structure

Your working directory should look like this:

Assignment2/
│
├── data/
│   ├── word_pdfs/          # 5,000 PDFs generated via Microsoft Word
│   ├── google_docs_pdfs/   # 5,000 PDFs generated using Google Docs API
│   ├── latex_pdfs/         # 5,000 PDFs generated via pdflatex
│   ├── python_pdfs/        # 5,000 PDFs generated via ReportLab
│   └── html_pdfs/          # 5,000 PDFs converted from HTML using pdfkit
│
├── src/
│   ├── generate_word_pdfs.py
│   ├── google_docs_pdfs.py
│   ├── latex_pdfs.py
│   ├── python_pdfs_generator.py
│   ├── html_pdfs.py
│   └── metadata_classifier.py
│
├── requirements.txt
└── SETUP.md

🧠 4. Google Docs API Setup

If generating Google Docs PDFs:

Go to the Google Cloud Console
.

Create a new project → Enable Google Docs API and Google Drive API.

Create OAuth 2.0 credentials (Desktop App).

Download the credentials as credentials.json and save it in your project root.

The first time you run the script, it will ask for browser authorization and create token.json.

🧾 5. PDF Generation

Each script generates a unique corpus of synthetic PDFs:

# Example: Generate 5,000 LaTeX PDFs
python src/latex_pdfs.py


All generators include:

Randomized page count (1–3 pages)

30% chance of embedded image

Slight variations in text and layout to simulate realistic documents

🔍 6. Metadata-Based Classification

The model does not rely on text or layout — only on metadata such as:

File size

Number of pages

Creation date

Modification date
(Fields like Producer or Creator were intentionally excluded to prevent easy classification.)

Run the classification pipeline:

python src/metadata_classifier.py


It trains multiple models:

SVM

SGD Classifier

Random Forest

XGBoost

Each model outputs:

Accuracy, Precision, Recall, and F1-score

Confusion matrix visualization (confusion_matrix.png)

🧮 7. Results Summary

After training, models typically achieve 92–96% accuracy when using metadata alone — proving that even limited metadata contains distinctive patterns that can identify document origins.