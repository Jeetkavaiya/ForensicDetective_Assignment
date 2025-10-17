import os
import time
import random
import string
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials

# ---------------------------
# CONFIGURATION
# ---------------------------
OUTPUT_DIR = r"D:\Assignments\Assignment2\google_docs_pdfs"
START_INDEX = 5001
END_INDEX = 6000  # you can change range

# Your existing public Drive image URL
PUBLIC_IMAGE_URL = "https://drive.google.com/file/d/14h1ku0p-yZvl6DFiP4awA9BTlxPr2Uf8/view"

# ---------------------------
# SETUP
# ---------------------------
creds = Credentials.from_authorized_user_file(
    "token.json",
    ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"]
)
docs_service = build("docs", "v1", credentials=creds)
drive_service = build("drive", "v3", credentials=creds)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def convert_drive_url_to_direct(url: str) -> str:
    """Convert a Google Drive share link to a direct viewable URL."""
    if "file/d/" in url:
        file_id = url.split("file/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=view&id={file_id}"
    return url

def random_text(paragraphs=3):
    """Generate random filler text."""
    text = ""
    for _ in range(paragraphs):
        words = ''.join(random.choices(string.ascii_lowercase + ' ', k=400))
        text += " ".join(words.split()) + ".\n\n"
    return text

def create_google_doc_with_content(title, image_url, pages=2):
    """Create a Google Doc with text + image from a direct URL."""
    doc = docs_service.documents().create(body={"title": title}).execute()
    doc_id = doc.get("documentId")

    requests = []
    num_pages = random.choice([2, 3])
    for _ in range(num_pages * 3):
        requests.append({"insertText": {"location": {"index": 1}, "text": random_text(1)}})

    # Insert image (converted to direct link)
    requests.append({
        "insertInlineImage": {
            "location": {"index": 1},
            "uri": image_url
        }
    })

    docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
    return doc_id

def export_doc_as_pdf(doc_id, filename):
    """Export a Google Doc as a PDF."""
    request = drive_service.files().export_media(fileId=doc_id, mimeType="application/pdf")
    fh = io.FileIO(filename, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()

# ---------------------------
# MAIN LOOP
# ---------------------------
DIRECT_IMAGE_URL = convert_drive_url_to_direct(PUBLIC_IMAGE_URL)
print(f"✅ Using direct image link:\n{DIRECT_IMAGE_URL}")

for i in range(START_INDEX, END_INDEX + 1):
    try:
        title = f"GoogleDoc_{i}"
        pages = random.choice([2, 3])
        doc_id = create_google_doc_with_content(title, DIRECT_IMAGE_URL, pages)

        output_path = os.path.join(OUTPUT_DIR, f"{title}.pdf")
        export_doc_as_pdf(doc_id, output_path)

        print(f"✅ Saved: {output_path} ({pages} pages)")
        time.sleep(2)

    except Exception as e:
        print(f"❌ Error at {i}: {e}")
        time.sleep(5)
