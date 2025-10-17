import os
import PyPDF2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = r"D:\Assignments\Assignment2\data"
FOLDERS = ["word_pdfs", "google_docs_pdfs", "Latex_pdfs", "python_pdfs"]

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def extract_metadata_features(pdf_path):
    """Extract numeric metadata features that do not trivially reveal type."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            info = reader.metadata

            # Feature 1: Number of pages
            num_pages = len(reader.pages)

            # Feature 2: File size (KB)
            file_size = os.path.getsize(pdf_path) / 1024

            # Feature 3 & 4: Creation & Mod date in days since epoch (truncated)
            # If missing, fill with -1
            creation_date = info.get("/CreationDate")
            mod_date = info.get("/ModDate")

            def parse_pdf_date(d):
                if d is None:
                    return -1
                # PDF date format: D:YYYYMMDDHHmmSS
                try:
                    y = int(d[2:6])
                    m = int(d[6:8])
                    day = int(d[8:10])
                    return y*365 + m*30 + day  # rough numeric representation
                except:
                    return -1

            creation_days = parse_pdf_date(creation_date)
            mod_days = parse_pdf_date(mod_date)

            # Return as numpy array
            return np.array([num_pages, file_size, creation_days, mod_days])

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

# ----------------------------
# BUILD DATASET
# ----------------------------
X = []
y = []

for label, folder in enumerate(FOLDERS):
    folder_path = os.path.join(DATA_DIR, folder)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, fname)
            features = extract_metadata_features(pdf_path)
            if features is not None:
                X.append(features)
                y.append(folder)

X = np.array(X)
y = np.array(y)

# ----------------------------
# TRAIN/TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ----------------------------
# STANDARDIZE FEATURES
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# SGD CLASSIFIER
# ----------------------------
clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)

# ----------------------------
# EVALUATE
# ----------------------------
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------------------
# CONFUSION MATRIX PLOT
# ----------------------------
cm = confusion_matrix(y_test, y_pred, labels=FOLDERS)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=FOLDERS, yticklabels=FOLDERS)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
