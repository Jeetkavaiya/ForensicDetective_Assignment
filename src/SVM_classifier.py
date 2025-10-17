import os
import PyPDF2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 📁 Folder paths
base_dir = r"D:\Assignments\Assignment2\data"
folders = {
    "word": "word_pdfs",
    "google_docs": "google_docs_pdfs",
    "latex": "latex_pdfs",
    "python": "python_pdfs"
}

# 🧩 Extract metadata
data = []
for label, folder in folders.items():
    folder_path = os.path.join(base_dir, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            try:
                reader = PyPDF2.PdfReader(path)
                info = reader.metadata
                num_pages = len(reader.pages)
                size = os.path.getsize(path)

                data.append({
                    "producer": str(info.get("/Producer", "")),
                    "creator": str(info.get("/Creator", "")),
                    "creation_date": str(info.get("/CreationDate", ""))[:8],
                    "num_pages": num_pages,
                    "file_size": size,
                    "label": label
                })
            except Exception as e:
                print(f"Error reading {path}: {e}")

# 📊 DataFrame
df = pd.DataFrame(data)
print(df.head())

# 🔠 Encode categorical fields
for col in ["producer", "creator", "creation_date"]:
    df[col] = LabelEncoder().fit_transform(df[col])

# ✂️ Split
X = df[["creation_date", "num_pages", "file_size"]]  # Drop producer/creator
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel="rbf", C=1, gamma="scale")
svm.fit(X_train_scaled, y_train)

y_pred = svm.predict(X_test_scaled)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=svm.classes_,
            yticklabels=svm.classes_)
plt.title("Confusion Matrix (SVM, Metadata Only)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
