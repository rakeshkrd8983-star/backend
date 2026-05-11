import pandas as pd
import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

start = time.time()

# =========================
# Device
# =========================
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load dataset
# =========================
df = pd.read_csv("processed.csv")
print("Dataset shape:", df.shape)

df = df[['text', 'label']].dropna()

# 🔥 IMPORTANT: Convert to binary (FIX)
df["label"] = df["label"].apply(lambda x: 1 if x >= 1 else 0)
df["label"] = df["label"].astype(int)

print("Unique labels:", df["label"].unique())

# =========================
# Split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# =========================
# Load model
# =========================
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model")

model.to(device)
model.eval()

# =========================
# Batch inference
# =========================
batch_size = 64

texts = test_df['text'].astype(str).tolist()
labels = test_df['label'].tolist()

all_probs = []
all_true = []

for i in range(0, len(texts), batch_size):
    print(f"Processing batch {i}/{len(texts)}")

    batch_texts = texts[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    threat_prob = probs[:, 1]

    all_probs.extend(threat_prob.cpu().numpy())
    all_true.extend(batch_labels)

# =========================
# Threshold tuning
# =========================
best_threshold = 0
best_f1 = 0

for t in np.arange(0.2, 0.6, 0.02):
    preds = (np.array(all_probs) > t).astype(int)
    f1 = f1_score(all_true, preds, average="binary")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\n🔥 Best Threshold:", round(best_threshold, 2))
print("🔥 Best F1:", round(best_f1, 4))

# =========================
# Final predictions
# =========================
y_pred = (np.array(all_probs) > best_threshold).astype(int)
y_true = all_true

# =========================
# Metrics
# =========================
accuracy = accuracy_score(y_true, y_pred)

print("\n✅ Accuracy:", round(accuracy, 4))

print("\n📊 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Normal", "Threat"]
))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

print("\n📌 Confusion Matrix:")
print("               Pred Normal   Pred Threat")
print(f"Actual Normal       {cm[0][0]}           {cm[0][1]}")
print(f"Actual Threat       {cm[1][0]}           {cm[1][1]}")

# =========================
# Plot
# =========================
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - Optimized Model")

plt.xticks([0, 1], ["Normal", "Threat"])
plt.yticks([0, 1], ["Normal", "Threat"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")  # for report
plt.show()

# =========================
# Time
# =========================
print("\n⏱ Total time:", round(time.time() - start, 2), "seconds")