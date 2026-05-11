import pandas as pd
import random

# =========================
# Load dataset
# =========================
df = pd.read_csv("train.csv")

# =========================
# Map labels
# =========================
def map_label(row):
    if row["threat"] == 1:
        return 2   # Threat
    elif row["toxic"] == 1 or row["insult"] == 1:
        return 1   # Abuse
    else:
        return 0   # Normal

df["label"] = df.apply(map_label, axis=1)

# Keep required columns
df = df[["comment_text", "label"]]
df.columns = ["text", "label"]

print("Original counts:")
print(df["label"].value_counts())

# =========================
# Split classes
# =========================
df_threat = df[df["label"] == 2]
df_abuse = df[df["label"] == 1]
df_normal = df[df["label"] == 0]

# =========================
# 🔥 Data Augmentation (Threat)
# =========================
def augment_text(text):
    variations = [
        text,
        text + " bro",
        "abe " + text,
        text + " 😡",
        text.replace("you", "u"),
        text.replace("kill", "k!ll"),
        text + " idiot",
        "listen " + text
    ]
    return list(set(variations))  # remove duplicates

augmented_threats = []

for t in df_threat["text"]:
    augmented_threats.extend(augment_text(str(t)))

df_threat_aug = pd.DataFrame({
    "text": augmented_threats,
    "label": 2
})

# Combine original + augmented threats
df_threat = pd.concat([df_threat, df_threat_aug])

# =========================
# Balance dataset
# =========================
threat_count = len(df_threat)

df_abuse = df_abuse.sample(min(len(df_abuse), threat_count), random_state=42)
df_normal = df_normal.sample(min(len(df_normal), threat_count), random_state=42)

df_balanced = pd.concat([df_threat, df_abuse, df_normal])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced counts:")
print(df_balanced["label"].value_counts())

# =========================
# Final size control
# =========================
target_size = min(10000, len(df_balanced))
df_final = df_balanced.sample(target_size, random_state=42).reset_index(drop=True)

print("\nFinal shape:", df_final.shape)

# =========================
# Save dataset
# =========================
df_final.to_csv("processed.csv", index=False)

print("\n✅ processed.csv saved successfully!")