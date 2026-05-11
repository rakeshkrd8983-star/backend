from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
from utils import clean_text

# =========================
# Load dataset
# =========================
df = pd.read_csv("processed.csv")

# 🔥 Clean text
df["text"] = df["text"].apply(clean_text)

# 🔥 Ensure binary labels
df["label"] = df["label"].apply(lambda x: 1 if x >= 1 else 0)
df["label"] = df["label"].astype(int)

# =========================
# Stratified split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)

# =========================
# Class weights (CRITICAL 🔥)
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights = torch.tensor(class_weights, dtype=torch.float)

# =========================
# Convert to HF dataset
# =========================
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =========================
# Model (FIXED 🔥)
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/bertweet-base",
    num_labels=2   # 🔥 FIX
)

# =========================
# Custom Trainer (weighted loss)
# =========================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# =========================
# Metrics (VERY IMPORTANT 🔥)
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =========================
# Training args
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50
)

# =========================
# Trainer
# =========================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# =========================
# Train
# =========================
trainer.train()

# =========================
# Save
# =========================
model.save_pretrained("model")
tokenizer.save_pretrained("model")