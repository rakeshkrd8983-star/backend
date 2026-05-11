from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import re

from rapidfuzz import fuzz
from googletrans import Translator
from langdetect import detect

from fastapi.middleware.cors import CORSMiddleware

# =========================
# FastAPI
# =========================
app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Translator
# =========================
translator = Translator()

@app.get("/")
def home():
    return {"status": "API Running"}

# =========================
# Safe Context
# =========================
safe_context = [
    "game",
    "match",
    "pubg",
    "free fire",
    "cricket",
    "football",
    "chess",
    "play",
    "tournament",
    "gaming",
    "battle",
    "competition",
    "score"
]

# =========================
# Safe Tone
# =========================
safe_tone = [
    "😂",
    "🤣",
    "lol",
    "haha",
    "hehe",
    "jk",
    "❤️",
    "🎉"
]

# =========================
# Aggressive Emojis
# =========================
aggressive_emojis = [
    "💀",
    "🔪",
    "😡",
    "🤬",
    "☠️",
    "💣",
    "🔥",
    "👿"
]

# =========================
# Indirect Threat Patterns
# =========================
indirect_patterns = [
    "you will regret",
    "you'll regret",
    "you will pay",
    "you'll pay",
    "this won't end well",
    "this wont end well"
]

# =========================
# Threat Keywords
# =========================
danger_words = [
    "kill",
    "murder",
    "attack",
    "destroy",
    "bomb",
    "shoot",
    "die",
    "stab",
    "slaughter",
    "rape",
    "burn",
    "terror",
    "explode"
]

# =========================
# Load Abuse Words
# =========================
def load_abuse_words():

    with open(
        "abuse.txt",
        "r",
        encoding="utf-8"
    ) as f:

        return [
            line.strip().lower()
            for line in f
            if line.strip()
        ]

abuse_words = load_abuse_words()

# =========================
# Language Detection
# =========================
def is_english(text):

    try:

        return detect(text) == "en"

    except:

        return True

# =========================
# Translation
# =========================
def translate_to_english(text):

    try:

        translated = translator.translate(
            text,
            dest="en"
        )

        return translated.text.lower()

    except Exception as e:

        print("Translation Error:", e)

        return text.lower()

# =========================
# Helpers
# =========================
def contains_abuse(text):

    words = re.findall(
        r'\b\w+\b',
        text.lower()
    )

    return any(
        word in abuse_words
        for word in words
    )

# =========================
# Fuzzy Abuse Detection
# =========================
def fuzzy_abuse_check(
    text,
    threshold=85
):

    words = re.findall(
        r'\b\w+\b',
        text.lower()
    )

    for word in words:

        for abuse in abuse_words:

            if (
                fuzz.ratio(word, abuse)
                >= threshold
            ):

                return True

    return False

# =========================
# Safe Context
# =========================
def is_safe_context(text):

    return any(
        word in text.lower()
        for word in safe_context
    )

# =========================
# Safe Tone
# =========================
def is_safe_tone(text):

    return any(
        word in text
        for word in safe_tone
    )

# =========================
# Aggressive Emojis
# =========================
def contains_aggressive_emoji(text):

    return any(
        emoji in text
        for emoji in aggressive_emojis
    )

# =========================
# Indirect Threat
# =========================
def is_indirect_threat(text):

    return any(
        pattern in text.lower()
        for pattern in indirect_patterns
    )

# =========================
# Explicit Threat
# =========================
def is_explicit_threat(text):

    lower = text.lower()

    return (

        "i will" in lower

        and

        any(
            word in lower
            for word in [
                "kill",
                "destroy",
                "beat",
                "shoot",
                "stab"
            ]
        )

    )

# =========================
# Danger Words
# =========================
def contains_danger_word(text):

    lower = text.lower()

    return any(
        word in lower
        for word in danger_words
    )

# =========================
# Severity Score
# =========================
def severity_score(text):

    score = 0

    lower = text.lower()

    if "kill" in lower:
        score += 3

    if "destroy" in lower:
        score += 3

    if "bomb" in lower:
        score += 3

    if "shoot" in lower:
        score += 3

    if "fuck" in lower:
        score += 2

    if "i will" in lower:
        score += 2

    if contains_aggressive_emoji(lower):
        score += 2

    return score

# =========================
# Load Model
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    "model"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "model"
)

model.eval()

# =========================
# Input Schema
# =========================
class InputText(BaseModel):
    text: str

# =========================
# Prediction Function
# =========================
def predict_model(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

    threat_prob = float(probs[0][1])

    threshold = 0.85

    pred = (
        1
        if threat_prob > threshold
        else 0
    )

    return pred, threat_prob

# =========================
# API Endpoint
# =========================
@app.post("/predict")
def predict(data: InputText):

    original_text = data.text.strip()

    # =========================
    # Ignore Empty
    # =========================
    if len(original_text) < 2:

        return {
            "input": original_text,
            "prediction": "Normal",
            "confidence": 0.99,
            "severity": "Low",
            "source": "filter",
            "reason": "Empty or invalid text"
        }

    # =========================
    # Translation Layer
    # =========================
    if is_english(original_text):

        text = original_text.lower()

    else:

        text = translate_to_english(
            original_text
        )

    print("Translated:", text)

    # =========================
    # Safe Context
    # =========================
    if is_safe_context(text):

        return {
            "input": original_text,
            "translated_text": text,
            "prediction": "Normal",
            "confidence": 0.95,
            "severity": "Low",
            "source": "context-rule",
            "reason": "Detected gaming/sports context"
        }

    # =========================
    # Safe Tone
    # =========================
    if is_safe_tone(text):

        return {
            "input": original_text,
            "translated_text": text,
            "prediction": "Normal",
            "confidence": 0.90,
            "severity": "Low",
            "source": "tone-rule",
            "reason": "Detected joking tone"
        }

    # =========================
    # Explicit Threat
    # =========================
    if is_explicit_threat(text):

        return {
            "input": original_text,
            "translated_text": text,
            "prediction": "Threat",
            "confidence": 0.99,
            "severity": "High",
            "source": "rule-based",
            "reason": "Explicit threat detected"
        }

    # =========================
    # Aggressive Emojis
    # =========================
    if contains_aggressive_emoji(text):

        return {
            "input": original_text,
            "translated_text": text,
            "prediction": "Toxic",
            "confidence": 0.90,
            "severity": "Medium",
            "source": "emoji-rule",
            "reason": "Aggressive emojis detected"
        }

    # =========================
    # Abuse Detection
    # =========================
    if (
        contains_abuse(text)
        or
        fuzzy_abuse_check(text)
    ):

        return {
            "input": original_text,
            "translated_text": text,
            "prediction": "Abuse",
            "confidence": 0.90,
            "severity": "Medium",
            "source": "rule-abuse",
            "reason": "Abusive language detected"
        }

    # =========================
    # Model Prediction
    # =========================
    pred, confidence = predict_model(
        text
    )

    # =========================
    # Indirect Threat
    # =========================
    if (

        is_indirect_threat(text)

        and

        confidence > 0.30

    ):

        return {
            "input": original_text,
            "translated_text": text,
            "prediction": "Threat",
            "confidence": round(confidence, 3),
            "severity": "Medium",
            "source": "hybrid-rule",
            "reason": "Indirect threat detected"
        }

    # =========================
    # Final Threat Check
    # =========================
    if (

        pred == 1

        and

        contains_danger_word(text)

    ):

        return {
            "input": original_text,
            "translated_text": text,
            "prediction": "Threat",
            "confidence": round(confidence, 3),
            "severity": "High",
            "source": "model",
            "reason": "Model detected threat intent"
        }

    # =========================
    # Default Normal
    # =========================
    return {
        "input": original_text,
        "translated_text": text,
        "prediction": "Normal",
        "confidence": round(confidence, 3),
        "severity": "Low",
        "source": "model",
        "reason": "No harmful intent detected"
    }