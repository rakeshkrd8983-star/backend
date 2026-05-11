from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

from fastapi.middleware.cors import CORSMiddleware

from rapidfuzz import fuzz
from deep_translator import GoogleTranslator
from langdetect import detect

import torch
import re

# =========================================
# FastAPI
# =========================================
app = FastAPI()

# =========================================
# CORS
# =========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# Optimize CPU Usage
# =========================================
torch.set_num_threads(1)

# =========================================
# Home Route
# =========================================
@app.get("/")
def home():
    return {"status": "API Running"}

# =========================================
# Load Abuse Words
# =========================================
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

# =========================================
# Safe Context
# =========================================
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

# =========================================
# Safe Tone
# =========================================
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

# =========================================
# Aggressive Emojis
# =========================================
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

# =========================================
# Indirect Threat Patterns
# =========================================
indirect_patterns = [
    "you will regret",
    "you'll regret",
    "you will pay",
    "you'll pay",
    "this won't end well",
    "this wont end well"
]

# =========================================
# Threat Keywords
# =========================================
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

# =========================================
# Translator
# =========================================
translator = GoogleTranslator(
    source='auto',
    target='en'
)

# =========================================
# Language Detection
# =========================================
def is_english(text):

    try:
        return detect(text) == "en"

    except:
        return True

# =========================================
# Translation
# =========================================
def translate_to_english(text):

    try:
        translated = translator.translate(text)
        return translated.lower()

    except Exception as e:

        print("Translation Error:", e)

        return text.lower()

# =========================================
# Abuse Detection
# =========================================
def contains_abuse(text):

    words = re.findall(
        r'\b\w+\b',
        text.lower()
    )

    return any(
        word in abuse_words
        for word in words
    )

# =========================================
# Fuzzy Abuse Detection
# =========================================
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

            if fuzz.ratio(word, abuse) >= threshold:
                return True

    return False

# =========================================
# Context Rules
# =========================================
def is_safe_context(text):

    return any(
        word in text.lower()
        for word in safe_context
    )

def is_safe_tone(text):

    return any(
        word in text
        for word in safe_tone
    )

def contains_aggressive_emoji(text):

    return any(
        emoji in text
        for emoji in aggressive_emojis
    )

def is_indirect_threat(text):

    return any(
        pattern in text.lower()
        for pattern in indirect_patterns
    )

def contains_danger_word(text):

    return any(
        word in text.lower()
        for word in danger_words
    )

# =========================================
# Load Model
# =========================================
tokenizer = None
model = None

def load_model():

    global tokenizer
    global model

    if tokenizer is None or model is None:

        print("Loading model...")

        tokenizer = BertTokenizer.from_pretrained(
            "rakeshkrd/bert-threat-detection"
        )

        model = BertForSequenceClassification.from_pretrained(
            "rakeshkrd/bert-threat-detection",
            low_cpu_mem_usage=True
        )

        model.eval()

        print("Model loaded.")

# =========================================
# Input Schema
# =========================================
class InputText(BaseModel):
    text: str

# =========================================
# Prediction Function
# =========================================
def predict_model(text):

    load_model()

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

# =========================================
# Predict API
# =========================================
@app.post("/predict")
async def predict(data: InputText):

    try:

        original_text = data.text.strip()

        if len(original_text) < 2:

            return {
                "prediction": "Normal",
                "confidence": 0.99,
                "severity": "Low",
                "reason": "Empty or invalid text"
            }

        # =====================================
        # Translation
        # =====================================
        if is_english(original_text):

            text = original_text.lower()

        else:

            text = translate_to_english(
                original_text
            )

        print("Processed:", text)

        # =====================================
        # Safe Context
        # =====================================
        if is_safe_context(text):

            return {
                "prediction": "Normal",
                "confidence": 0.95,
                "severity": "Low",
                "reason": "Gaming/Sports context detected"
            }

        # =====================================
        # Safe Tone
        # =====================================
        if is_safe_tone(text):

            return {
                "prediction": "Normal",
                "confidence": 0.90,
                "severity": "Low",
                "reason": "Joking tone detected"
            }

        # =====================================
        # Emoji Aggression
        # =====================================
        if contains_aggressive_emoji(text):

            return {
                "prediction": "Toxic",
                "confidence": 0.90,
                "severity": "Medium",
                "reason": "Aggressive emoji detected"
            }

        # =====================================
        # Abuse Detection
        # =====================================
        if (
            contains_abuse(text)
            or fuzzy_abuse_check(text)
        ):

            return {
                "prediction": "Abuse",
                "confidence": 0.90,
                "severity": "Medium",
                "reason": "Abusive language detected"
            }

        # =====================================
        # Model Prediction
        # =====================================
        pred, confidence = predict_model(text)

        # =====================================
        # Indirect Threat
        # =====================================
        if (
            is_indirect_threat(text)
            and confidence > 0.30
        ):

            return {
                "prediction": "Threat",
                "confidence": round(confidence, 3),
                "severity": "Medium",
                "reason": "Indirect threat detected"
            }

        # =====================================
        # Final Threat
        # =====================================
        if (
            pred == 1
            and contains_danger_word(text)
        ):

            return {
                "prediction": "Threat",
                "confidence": round(confidence, 3),
                "severity": "High",
                "reason": "Threat intent detected"
            }

        # =====================================
        # Default Safe
        # =====================================
        return {
            "prediction": "Normal",
            "confidence": round(confidence, 3),
            "severity": "Low",
            "reason": "No harmful intent detected"
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "error": str(e)
        }
