import re, emoji

def clean_text(text):
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text