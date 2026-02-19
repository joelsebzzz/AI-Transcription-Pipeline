import re
import unicodedata
from typing import Optional

ENGLISH_FILLERS = {
    "um", "uh", "hmm", "hm", "mhm", "uh-huh", "uhh",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "literally", "actually", "honestly",
}

HALLUCINATION_PATTERNS = [
    r"\[.*?\]",
    r"\(.*?\)",
    r"(?i)thank you for watching",
    r"(?i)subscribe to my channel",
    r"(?i)like and subscribe",
    r"(?i)www\.\S+",
]


def clean_transcript(text: str, remove_fillers: bool = True, remove_hallucinations: bool = True,
                     fix_encoding: bool = True, collapse_whitespace: bool = True,
                     language: Optional[str] = None) -> str:
    if not text or not text.strip():
        return ""
    if fix_encoding:
        text = _fix_unicode(text)
    if remove_hallucinations:
        text = _remove_hallucinations(text)
    if remove_fillers:
        text = _remove_fillers(text, language=language)
    text = _remove_repeated_phrases(text)
    if collapse_whitespace:
        text = _collapse_whitespace(text)
    return text.strip()


def clean_segments(segments: list[dict], **kwargs) -> list[dict]:
    cleaned = []
    for seg in segments:
        cleaned_text = clean_transcript(seg.get("text", ""), **kwargs)
        if cleaned_text:
            cleaned.append({**seg, "text": cleaned_text})
    return cleaned


def _fix_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufffd", "")
    return text


def _remove_hallucinations(text: str) -> str:
    for pattern in HALLUCINATION_PATTERNS:
        text = re.sub(pattern, "", text)
    return text


def _remove_fillers(text: str, language: Optional[str] = None) -> str:
    if language and language != "en":
        return text
    fillers_sorted = sorted(ENGLISH_FILLERS, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(f) for f in fillers_sorted) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def _remove_repeated_phrases(text: str) -> str:
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+ \w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    return text


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
