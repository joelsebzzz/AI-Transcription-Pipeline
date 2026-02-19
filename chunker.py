import re

WORDS_PER_TOKEN = 1.3


def chunk_text(text: str, max_tokens: int = 400, overlap_sentences: int = 1) -> list[str]:
    if not text.strip():
        return []

    sentences = _split_into_sentences(text)
    if not sentences:
        return [text]

    if _estimate_tokens(text) <= max_tokens:
        return [text]

    chunks      = []
    current     = []
    current_tok = 0

    for sentence in sentences:
        sentence_tok = _estimate_tokens(sentence)
        if current_tok + sentence_tok > max_tokens and current:
            chunks.append(" ".join(current))
            if overlap_sentences > 0:
                current     = current[-overlap_sentences:]
                current_tok = sum(_estimate_tokens(s) for s in current)
            else:
                current     = []
                current_tok = 0
        current.append(sentence)
        current_tok += sentence_tok

    if current:
        chunks.append(" ".join(current))

    print(f"📄  Text split into {len(chunks)} chunk(s)")
    return chunks


def chunk_segments(segments: list[dict], max_tokens: int = 400) -> list[str]:
    chunks      = []
    current     = []
    current_tok = 0

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        seg_tok = _estimate_tokens(seg_text)
        if current_tok + seg_tok > max_tokens and current:
            chunks.append(" ".join(current))
            current     = []
            current_tok = 0
        current.append(seg_text)
        current_tok += seg_tok

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+', text)
    if len(raw) <= 2 and len(text.split()) > 60:
        return _split_by_word_count(text, words_per_chunk=50)
    return [s.strip() for s in raw if s.strip()]


def _split_by_word_count(text: str, words_per_chunk: int = 50) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * WORDS_PER_TOKEN)
