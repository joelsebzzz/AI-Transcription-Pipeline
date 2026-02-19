import os
import tempfile
import traceback
from flask import Flask, request, jsonify, render_template

from transcriber import Transcriber, SeamlessTranscriber
from summarizer  import Summarizer
from cleaner     import clean_transcript
from chunker     import chunk_text, chunk_segments

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

_whisper_transcriber  = None
_seamless_transcriber = None
_summarizer           = None
_loaded_whisper_size  = None
_loaded_seamless_key  = None


def get_transcriber(model_size: str = "small") -> Transcriber:
    global _whisper_transcriber, _loaded_whisper_size
    if _whisper_transcriber is None or _loaded_whisper_size != model_size:
        _whisper_transcriber = Transcriber(model_size=model_size)
        _loaded_whisper_size = model_size
    return _whisper_transcriber


def get_seamless_transcriber(model_key: str = "seamless-v2-large") -> SeamlessTranscriber:
    global _seamless_transcriber, _loaded_seamless_key
    if _seamless_transcriber is None or _loaded_seamless_key != model_key:
        import psutil, os
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        required_gb  = 12 if "v2" in model_key else 6
        if available_gb < required_gb:
            raise MemoryError(
                f"Not enough RAM to load SeamlessM4T ({model_key}). "
                f"Required: ~{required_gb}GB free — Available: {available_gb:.1f}GB. "
                f"Use Whisper instead, or run on a machine with more RAM."
            )
        _seamless_transcriber = SeamlessTranscriber(model_key=model_key)
        _loaded_seamless_key  = model_key
    return _seamless_transcriber


def get_summarizer(model_name: str = "t5-small") -> Summarizer:
    global _summarizer
    if _summarizer is None or _summarizer.config["model"] != model_name:
        _summarizer = Summarizer(model_name)
    return _summarizer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcribe/whisper", methods=["POST"])
def transcribe_whisper():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file   = request.files["audio"]
        language     = request.form.get("language", "auto")
        whisper_size = request.form.get("whisper_model", "small")

        if language == "auto":
            language = None

        suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        wav_path    = _ensure_wav(tmp_path)
        transcriber = get_transcriber(whisper_size)
        result      = transcriber.transcribe(wav_path, language=language)
        cleaned     = clean_transcript(result["text"], language=result["language"])

        os.unlink(tmp_path)
        if wav_path != tmp_path:
            os.unlink(wav_path)

        return jsonify({
            "transcript":     cleaned,
            "raw_transcript": result["text"],
            "language":       result["language"],
            "segments":       result["segments"],
            "model_used":     f"whisper-{whisper_size}",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcribe/seamless", methods=["POST"])
def transcribe_seamless():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file  = request.files["audio"]
        language    = request.form.get("language", "auto")
        model_key   = request.form.get("seamless_model", "seamless-v2-large")

        if language == "auto":
            language = None

        suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        wav_path    = _ensure_wav(tmp_path)
        transcriber = get_seamless_transcriber(model_key)
        result      = transcriber.transcribe(wav_path, language=language)
        cleaned     = clean_transcript(result["text"], language=result.get("language"))

        os.unlink(tmp_path)
        if wav_path != tmp_path:
            os.unlink(wav_path)

        return jsonify({
            "transcript":     cleaned,
            "raw_transcript": result["text"],
            "language":       result.get("language", "unknown"),
            "segments":       [],
            "model_used":     model_key,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize():
    try:
        data     = request.get_json()
        text     = data.get("text", "").strip()
        model    = data.get("model", "t5-small")
        language = data.get("language", "en")
        segments = data.get("segments", [])

        if not text:
            return jsonify({"error": "No text provided"}), 400
        if len(text.split()) < 20:
            return jsonify({"error": "Text too short (need at least 20 words)"}), 400

        chunks = chunk_segments(segments, max_tokens=400) if segments else chunk_text(text, max_tokens=400)
        if not chunks:
            chunks = [text]

        summarizer = get_summarizer(model)
        result     = summarizer.summarize(chunks)

        return jsonify({
            "summary":         result["final_summary"],
            "chunk_summaries": result["chunk_summaries"],
            "num_chunks":      len(chunks),
            "model_used":      result["model_used"],
            "time_seconds":    result["time_seconds"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/detect-language", methods=["POST"])
def detect_language():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400
        audio_file = request.files["audio"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        wav_path    = _ensure_wav(tmp_path)
        transcriber = get_transcriber("small")
        detected    = transcriber.detect_language_only(wav_path)
        os.unlink(tmp_path)
        if wav_path != tmp_path:
            os.unlink(wav_path)
        return jsonify({"language": detected})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _ensure_wav(path: str) -> str:
    if path.endswith(".wav"):
        return path
    out_path = path.rsplit(".", 1)[0] + ".wav"
    ret = os.system(f'ffmpeg -y -i "{path}" -ar 16000 -ac 1 "{out_path}" -loglevel quiet')
    if ret != 0:
        print("⚠  ffmpeg not found. Passing original file to Whisper.")
        return path
    return out_path


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🎙  Multilingual Speech-to-Summary — Web UI")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)