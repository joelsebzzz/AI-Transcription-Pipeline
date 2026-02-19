import whisper
from typing import Optional

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ml": "Malayalam",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese",
    "ar": "Arabic",
}

SEAMLESS_LANG_MAP = {
    "en":   "eng",
    "ml":   "mal",
    "hi":   "hin",
    "ta":   "tam",
    "te":   "tel",
    "fr":   "fra",
    "de":   "deu",
    "es":   "spa",
    "zh":   "cmn",
    "ar":   "arb",
    "auto": "eng",
}

SEAMLESS_MODELS = {
    "seamless-v2-large":  "facebook/seamless-m4t-v2-large",
    "seamless-v1-large":  "facebook/hf-seamless-m4t-large",
    "seamless-v1-medium": "facebook/hf-seamless-m4t-medium",
}


class Transcriber:
    def __init__(self, model_size: str = "small"):
        print(f"⏳  Loading Whisper-{model_size} model ...")
        self.model      = whisper.load_model(model_size)
        self.model_size = model_size
        print(f"✅  Whisper-{model_size} loaded.")

    def transcribe(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe") -> dict:
        options = {"task": task, "fp16": False, "verbose": False}
        if language:
            options["language"] = language
            print(f"🌐  Transcribing: {SUPPORTED_LANGUAGES.get(language, language)}")
        else:
            print("🔍  Auto-detecting language ...")

        result   = self.model.transcribe(audio_path, **options)
        detected = result.get("language", "unknown")
        print(f"✅  Done. Detected: {detected}")

        return {
            "text":     result["text"].strip(),
            "language": detected,
            "segments": result.get("segments", []),
        }

    def transcribe_with_language_segments(self, audio_path: str, segments: list[dict], languages: list[str]) -> str:
        import soundfile as sf
        import tempfile, os

        audio_data, sample_rate = sf.read(audio_path)
        combined_text = []

        for i, (seg, lang) in enumerate(zip(segments, languages)):
            start_sample = int(seg["start"] * sample_rate)
            end_sample   = int(seg["end"]   * sample_rate)
            chunk        = audio_data[start_sample:end_sample]
            tmp          = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, chunk, sample_rate)
            print(f"   Segment {i+1}/{len(segments)} [{seg['start']:.1f}s–{seg['end']:.1f}s] → {SUPPORTED_LANGUAGES.get(lang, lang)}")
            result = self.transcribe(tmp.name, language=lang)
            combined_text.append(f"[{lang.upper()}] {result['text']}")
            os.unlink(tmp.name)

        return "\n".join(combined_text)

    def detect_language_only(self, audio_path: str) -> str:
        print("🔍  Running language detection probe ...")
        audio      = whisper.load_audio(audio_path)
        audio      = whisper.pad_or_trim(audio)
        mel        = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs   = self.model.detect_language(mel)
        detected   = max(probs, key=probs.get)
        confidence = probs[detected] * 100
        print(f"   Detected: {detected} ({confidence:.1f}% confidence)")
        if confidence < 60:
            print("   ⚠  Low confidence — consider specifying language manually.")
        return detected


class SeamlessTranscriber:
    def __init__(self, model_key: str = "seamless-v2-large"):
        import torch
        import transformers

        if model_key not in SEAMLESS_MODELS:
            raise ValueError(f"Unknown SeamlessM4T model '{model_key}'. Choose from: {list(SEAMLESS_MODELS.keys())}")

        version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if version < (4, 33):
            raise RuntimeError(
                f"transformers {transformers.__version__} is too old for SeamlessM4T.\n"
                f"Run: pip install --upgrade transformers==4.40.0\n"
                f"Then restart the server."
            )

        model_id       = SEAMLESS_MODELS[model_key]
        self.model_key = model_key
        self.model_id  = model_id
        is_v2          = "v2" in model_key  # v2-large uses SeamlessM4Tv2Model, v1-large uses SeamlessM4TModel

        print(f"⏳  Loading SeamlessM4T model: {model_id}")
        print(f"⚠   First run downloads ~{'10GB' if is_v2 else '5GB'} — this will take a while.")

        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id)

            if is_v2:
                from transformers import SeamlessM4Tv2Model
                self.model = SeamlessM4Tv2Model.from_pretrained(model_id)
            else:
                from transformers import SeamlessM4TModel
                self.model = SeamlessM4TModel.from_pretrained(model_id)
        except ImportError as e:
            raise RuntimeError(
                f"SeamlessM4T classes missing even on transformers {transformers.__version__}.\n"
                f"Run: pip install --upgrade transformers==4.40.0 sentencepiece\n"
                f"Then restart the server.\n"
                f"Original error: {e}"
            )

        self.model.eval()
        print(f"✅  SeamlessM4T loaded: {model_id}")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> dict:
        import torchaudio
        import torch

        tgt_lang = SEAMLESS_LANG_MAP.get(language or "auto", "eng")
        print(f"🌐  SeamlessM4T transcribing → target: {tgt_lang}")

        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform  = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio_inputs = self.processor(
            audios=waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )

        with torch.no_grad():
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=tgt_lang,
                generate_speech=False,
            )

        transcription = self.processor.decode(
            output_tokens[0].tolist(),
            skip_special_tokens=True,
        )

        print(f"✅  SeamlessM4T done.")

        return {
            "text":     transcription.strip(),
            "language": language or tgt_lang,
            "segments": [],
        }
