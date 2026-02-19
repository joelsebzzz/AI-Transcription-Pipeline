import os
import json
import time
from datetime import datetime
from typing import Optional

from audio_capture import record_from_mic, load_from_file
from transcriber  import Transcriber, SUPPORTED_LANGUAGES
from cleaner      import clean_transcript, clean_segments
from chunker      import chunk_text, chunk_segments
from summarizer   import Summarizer, compare_models

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Pipeline:
    def __init__(self, whisper_model: str = "small", summarizer_model: str = "t5-small"):
        self.whisper_model    = whisper_model
        self.summarizer_model = summarizer_model
        self._transcriber     = None
        self._summarizer      = None

    @property
    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            self._transcriber = Transcriber(self.whisper_model)
        return self._transcriber

    @property
    def summarizer(self) -> Summarizer:
        if self._summarizer is None:
            self._summarizer = Summarizer(self.summarizer_model)
        return self._summarizer

    def run_from_mic(self, duration: int = 30, language: Optional[str] = None) -> dict:
        print("\n" + "="*60)
        print("  PIPELINE — Live Microphone")
        print("="*60)
        audio_path = record_from_mic(duration_seconds=duration)
        return self._run_pipeline(audio_path, language=language)

    def run_from_file(self, file_path: str, language: Optional[str] = None) -> dict:
        print("\n" + "="*60)
        print(f"  PIPELINE — File: {os.path.basename(file_path)}")
        print("="*60)
        audio_path = load_from_file(file_path)
        return self._run_pipeline(audio_path, language=language)

    def run_from_text(self, text: str) -> dict:
        print("\n" + "="*60)
        print("  PIPELINE — Text-Only Mode")
        print("="*60)
        return self._run_summarization(raw_text=text, language="en", segments=[])

    def run_comparison(self, audio_path: str, language: Optional[str] = None) -> dict:
        print("\n" + "="*60)
        print("  MODEL COMPARISON MODE")
        print("="*60)
        audio_path = load_from_file(audio_path)
        result     = self.transcriber.transcribe(audio_path, language=language)
        cleaned    = clean_transcript(result["text"], language=result["language"])
        print(f"\n📋  Transcript ({len(cleaned.split())} words):\n{cleaned[:500]}...")
        comparison = compare_models(cleaned, models=["t5-small", "distilbart"])
        self._print_comparison(comparison)
        return comparison

    def _run_pipeline(self, audio_path: str, language: Optional[str]) -> dict:
        print("\n[1/4] 🎤  Transcription")
        result    = self.transcriber.transcribe(audio_path, language=language)
        raw_text  = result["text"]
        print(f"   Raw ({len(raw_text.split())} words): {raw_text[:300]}{'...' if len(raw_text) > 300 else ''}")
        return self._run_summarization(raw_text, result["language"], result["segments"])

    def _run_summarization(self, raw_text: str, language: str, segments: list) -> dict:
        pipeline_start = time.time()

        print("\n[2/4] 🧹  Cleaning")
        cleaned_text = clean_transcript(raw_text, language=language)
        print(f"   Cleaned ({len(cleaned_text.split())} words)")

        print("\n[3/4] ✂️   Chunking")
        if segments:
            cleaned_segs = clean_segments(segments, language=language)
            chunks = chunk_segments(cleaned_segs, max_tokens=400)
        else:
            chunks = chunk_text(cleaned_text, max_tokens=400)

        print("\n[4/4] 🧠  Summarization")
        summary_result = self.summarizer.summarize(chunks)

        total_time = time.time() - pipeline_start

        final = {
            "timestamp":        datetime.now().isoformat(),
            "language":         language,
            "raw_transcript":   raw_text,
            "clean_transcript": cleaned_text,
            "num_chunks":       len(chunks),
            "chunk_summaries":  summary_result["chunk_summaries"],
            "final_summary":    summary_result["final_summary"],
            "model_used":       summary_result["model_used"],
            "total_time_sec":   round(total_time, 2),
        }

        self._print_result(final)
        self._save_result(final)
        return final

    def _print_result(self, result: dict):
        lang_name = SUPPORTED_LANGUAGES.get(result["language"], result["language"])
        print("\n" + "="*60)
        print("  RESULT")
        print("="*60)
        print(f"  Language  : {lang_name} ({result['language']})")
        print(f"  Words     : {len(result['clean_transcript'].split())}")
        print(f"  Chunks    : {result['num_chunks']}")
        print(f"  Model     : {result['model_used']}")
        print(f"  Time      : {result['total_time_sec']}s")
        print(f"\n  TRANSCRIPT:\n  {result['clean_transcript']}")
        print(f"\n  SUMMARY:\n  {result['final_summary']}")
        print("="*60)

    def _print_comparison(self, comparison: dict):
        print("\n" + "="*60 + "\n  MODEL COMPARISON\n" + "="*60)
        for model_name, result in comparison.items():
            print(f"\n  [{model_name}]  ({result['time_seconds']}s)\n  {result['final_summary']}")
        print("="*60)

    def _save_result(self, result: dict):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"summary_{ts}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n💾  Saved → {filename}")
