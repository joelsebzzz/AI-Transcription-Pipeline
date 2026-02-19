from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

MODEL_CONFIGS = {
    "t5-small": {
        "model":     "t5-small",
        "max_input": 400,
        "min_len":   30,
        "max_len":   120,
        "prefix":    "summarize: ",
    },
    "distilbart": {
        "model":     "sshleifer/distilbart-cnn-12-6",
        "max_input": 900,
        "min_len":   40,
        "max_len":   200,
        "prefix":    "",
    },
    "bart-large": {
        "model":     "facebook/bart-large-cnn",
        "max_input": 900,
        "min_len":   50,
        "max_len":   250,
        "prefix":    "",
    },
}


class Summarizer:
    def __init__(self, model_name: str = "t5-small"):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}")
        self.config    = MODEL_CONFIGS[model_name]
        model_id       = self.config["model"]
        print(f"⏳  Loading summarizer: {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self.model.eval()
        print(f"✅  Summarizer loaded.")

    def summarize_chunk(self, text: str) -> str:
        if not text.strip():
            return ""
        input_text = self.config["prefix"] + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config["max_input"],
            truncation=True,
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                min_length=self.config["min_len"],
                max_length=self.config["max_len"],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def summarize(self, chunks: list[str]) -> dict:
        start_time = time.time()
        if not chunks:
            return {"final_summary": "No content to summarize.", "chunk_summaries": []}

        print(f"\n📝  Summarizing {len(chunks)} chunk(s)...")
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"   Chunk {i+1}/{len(chunks)} ...", end=" ", flush=True)
            summary = self.summarize_chunk(chunk)
            chunk_summaries.append(summary)
            print("done")

        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0]
        else:
            combined = " ".join(chunk_summaries)
            print(f"   Combining {len(chunk_summaries)} chunk summaries...")
            final_summary = self.summarize_chunk(combined)

        elapsed = time.time() - start_time
        print(f"✅  Summarization complete in {elapsed:.1f}s")

        return {
            "final_summary":   final_summary,
            "chunk_summaries": chunk_summaries,
            "model_used":      self.config["model"],
            "time_seconds":    round(elapsed, 2),
        }


def compare_models(text: str, models: list[str] = None) -> dict:
    if models is None:
        models = ["t5-small", "distilbart"]
    results = {}
    for model_name in models:
        print(f"\n{'='*50}\n  Running comparison: {model_name}\n{'='*50}")
        summarizer = Summarizer(model_name)
        results[model_name] = summarizer.summarize([text])
        del summarizer.model, summarizer.tokenizer, summarizer
    return results
