# VoxScript — Multilingual Speech-to-Summary Pipeline

A full AI pipeline that records multilingual speech, transcribes it using OpenAI Whisper or Meta's SeamlessM4T, cleans the output, and generates a structured summary using transformer models. Comes with both a command-line interface and a browser-based web UI.

---

## Project Structure

```
multilingual_pipeline/
├── app.py                  Flask web server and API endpoints
├── main.py                 Command-line interface entry point
├── pipeline.py             Orchestrator connecting all modules
├── audio_capture.py        Microphone recording and audio file loading
├── transcriber.py          Whisper + SeamlessM4T speech-to-text wrappers
├── cleaner.py              Transcript noise removal
├── chunker.py              Splits long text for summarizer context limits
├── summarizer.py           T5 / DistilBART / BART summarization
├── requirements.txt        All Python dependencies
└── templates/
    └── index.html          Full web UI — HTML, CSS, JavaScript
```

---

## Setup

```bash
pip install -r requirements.txt
```

Install ffmpeg separately for MP3/M4A file support:
- **Windows** — https://ffmpeg.org/download.html (add to PATH)
- **macOS** — `brew install ffmpeg`
- **Linux** — `sudo apt install ffmpeg`

---

## Important: NumPy Version

PyTorch, Whisper, and Transformers were all compiled against NumPy 1.x. If you have NumPy 2.x installed, the server will crash on load. Fix it with:

```bash
pip install "numpy<2"
```

Always use `python -m pip install` rather than just `pip install` to ensure packages go into the correct Python environment.

---

## How to Run

**Web UI:**
```bash
python app.py
# Open http://localhost:5000 in Google Chrome
# Must be localhost — not an IP address — for Web Speech API to work
```

**Command line:**
```bash
python main.py                              # interactive menu
python main.py --mic --duration 30         # record from mic
python main.py --file audio.wav --lang ml  # transcribe a file
python main.py --text "your text here"     # summarize text only
python main.py --compare --file audio.wav  # compare T5 vs DistilBART
```

---

## Web UI — Three ASR Modes

The UI has three tabs at the top. Each tab has its own recording/upload section. The transcript box and summary section below are shared — they stay visible regardless of which tab is selected.

### 🌐 Web Speech API Tab
Browser-native speech recognition powered by Google's servers. Requires Chrome and internet. Best choice for Malayalam because Google's backend models are significantly larger than Whisper-small.

How it works: Chrome sends your microphone audio to Google's servers in real-time and streams back the transcript as you speak. Interim results appear as grey text; finalized words are locked in permanently.

**Continuous recording fix:** Chrome's Web Speech API internally stops after every few seconds even with `continuous = true`. The UI handles this by creating a fresh `SpeechRecognition` object and restarting it automatically via the `onend` event, with a 150ms delay to prevent restart loops. A `wsRestarting` flag prevents multiple simultaneous restarts. `no-speech` and `aborted` errors are silently ignored — the restart loop handles recovery automatically.

**Known limitation:** Requires internet. If Google's servers are blocked on your network, switch to the Whisper tab.

### 🤖 Whisper Tab
OpenAI's Whisper model running locally on your machine. No internet needed after the first download. You can either drop an audio file or record directly in the browser. The browser records as a `.webm` blob using `MediaRecorder`, which is sent to the Flask server where ffmpeg converts it to 16kHz mono WAV before Whisper processes it.

### 🔬 SeamlessM4T Tab
Meta's SeamlessM4T model — see full details in the SeamlessM4T section below.

---

## Settings

**Language** — Choose from auto-detect, Malayalam, English, Hindi, Tamil, Telugu, French, German, Spanish, Chinese. For Malayalam with Whisper, always select Malayalam explicitly — auto-detect fails roughly 40% of the time.

**Summarizer Model** — Selects which transformer model generates the summary. See the Summarizer section for details.

**Whisper Model** — Only visible on the Whisper tab. Selects which Whisper size to use. The selection is sent to the server with each transcription request.

**Seamless Model** — Only visible on the SeamlessM4T tab.

---

## File-by-File Reference

---

### `audio_capture.py`

Handles all audio input for the command-line pipeline.

**Constants:**
- `SAMPLE_RATE = 16000` — Whisper was trained on 16kHz audio. All input is normalised to this rate.
- `CHANNELS = 1` — Mono only. Stereo adds no benefit for speech recognition.

**`record_from_mic(duration_seconds)`**
Records from the system microphone using `sounddevice`. Blocks the thread during recording. Returns a path to a temporary `.wav` file.

**`load_from_file(file_path)`**
Loads an audio file from disk. If already a 16kHz mono WAV, returns it unchanged. For all other formats (MP3, M4A, OGG, FLAC), uses `pydub` and `ffmpeg` to convert. Phone recordings in M4A format particularly need this step.

**`_save_to_wav(audio_array)`**
Writes a NumPy int16 array from `sounddevice` to a temporary WAV file.

**`_normalise_wav(path)`**
Checks a WAV file's sample rate and channel count. Resamples using pydub if not 16kHz mono.

---

### `transcriber.py`

Contains two completely separate transcription classes — one for Whisper, one for SeamlessM4T.

**Constants:**

`SUPPORTED_LANGUAGES` — Maps 2-letter ISO 639-1 codes to language names. Used by Whisper.

`SEAMLESS_LANG_MAP` — Maps 2-letter codes to 3-letter ISO 639-3 codes. SeamlessM4T requires 3-letter codes (`eng`, `mal`, `hin`) while Whisper uses 2-letter codes (`en`, `ml`, `hi`).

`SEAMLESS_MODELS` — Maps friendly key names to the correct HuggingFace repo IDs:
- `seamless-v2-large` → `facebook/seamless-m4t-v2-large`
- `seamless-v1-large` → `facebook/hf-seamless-m4t-large`
- `seamless-v1-medium` → `facebook/hf-seamless-m4t-medium`

Important: the repos without `hf-` prefix (`facebook/seamless-m4t-large`, `facebook/seamless-m4t-medium`) are Meta's original PyTorch checkpoints and are missing `preprocessor_config.json` — they cannot be loaded with the transformers library and will throw an error. Only the `hf-` prefixed repos and `seamless-m4t-v2-large` work correctly.

---

**Class `Transcriber` (Whisper)**

`__init__(model_size)`
Loads the Whisper model. Downloads on first run and caches locally. `fp16=False` is hardcoded because CPU inference does not support half-precision floats.

Whisper model sizes and their practical use on CPU:

| Size | Download | RAM | Speed (1 min audio) | Malayalam |
|---|---|---|---|---|
| tiny | 75MB | ~1GB | ~20s | Poor |
| base | 145MB | ~1GB | ~40s | Unstable |
| small | 460MB | ~2GB | ~90s | Stable ✅ |
| medium | 1.5GB | ~5GB | ~8min | Good |
| large-v3 | 3GB | ~10GB | ~20min | Best |

`transcribe(audio_path, language, task)`
Runs Whisper on the audio file. Pass `language="ml"` explicitly for Malayalam. Use `task="translate"` to convert any language to English text instead of transcribing in the original language. Returns a dict with `text`, `language`, and `segments`.

`transcribe_with_language_segments(audio_path, segments, languages)`
For code-switching — slices audio by time range and transcribes each slice with a forced language. Returns a combined transcript with language labels.

`detect_language_only(audio_path)`
Runs Whisper's language detection on the first 30 seconds only. Faster than full transcription. Warns if confidence is below 60%.

---

**Class `SeamlessTranscriber`**

Meta's SeamlessM4T is a unified model that handles speech-to-text, text-to-text, speech-to-speech, and text-to-speech all in one architecture. For this project only the speech-to-text capability is used.

`__init__(model_key)`
Checks transformers version first — requires 4.33 or newer. Checks available RAM before attempting to load. Raises a `MemoryError` with a clear message if the system doesn't have enough free RAM rather than crashing the process.

Loading uses `AutoProcessor` for v1 models and the dedicated `SeamlessM4Tv2Model` class for v2. The v2 model has a different internal architecture and requires its own class.

`transcribe(audio_path, language)`
Loads audio with `torchaudio`, resamples to 16kHz if needed, converts stereo to mono by averaging channels, processes through the processor, generates tokens with `generate_speech=False` (text output only), and decodes back to a string. Uses `torch.no_grad()` for inference to save memory.

**RAM requirements:**
- `seamless-v1-medium` — ~3GB download, needs ~4GB free RAM
- `seamless-v1-large` — ~5GB download, needs ~6GB free RAM
- `seamless-v2-large` — ~10GB download, needs ~12GB free RAM

On low-end laptops (4-8GB total RAM), SeamlessM4T will not run. The server checks available RAM before attempting to load and returns a clear error message instead of crashing.

---

### `cleaner.py`

Removes noise from raw Whisper output before summarization.

**`clean_transcript(text, ...)`**
Runs all cleaning steps in sequence: Unicode normalisation, hallucination pattern removal, filler word removal, repeated phrase collapse, whitespace collapse.

**`clean_segments(segments, **kwargs)`**
Applies cleaning to each Whisper segment individually, preserving timestamps. Drops empty segments.

**`_fix_unicode(text)`**
Normalises to NFC form. Malayalam uses composed Unicode characters — NFC ensures they are stored as single codepoints which models handle more reliably. Removes the Unicode replacement character (U+FFFD) from encoding failures.

**`_remove_hallucinations(text)`**
Removes patterns like `[Music]`, `[Applause]`, "thank you for watching" — text Whisper generates during silence.

**`_remove_fillers(text, language)`**
Removes spoken filler words (um, uh, like, you know). Skips non-English text to avoid corrupting it.

**`_remove_repeated_phrases(text)`**
Collapses consecutive repeated words and 2-word phrases using regex backreferences. Handles Whisper stuttering like "thank you thank you thank you".

---

### `chunker.py`

Splits long transcripts into smaller pieces that fit the summariser's context window.

**Why chunking is needed:**
T5-small accepts at most 512 tokens (~380 words). A 10-minute conversation produces ~1500 words. Without chunking, the model silently truncates and only summarises the first portion.

**Strategy — map-reduce:** Summarise each chunk independently, concatenate the chunk summaries, then summarise the summaries for a final output.

**`chunk_text(text, max_tokens, overlap_sentences)`**
Splits at sentence boundaries. The `overlap_sentences` parameter includes the last N sentences of the previous chunk at the start of the next, preventing context loss at boundaries. Falls back to word-count splitting when punctuation is sparse — common with Malayalam Whisper output.

**`chunk_segments(segments, max_tokens)`**
Chunks using Whisper's timed segments instead of punctuation. More reliable for Malayalam.

---

### `summarizer.py`

Loads and runs sequence-to-sequence transformer models.

**Why models load directly instead of `pipeline("summarization")`**
Newer versions of the transformers library removed `"summarization"` as a valid pipeline task string. Direct loading with `AutoTokenizer` and `AutoModelForSeq2SeqLM` works across all versions and gives control over generation parameters.

**Model configurations:**

| Key | Model | RAM | Speed | Notes |
|---|---|---|---|---|
| `t5-small` | google-t5/t5-small | 1.5GB | Fast | Default. Requires `"summarize: "` prefix |
| `distilbart` | sshleifer/distilbart-cnn-12-6 | 2GB | Medium | Good balance |
| `bart-large` | facebook/bart-large-cnn | 4GB | Slow | Best quality |

**`summarize_chunk(text)`**
Tokenises input with truncation, then runs `model.generate()` under `torch.no_grad()`. Uses beam search (`num_beams=4`) for quality. `no_repeat_ngram_size=3` prevents repeated phrases in output.

**`summarize(chunks)`**
Map-reduce: summarises each chunk, then summarises the combined chunk summaries. Returns `final_summary`, `chunk_summaries`, `model_used`, and `time_seconds`.

**`compare_models(text, models)`**
Loads each model in sequence, runs summarisation, deletes the model before loading the next to stay within RAM. For comparison/presentation use only — slow.

**Important limitation:** All three models are trained on English data. For Malayalam input, they produce an English summary. Use Whisper's `task="translate"` option first if you need to summarise Malayalam speech with English output.

---

### `pipeline.py`

Orchestrates all modules in order. The only file needed if using the system programmatically.

`transcriber` and `summarizer` are lazy-loaded properties — models load only on first use and are reused for all subsequent calls, avoiding the 10-20 second reload on every request.

`run_from_mic(duration, language)` → records then runs full pipeline.

`run_from_file(file_path, language)` → loads file then runs full pipeline.

`run_from_text(text)` → skips audio and transcription, runs cleaning/chunking/summarisation on provided text.

`run_comparison(audio_path, language)` → transcribes once, then runs `compare_models()` for side-by-side output.

`_run_summarization(raw_text, language, segments)` → internal step 2-4. Uses segment-based chunking when Whisper segments are available (better for low-punctuation languages), otherwise falls back to sentence-based chunking.

`_save_result(result)` → saves full result dict as timestamped JSON in `outputs/`.

---

### `app.py`

Flask web server. All models are lazy-loaded singletons — loaded once on first request and reused. Flask runs single-threaded in development mode so no locking is needed.

**Global state:**
- `_whisper_transcriber` + `_loaded_whisper_size` — reloads only when model size changes
- `_seamless_transcriber` + `_loaded_seamless_key` — reloads only when model key changes
- `_summarizer` — reloads only when model name changes

**Routes:**

`GET /` — serves `templates/index.html`

`POST /api/transcribe/whisper` — accepts multipart form with `audio` (file), `language` (code or `"auto"`), and `whisper_model` (size key). Saves to temp file, converts via ffmpeg, transcribes, cleans, returns JSON with `transcript`, `raw_transcript`, `language`, `segments`, `model_used`.

`POST /api/transcribe/seamless` — accepts multipart form with `audio`, `language`, and `seamless_model`. Checks available RAM before loading. Same return format as Whisper endpoint.

`POST /api/summarize` — accepts JSON with `text`, `model`, `language`, `segments`. Chunks then summarises. Returns `summary`, `chunk_summaries`, `num_chunks`, `model_used`, `time_seconds`.

`POST /api/detect-language` — fast language probe without full transcription. Returns detected language code.

`_ensure_wav(path)` — calls ffmpeg to convert browser `.webm` recordings to 16kHz mono WAV. Falls back gracefully if ffmpeg is not installed.

---

### `templates/index.html`

Single-file web UI. All CSS and JavaScript are inline — no build step, no external framework.

**Layout — 5 numbered steps:**
1. Choose recognition method (3 tabs: Web Speech API / Whisper / SeamlessM4T)
2. Settings (language, summariser model, ASR model — ASR model dropdown changes based on active tab)
3. Transcript textarea — shared across all tabs, always visible, editable, shows word count
4. Summarize button
5. Summary output with metadata and per-chunk detail

**CSS design system:** Dark theme (`#0a0c0f` background) with subtle 40px grid texture. Accent colour `#00d4aa` (teal). SeamlessM4T accent `#a78bfa` (purple). `Space Mono` for labels/metadata, `DM Sans` for body. All values as CSS custom properties.

**Panel visibility:** All three panels (`#webspeech-panel`, `#whisper-panel`, `#seamless-panel`) are hidden by default with `display: none`. The active class sets `display: block`. `setMode()` toggles the active class and also shows/hides the correct settings dropdown.

**Web Speech API — continuous recording implementation:**

The core challenge: Chrome stops the recogniser after every few seconds even with `continuous = true`. The solution:

- `buildRecognition()` creates a fresh `SpeechRecognition` object every time — reusing after `.stop()` is unreliable
- `attachHandlers(recognition)` binds all events to the new object
- `onend` waits 150ms then calls `buildRecognition()` and starts again
- `wsRestarting` flag prevents multiple simultaneous restarts racing each other
- `no-speech` and `aborted` errors are silently ignored — the restart loop handles recovery
- `wsFullTranscript` accumulates only finalised results; interim results display but don't save
- `stopWebSpeech()` sets `wsRecording = false` and nulls `onend` before calling `.stop()` to prevent the restart loop triggering after an intentional stop

**SeamlessM4T fetch timeout:**
Normal browser fetch times out after ~2 minutes. SeamlessM4T can take longer to load on first run. The Seamless fetch uses `AbortController` with a 600-second (10-minute) timeout. If it times out, the error message tells the user to check the terminal for the `✅ SeamlessM4T loaded` confirmation.

---

## Language Support

| Language | Code | Web Speech API | Whisper-small | SeamlessM4T |
|---|---|---|---|---|
| English | en | Excellent | Excellent | Excellent |
| Malayalam | ml | Good (Chrome only) | Fair — always specify `--lang ml` | Very Good |
| Hindi | hi | Good | Good | Very Good |
| Tamil | ta | Good | Fair | Very Good |
| Telugu | te | Good | Fair | Very Good |

For Malayalam, Web Speech API and SeamlessM4T both outperform Whisper-small. The difference is because Google's Web Speech backend and Meta's SeamlessM4T both use significantly larger models with more Indic language training data.

---

## ASR Model Comparison

| Model | Type | Size | RAM | Malayalam (offline) | Speed on CPU |
|---|---|---|---|---|---|
| Whisper-tiny | Local | 75MB | 1GB | Poor | ~20s/min |
| Whisper-base | Local | 145MB | 1GB | Unstable | ~40s/min |
| Whisper-small | Local | 460MB | 2GB | ~65% ✅ default | ~90s/min |
| Whisper-medium | Local | 1.5GB | 5GB | ~80% | ~8min/min |
| Whisper-large-v3 | Local | 3GB | 10GB | ~88% | ~20min/min |
| SeamlessM4T v1-medium | Local | 2.5GB | 4GB | ~85% | ~15min/min |
| SeamlessM4T v1-large | Local | 5GB | 6GB | ~88% | ~20min/min |
| SeamlessM4T v2-large | Local | 10GB | 12GB | ~92% | ~25min/min |
| Web Speech API | Google servers | 0MB | 0GB | ~88% (online) | Real-time |

---

## Output Format

Every run saves a JSON file to `outputs/summary_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2024-01-15T14:23:10",
  "language": "ml",
  "raw_transcript": "raw whisper output...",
  "clean_transcript": "cleaned version...",
  "num_chunks": 2,
  "chunk_summaries": ["summary of chunk 1", "summary of chunk 2"],
  "final_summary": "Combined final summary.",
  "model_used": "t5-small",
  "total_time_sec": 94.7
}
```

---

## Common Errors and Fixes

**`Unknown task summarization`**
Newer transformers versions removed this pipeline task name. Fixed in `summarizer.py` by loading models directly with `AutoTokenizer` and `AutoModelForSeq2SeqLM`.

**`Could not import module 'AutoProcessor'`**
transformers version is older than 4.33. Run `pip install --upgrade transformers==4.40.0` then restart.

**`Could not import module 'SeamlessM4TProcessor'`**
Same cause — old transformers. Same fix.

**`does not appear to have a file named preprocessor_config.json`**
Wrong HuggingFace repo. The repos `facebook/seamless-m4t-large` and `facebook/seamless-m4t-medium` are raw PyTorch checkpoints that cannot be loaded with transformers. The correct repos are `facebook/hf-seamless-m4t-large` and `facebook/hf-seamless-m4t-medium`.

**`A module compiled with NumPy 1.x cannot run in NumPy 2.x`**
Run `pip install "numpy<2"` then restart.

**`TemplateNotFound: index.html`**
Run `python app.py` from inside the `multilingual_pipeline/` folder.

**Web Speech API `network` error**
Open the app at exactly `http://localhost:5000`. Chrome blocks Web Speech on plain IP addresses. Also requires internet to reach Google's servers.

**Web Speech stops after a few words**
Fixed by the `onend` restart loop. If still happening, check that you are using Chrome, not Firefox or Edge.

**SeamlessM4T crashes the terminal (out of memory)**
The model exceeds your available RAM. The server now checks RAM before loading and returns an error message instead of crashing. For a presentation on a low-end laptop, use the SeamlessM4T tab to explain the model's architecture and advantages, and show a comparison table rather than running it live.

**`Not enough RAM` error from server**
Expected on laptops with less than 6GB free RAM. Either use Whisper instead, or run SeamlessM4T on Google Colab.

---

## Hardware Recommendations

| Task | Minimum RAM | Recommended |
|---|---|---|
| Whisper-small + T5-small | 4GB | 8GB |
| Whisper-medium + DistilBART | 8GB | 16GB |
| SeamlessM4T v1-medium | 6GB free | 12GB total |
| SeamlessM4T v2-large | 12GB free | 24GB total |

For SeamlessM4T on constrained hardware, Google Colab (free tier) provides 12GB RAM and is sufficient for v1-large.
