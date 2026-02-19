import os
import wave
import tempfile
import numpy as np

try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

SAMPLE_RATE = 16_000
CHANNELS    = 1


def record_from_mic(duration_seconds: int = 30) -> str:
    if not MIC_AVAILABLE:
        raise RuntimeError("sounddevice not installed. Run: pip install sounddevice")
    print(f"🎙  Recording for {duration_seconds} seconds...")
    audio_data = sd.rec(
        frames=int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    print("✅  Recording complete.")
    return _save_to_wav(audio_data)


def load_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        return _normalise_wav(file_path)
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub not installed. Run: pip install pydub")
    print(f"🔄  Converting {ext} → 16kHz mono WAV ...")
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    print(f"✅  Converted → {tmp.name}")
    return tmp.name


def _save_to_wav(audio_array: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_array.tobytes())
    return tmp.name


def _normalise_wav(path: str) -> str:
    with wave.open(path, "r") as wf:
        rate     = wf.getframerate()
        channels = wf.getnchannels()
    if rate == SAMPLE_RATE and channels == CHANNELS:
        return path
    if not PYDUB_AVAILABLE:
        print(f"⚠  Warning: audio is {rate}Hz/{channels}ch but pydub is missing.")
        return path
    print(f"🔄  Resampling WAV from {rate}Hz to {SAMPLE_RATE}Hz ...")
    audio = AudioSegment.from_wav(path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    return tmp.name
