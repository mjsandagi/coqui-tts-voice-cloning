import torch
from pathlib import Path
import torchaudio
import soundfile as sf
from datetime import datetime
import numpy as np
from TTS.api import TTS

# --- FIX FOR PYTORCH 2.6+ weights_only=True default ---
# Monkeypatch torch.load to use weights_only=False for TTS model loading
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# --- FIX FOR TORCHAUDIO 2.10+ trying to use torchcodec (needs FFmpeg on Windows) ---
# Bypass torchaudio entirely and use soundfile directly
def _patched_torchaudio_load(filepath, *args, **kwargs):
    audio, sr = sf.read(str(filepath), dtype='float32')
    # soundfile returns (samples, channels), convert to (channels, samples) tensor
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # mono -> (1, samples)
    else:
        audio = audio.T  # (samples, channels) -> (channels, samples)
    return torch.from_numpy(audio), sr
torchaudio.load = _patched_torchaudio_load

# --- CREATE TTS OBJECT ---
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")  # Use GPU

# --- SELECT WAVS FOR VOICE CLONING ---
wav_folder = Path("PATH/TO/YOUR/WAVS") # TODO: Update this to your folder with voice samples
selected_numbers = [1, 2, 3, 4, 5, 6]  # TODO: Adjust these after listening, and pick your best ones
speaker_wavs = [
    wav_folder / f"voice_file_{num}.wav"
    for num in selected_numbers
    if (wav_folder / f"sample_{num}.wav").exists() # TODO: Change the name if your filenames don't follow the format here
]
print(f"Using {len(speaker_wavs)} reference audio files")

# --- TEST TEXT ---
test_text = "This is a first test of my locally cloned voice. If you can hear this clearly, it worked."
stress_text = (
    "On a bright July morning, thirty-three quick brown foxes jumped over the lazy dog's dozen eggs, "
    "while £12.50 flew from the register, and I exclaimed, Wow! Can this really happen?"
)

# --- OUTPUT FOLDER ---
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

# --- GENERATE TTS FILES ---
speaker_wavs_str = [str(w) for w in speaker_wavs]

# Quality settings - experiment with these!
# Lower temperature = more stable/consistent but less expressive
# Higher temperature = more expressive but can get unstable
tts.tts_to_file(
    text=test_text,
    speaker_wav=speaker_wavs_str,
    language="en",
    file_path=output_folder / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.wav",
    split_sentences=True,  # Process sentence by sentence for better quality
)

# Uncomment to generate stress test
# tts.tts_to_file(
#     text=stress_text,
#     speaker_wav=speaker_wavs_str,
#     language="en",
#     file_path=output_folder / "stress_test.wav"
#)

# For even more control, use the model directly:
wav = tts.synthesizer.tts(
    text=test_text,
    speaker_wav=speaker_wavs_str,
    language="en",
    temperature=0.65,      # Default 0.75 - lower = more stable
    length_penalty=1.0,    # Controls speech pace
    repetition_penalty=2.0,  # Default 2.0 - prevents repetition
    top_k=50,              # Default 50
    top_p=0.85,            # Default 0.85 - lower = less random
)
tts.synthesizer.save_wav(wav, output_folder / f"output_custom_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.wav") 

print("TTS generation complete")