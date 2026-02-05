import torch
from pathlib import Path
import torchaudio
import soundfile as sf
from datetime import datetime
import numpy as np
from scipy import signal
from scipy.io import wavfile
from TTS.api import TTS

# --- FIX FOR PYTORCH 2.6+ weights_only=True default ---
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# --- FIX FOR TORCHAUDIO 2.10+ ---
def _patched_torchaudio_load(filepath, *args, **kwargs):
    audio, sr = sf.read(str(filepath), dtype='float32')
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    return torch.from_numpy(audio), sr
torchaudio.load = _patched_torchaudio_load


# --- POST-PROCESSING FUNCTION ---
def enhance_audio(audio_path, output_path):
    """Apply enhancement to make audio crisper"""
    sr, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float32) / 32768.0  # Normalise to [-1, 1]
    
    # 1. High-pass filter to remove muddiness (cut below 80Hz)
    b, a = signal.butter(4, 80 / (sr / 2), btype='high')
    audio = signal.filtfilt(b, a, audio)
    
    # 2. Gentle high-shelf boost for presence (boost above 3kHz)
    b, a = signal.butter(2, 3000 / (sr / 2), btype='high')
    high_freq = signal.filtfilt(b, a, audio)
    audio = audio + 0.3 * high_freq  # Add 30% of highs back
    
    # 3. Normalise to prevent clipping
    audio = audio / np.max(np.abs(audio)) * 0.95
    
    # Save
    audio = (audio * 32767).astype(np.int16)
    wavfile.write(output_path, sr, audio)
    print(f"Enhanced audio saved to {output_path}")

# --- CREATE TTS OBJECT ---
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")

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

# --- OUTPUT FOLDER ---
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

# --- GENERATE WITH QUALITY SETTINGS ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
raw_output = output_folder / f"raw_{timestamp}.wav"
enhanced_output = output_folder / f"enhanced_{timestamp}.wav"

# Generate with lower temperature for stability
tts.tts_to_file(
    text=test_text,
    speaker_wav=[str(w) for w in speaker_wavs],
    language="en",
    file_path=str(raw_output),
    split_sentences=True,
)

# Apply enhancement
enhance_audio(str(raw_output), str(enhanced_output))

print(f"Done! Compare:\n  Raw: {raw_output}\n  Enhanced: {enhanced_output}")