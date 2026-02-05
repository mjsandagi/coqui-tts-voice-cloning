# Coqui TTS Voice Cloning

A practical implementation of voice cloning using Coqui Text-To-Speech (XTTS v2) with PyTorch and CUDA acceleration. This project demonstrates how to clone a speaker's voice from reference audio files and generate speech synthesis in that voice.

## Features

- **Voice Cloning**: Clone any voice using reference audio files (we recommend 3-6 samples of 5–15 seconds each)
- **GPU Acceleration**: CUDA-enabled inference for fast synthesis
- **Audio Enhancement**: Post-processing pipeline to improve output clarity
- **Robust Compatibility**: Fixes for PyTorch 2.6+ and torchaudio 2.10+ compatibility issues
- **Quality Control**: Adjustable temperature and inference parameters for fine-tuned output

## Requirements

- Python 3.10-3.11 (for compatibility with Coqui TTS and PyTorch)
- CUDA 13.0+ (for GPU acceleration)
- PyTorch 2.10.0
- Coqui TTS (from git repository)

Clone the repository, create a venv and install dependencies:

```bash
git clone https://github.com/mjsandagi/coqui-tts-voice-cloning.git
python -m venv venv # Ensure that you are using Python 3.10 - 3.11
source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Structure

```
.
├── tts.py              # Basic voice cloning synthesis
├── enhancement.py      # Voice cloning with post-processing audio enhancement
├── requirements.txt    # Python dependencies
└── outputs/            # Generated audio files
```

## Quick Start

### 1. Prepare Reference Audio

Place reference WAV files in a folder:

```
reference_audio/
├── sample_1.wav
├── sample_2.wav
└── sample_3.wav
```

### 2. Update Your Script

Edit `tts.py:34-39` or `enhancement.py:57-62` to point to your reference files:

```python
wav_folder = Path("reference_audio")
selected_wavs = ["sample_1.wav", "sample_2.wav", "sample_3.wav"]
speaker_wavs = [wav_folder / f for f in selected_wavs]
```

### 3. Run Synthesis

```python
python tts.py
# or for enhanced output:
python enhancement.py
```

Output files are saved to `outputs/` with timestamps.

## Usage

### Basic Synthesis

```python
from TTS.api import TTS
from pathlib import Path

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")

speaker_wavs = [str(Path("reference_audio/sample.wav"))]
tts.tts_to_file(
    text="Hello, this is a test.",
    speaker_wav=speaker_wavs,
    language="en",
    file_path="output.wav",
    split_sentences=True,
)
```

### Advanced Synthesis with Quality Parameters

```python
# For custom control, use the synthesizer directly:
wav = tts.synthesizer.tts(
    text="Your text here",
    speaker_wav=speaker_wavs,
    language="en",
    temperature=0.65,        # Lower = more stable/consistent (default 0.75)
    length_penalty=1.0,      # Controls speech pace
    repetition_penalty=2.0,  # Prevents word repetition (default 2.0)
    top_k=50,                # Default 50
    top_p=0.85,              # Lower = less randomness (default 0.85)
)
tts.synthesizer.save_wav(wav, "output_custom.wav")
```

### Audio Enhancement

The `enhancement.py` script applies post-processing:

1. **High-pass filter** (80 Hz): Removes low-frequency muddiness
2. **High-shelf boost** (3 kHz+): Adds presence and clarity
3. **Normalisation**: Prevents clipping and ensures consistent levels

```python
enhance_audio("raw_output.wav", "enhanced_output.wav")
```

## Technical Details

### Compatibility Fixes

This project includes fixes for common compatibility issues:

#### PyTorch 2.6+ `weights_only` Default

PyTorch 2.6 changed the default behaviour of `torch.load()` to use `weights_only=True`, which can cause issues with TTS model loading. The code monkeypatches `torch.load` to use `weights_only=False`:

```python
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
```

#### Torchaudio 2.10+ with FFmpeg Dependency

Newer versions of torchaudio attempt to use `torchcodec` for audio loading, which requires FFmpeg on Windows. This implementation replaces torchaudio's loader with `soundfile`:

```python
def _patched_torchaudio_load(filepath, *args, **kwargs):
    audio, sr = sf.read(str(filepath), dtype='float32')
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T
    return torch.from_numpy(audio), sr
torchaudio.load = _patched_torchaudio_load
```

## Quality Tuning

Voice cloning quality depends on several factors:

| Parameter            | Effect                       | Range   | Default |
| -------------------- | ---------------------------- | ------- | ------- |
| `temperature`        | Stability vs. expressiveness | 0.1–1.5 | 0.75    |
| `length_penalty`     | Speech pace                  | 0.5–2.0 | 1.0     |
| `repetition_penalty` | Prevents word repetition     | 1.0–5.0 | 2.0     |
| `top_p`              | Controls randomness          | 0.1–1.0 | 0.85    |

**Recommended settings:**

- **Stable synthesis**: `temperature=0.65`, `top_p=0.75`
- **Natural expressiveness**: `temperature=0.75`, `top_p=0.85`
- **Creative variation**: `temperature=1.0`, `top_p=0.9`

## Reference Audio Guidelines

For best results:

- **Duration**: 5–15 seconds per reference file
- **Quality**: Clear audio without background noise
- **Content**: Natural speech, conversational tone
- **Count**: 3–6 reference files recommended
- **Format**: WAV files at 22050 Hz or higher

Avoid:

- Heavily processed audio (heavy reverb, compression)
- Scripts or unnatural speech patterns
- Whispered or emotional extremes
- Background music or noise

## Troubleshooting

### CUDA Out of Memory

Reduce reference audio files or use CPU mode:

```python
tts.to("cpu")  # instead of tts.to("cuda")
```

### Model Download Fails

The first run downloads ~2 GB. Ensure sufficient free disk space and internet connection.

### Audio Quality Issues

- Experiment with `temperature` and `top_p` parameters
- Use higher-quality reference audio
- Increase the number of reference samples

## License

This project uses Coqui TTS, which is licensed under Mozilla Public Licence 2.0.

## Disclaimer

This implementation is for educational and research purposes. Ensure you have appropriate rights to any reference audio used for training, and respect intellectual property rights when generating speech synthesis.

## References

- [Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [XTTS v2 Model Paper](https://arxiv.org/abs/2305.15167)
- [PyTorch Documentation](https://pytorch.org/docs/)
