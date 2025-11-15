# VoiceSpoof - Voice Spoofing System

A production-grade voice spoofing system using **Whisper Large** for ASR (Automatic Speech Recognition) and **XTTSV2** for TTS (Text-to-Speech) with speaker cloning.

## Features

- 🎤 **High-Quality ASR**: Whisper Large model (1.5B parameters) for perfect transcription
- 🎵 **Advanced TTS**: XTTSV2 for natural voice synthesis with speaker cloning
- 🌐 **Multi-Language**: Supports English and Hindi
- 🎙️ **Speaker Cloning**: Uses reference audio to match voice characteristics
- 🚀 **CPU Compatible**: Runs on CPU (no GPU required)
- 💾 **Easy Download**: Export generated spoof audio as WAV

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- Whisper
- TTS (Coqui)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Workflow

1. **Record**: Record audio in English or Hindi
2. **Transcribe**: AI transcribes your audio perfectly
3. **Edit**: Review or edit the transcribed text
4. **Generate**: Create spoofed speech using your voice as reference
5. **Download**: Save the generated spoof audio

## Models

- **ASR**: OpenAI Whisper Large (1.5B parameters)
- **TTS**: Coqui XTTSV2 (500M parameters)
- **Device**: CPU (CUDA supported)

## Architecture

```
Audio Input
    ↓
Whisper ASR (Large)
    ↓
Text Processing
    ↓
XTTSV2 TTS (with Speaker Cloning)
    ↓
Audio Output
```

## Project Structure

```
VoiceSpoof_V2/
├── app.py                 # Main Streamlit app
├── asr_handler.py        # Whisper ASR wrapper
├── text_handler.py       # Text processing
├── tts_handler.py        # XTTSV2 wrapper
├── audio_handler.py      # Audio utilities
├── config.py             # Configuration
├── requirements.txt      # Dependencies
└── README.md
```

## Notes

- First run will download ~2GB for Whisper Large and TTS models
- CPU synthesis takes 5-15 seconds per sentence
- Best quality with clear reference audio
- Works best with 16kHz audio input

## License

MIT

## Author

Abhimanyu Raj Jha
