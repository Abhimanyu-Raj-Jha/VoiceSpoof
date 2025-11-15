"""
Configuration for VoiceSpoof V2
"""
import os
from pathlib import Path

# Project directories
PROJECT_DIR = Path(__file__).parent
TEMP_AUDIO_DIR = PROJECT_DIR / "temp_audio"
OUTPUT_AUDIO_DIR = PROJECT_DIR / "output_audio"

# Create directories if they don't exist
TEMP_AUDIO_DIR.mkdir(exist_ok=True)
OUTPUT_AUDIO_DIR.mkdir(exist_ok=True)

# Audio settings
SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 5
MAX_RECORDING_SECONDS = 60

# Model settings
ASR_MODEL_SIZE = "large"  # Options: tiny, base, small, medium, large | Using LARGE for superior accuracy
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Supported languages
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi"
}

# UI settings
STREAMLIT_THEME = "dark"
PAGE_TITLE = "🎙️ VoiceSpoof - ASR + TTS Pipeline"
PAGE_LAYOUT = "wide"

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
LOG_LEVEL = "INFO"
