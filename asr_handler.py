"""
ASR Handler - Handles Whisper transcription
Based on reference pipeline implementation
"""
import whisper
import torch
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class ASRHandler:
    """Handles Automatic Speech Recognition using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper ASR model
        
        Args:
            model_size: Size of model ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper {self.model_size} model on {self.device}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("✓ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            self.model = None
    
    def transcribe_audio_file(self, audio_path: str, language: str = None) -> Tuple[str, str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code ('en', 'hi', None for auto-detect)
        
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return "", "error"
        
        try:
            logger.info(f"Transcribing: {audio_path}")
            result = self.model.transcribe(
                audio_path,
                language=language,
                verbose=False,
                fp16=False,  # Use FP32 for better accuracy
                temperature=0.0,  # Use deterministic decoding
                best_of=5  # Consider best of 5 candidates
            )
            
            text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            
            logger.info(f"Transcription complete: {text}")
            logger.info(f"Detected language: {detected_lang}")
            
            return text, detected_lang
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return "", "error"
    
    def transcribe_numpy_array(self, audio_data: np.ndarray, sr: int = 16000, 
                               language: str = None) -> Tuple[str, str]:
        """
        Transcribe audio from numpy array
        
        Args:
            audio_data: Audio data as numpy array
            sr: Sample rate
            language: Language code
        
        Returns:
            Tuple of (transcribed_text, detected_language)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return "", "error"
        
        try:
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            logger.info(f"Transcribing audio array (sr={sr})")
            result = self.model.transcribe(
                audio_data,
                language=language,
                verbose=False,
                fp16=False,  # Use FP32 for better accuracy
                temperature=0.0,  # Use deterministic decoding
                best_of=5  # Consider best of 5 candidates
            )
            
            text = result["text"].strip()
            detected_lang = result.get("language", "unknown")
            
            logger.info(f"Transcription complete: {text}")
            logger.info(f"Detected language: {detected_lang}")
            
            return text, detected_lang
        
        except Exception as e:
            logger.error(f"Error transcribing numpy array: {str(e)}")
            return "", "error"
