"""
Audio Handler - Handles audio processing, recording, and file operations
Based on reference pipeline implementation
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioHandler:
    """Handles audio processing, recording, and file operations"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio handler
        
        Args:
            sample_rate: Default sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    @staticmethod
    def load_audio_file(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate to resample to
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            logger.info(f"Loading audio from: {audio_path}")
            audio, sample_rate = librosa.load(audio_path, sr=sr, mono=True)
            logger.info(f"Audio loaded: shape={audio.shape}, sr={sample_rate}")
            return audio, sample_rate
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            return np.array([]), 16000
    
    @staticmethod
    def save_audio_file(audio_data: np.ndarray, output_path: str, sr: int = 16000) -> bool:
        """
        Save audio to file
        
        Args:
            audio_data: Audio data as numpy array
            output_path: Path to save audio file
            sr: Sample rate
        
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio_data, sr)
            logger.info(f"Audio saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            return False
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio: Input audio data
        
        Returns:
            Normalized audio
        """
        try:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            logger.info(f"Audio normalized: peak={np.max(np.abs(audio)):.4f}")
            return audio
        except Exception as e:
            logger.warning(f"Normalization error: {str(e)}")
            return audio
    
    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Input audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
        
        Returns:
            Resampled audio
        """
        try:
            if orig_sr == target_sr:
                return audio
            
            logger.info(f"Resampling audio: {orig_sr} Hz -> {target_sr} Hz")
            resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            logger.info(f"Resampling complete: shape {audio.shape} -> {resampled.shape}")
            return resampled
        except Exception as e:
            logger.error(f"Error resampling audio: {str(e)}")
            return audio
    
    @staticmethod
    def apply_gain(audio: np.ndarray, db: float = 0.0) -> np.ndarray:
        """
        Apply gain adjustment to audio
        
        Args:
            audio: Input audio data
            db: Gain in decibels (positive to increase, negative to decrease)
        
        Returns:
            Audio with gain applied
        """
        try:
            if db == 0:
                return audio
            
            gain_linear = 10 ** (db / 20.0)
            adjusted = audio * gain_linear
            
            # Clip to prevent clipping
            adjusted = np.clip(adjusted, -1.0, 1.0)
            logger.info(f"Gain applied: {db:+.1f} dB")
            return adjusted
        except Exception as e:
            logger.warning(f"Gain adjustment error: {str(e)}")
            return audio
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.02, sr: int = 16000) -> np.ndarray:
        """
        Trim silence from beginning and end of audio
        
        Args:
            audio: Input audio data
            threshold: Amplitude threshold for silence detection
            sr: Sample rate
        
        Returns:
            Audio with silence trimmed
        """
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=20, ref=threshold)
            logger.info(f"Silence trimmed: {len(audio)} -> {len(trimmed)} samples")
            return trimmed
        except Exception as e:
            logger.warning(f"Trim silence error: {str(e)}")
            return audio
    
    @staticmethod
    def concatenate_audio(audio_list: list, sr: int = 16000) -> np.ndarray:
        """
        Concatenate multiple audio arrays
        
        Args:
            audio_list: List of audio arrays
            sr: Sample rate
        
        Returns:
            Concatenated audio
        """
        try:
            if not audio_list:
                return np.array([])
            
            concatenated = np.concatenate(audio_list)
            logger.info(f"Audio concatenated: {len(audio_list)} pieces -> {len(concatenated)} samples")
            return concatenated
        except Exception as e:
            logger.error(f"Error concatenating audio: {str(e)}")
            return np.array([])
    
    @staticmethod
    def get_audio_duration(audio: np.ndarray, sr: int = 16000) -> float:
        """
        Get duration of audio in seconds
        
        Args:
            audio: Audio data
            sr: Sample rate
        
        Returns:
            Duration in seconds
        """
        try:
            duration = len(audio) / sr
            return duration
        except Exception as e:
            logger.warning(f"Error calculating duration: {str(e)}")
            return 0.0
    
    @staticmethod
    def convert_to_float32(audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to float32
        
        Args:
            audio: Input audio data
        
        Returns:
            Audio as float32
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio
