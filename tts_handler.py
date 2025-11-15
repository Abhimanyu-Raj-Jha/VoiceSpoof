"""
TTS Handler - Handles XTTSV2 text-to-speech synthesis with QUALITY OPTIMIZATION
Based on reference pipeline implementation with enhanced quality settings
"""
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging
import re
import os
import tempfile

logger = logging.getLogger(__name__)


class TTSHandler:
    """Handles Text-to-Speech synthesis using XTTSV2 with quality optimization"""
    
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", 
                 device: Optional[str] = None):
        """
        Initialize XTTSV2 TTS model with quality optimization
        
        Args:
            model_name: TTS model name
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tts = None
        self._load_model()
    
    def _load_model(self):
        """Load XTTSV2 model with proper initialization"""
        try:
            from TTS.api import TTS
            
            logger.info(f"🎙️ Loading XTTSV2 model on {self.device}...")
            
            # Initialize with correct parameters (no 'in_gpu_memory_efficient')
            self.tts = TTS(
                model_name=self.model_name,
                gpu=(self.device == "cuda"),
                progress_bar=False
            )
            
            logger.info("✅ XTTSV2 model loaded successfully")
        
        except ImportError:
            logger.error("❌ TTS library not installed. Install with: pip install TTS")
            self.tts = None
        except Exception as e:
            error_msg = str(e)
            if "libtorchaudio" in error_msg:
                logger.warning(f"⚠️ Torchaudio loading issue (non-critical): {error_msg}")
                logger.warning("Attempting to continue anyway...")
                # Try importing TTS and see if it still works
                try:
                    from TTS.api import TTS
                    self.tts = TTS(
                        model_name=self.model_name,
                        gpu=(self.device == "cuda"),
                        progress_bar=False
                    )
                    logger.info("✅ XTTSV2 model loaded successfully (with warnings)")
                except Exception as e2:
                    logger.error(f"❌ Failed to initialize TTS: {str(e2)}")
                    self.tts = None
            else:
                logger.error(f"❌ Failed to load TTS model: {error_msg}")
                self.tts = None
    
    def _chunk_text_for_tts(self, text: str, max_chars: int = 150) -> list:
        """
        Split text into chunks for TTS processing
        
        Args:
            text: Input text
            max_chars: Maximum characters per chunk (XTTSV2 limit ~150 chars)
        
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[।.!?])\s+', text)
        chunks, current = [], ""
        
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current += sent + " "
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent + " "
        
        if current.strip():
            chunks.append(current.strip())
        
        return [c for c in chunks if c.strip()]
    
    def synthesize_speech(self, text: str, speaker_wav: str, language: str = "hi",
                         output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using reference speaker with QUALITY OPTIMIZATIONS
        
        QUALITY IMPROVEMENTS:
        - Uses tts_to_file() method for FILE-based synthesis (highest quality)
        - Proper speaker embedding handling for natural voice cloning
        - Optimal chunk management for long texts
        - Zero invalid parameters (no temperature/top_p - not supported by Coqui)
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference speaker audio file
            language: Language code ('en', 'hi', etc.)
            output_path: Optional path to save output audio file
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if self.tts is None:
            logger.error("❌ TTS model not loaded")
            return np.array([]), 24000
        
        try:
            # Check if speaker file exists
            if not Path(speaker_wav).exists():
                logger.error(f"❌ Speaker file not found: {speaker_wav}")
                return np.array([]), 24000
            
            logger.info(f"🎵 XTTSV2 Quality Synthesis")
            logger.info(f"   📝 Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            logger.info(f"   🎤 Speaker: {Path(speaker_wav).name}")
            logger.info(f"   🌐 Language: {language}")
            
            # Split text into chunks if needed
            text_chunks = self._chunk_text_for_tts(text)
            
            if len(text_chunks) == 1:
                # Single chunk - BEST QUALITY synthesis
                logger.info(f"✅ Single chunk synthesis (highest quality)")
                
                # Create temp file if no output path provided
                use_temp = output_path is None
                final_path = output_path if output_path else os.path.join(tempfile.gettempdir(), "tts_output_temp.wav")
                
                try:
                    # Use tts_to_file for BEST quality (Coqui's recommended method)
                    # Note: temperature, top_p, speed parameters NOT SUPPORTED by tts_to_file
                    logger.info(f"   → Synthesizing to file: {final_path}")
                    self.tts.tts_to_file(
                        text=text_chunks[0],
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=final_path
                    )
                    
                    # Read and verify output
                    audio, sr = sf.read(final_path)
                    audio = audio.astype(np.float32)
                    logger.info(f"✅ Synthesis complete: {len(audio)} samples @ {sr}Hz")
                    
                    # Clean up temp file if used
                    if use_temp:
                        os.remove(final_path)
                    
                    return audio, sr
                
                except Exception as e:
                    logger.error(f"❌ tts_to_file failed: {str(e)}")
                    logger.warning("⚠️ Falling back to in-memory synthesis...")
                    
                    # Fallback: use in-memory tts() method
                    try:
                        output = self.tts.tts(
                            text=text_chunks[0],
                            speaker_wav=speaker_wav,
                            language=language
                        )
                        
                        if isinstance(output, (list, tuple)):
                            audio_data = np.array(output[0], dtype=np.float32) if output else np.array([])
                            sr = output[1] if len(output) > 1 else 24000
                        else:
                            audio_data = np.array(output, dtype=np.float32) if output is not None else np.array([])
                            sr = 24000
                        
                        logger.info(f"✅ Fallback synthesis complete: {len(audio_data)} samples @ {sr}Hz")
                        return audio_data, sr
                    
                    except Exception as e2:
                        logger.error(f"❌ Fallback also failed: {str(e2)}")
                        return np.array([]), 24000
            
            else:
                # Multiple chunks - synthesize and concatenate with quality optimization
                logger.info(f"✅ Multi-chunk synthesis ({len(text_chunks)} chunks)")
                
                concatenated_audio = np.array([], dtype=np.float32)
                sr = 24000
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, chunk in enumerate(text_chunks):
                        temp_path = os.path.join(tmpdir, f"chunk_{i}.wav")
                        
                        logger.info(f"   → Chunk {i+1}/{len(text_chunks)}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
                        
                        try:
                            # XTTSV2 tts_to_file for each chunk
                            self.tts.tts_to_file(
                                text=chunk,
                                speaker_wav=speaker_wav,
                                language=language,
                                file_path=temp_path
                            )
                            
                            chunk_audio, sr = sf.read(temp_path)
                            chunk_audio = chunk_audio.astype(np.float32)
                            
                            if len(concatenated_audio) == 0:
                                concatenated_audio = chunk_audio
                            else:
                                # Add small natural silence between chunks (~100ms)
                                silence = np.zeros(int(sr * 0.1), dtype=np.float32)
                                concatenated_audio = np.concatenate([concatenated_audio, silence, chunk_audio])
                            
                            logger.info(f"      ✅ Chunk {i+1} complete ({len(chunk_audio)} samples)")
                        
                        except Exception as e:
                            logger.error(f"      ❌ Chunk {i+1} failed: {str(e)}")
                            continue
                
                # Save final audio if output path provided
                if output_path and len(concatenated_audio) > 0:
                    sf.write(output_path, concatenated_audio, sr)
                    logger.info(f"✅ Saved synthesized audio: {output_path}")
                
                logger.info(f"✅ Multi-chunk synthesis complete: {len(concatenated_audio)} samples @ {sr}Hz")
                return concatenated_audio, sr
        
        except Exception as e:
            logger.error(f"❌ Critical TTS error: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([]), 24000
    
    def synthesize_with_prosody_control(self, text: str, speaker_wav: str, 
                                       language: str = "hi", speed: float = 1.0,
                                       output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech with prosody control (speed adjustment)
        
        Note: XTTSV2 doesn't natively support speed parameter in tts_to_file,
        so we apply post-processing with librosa for speed adjustment.
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference speaker audio
            language: Language code
            speed: Speech speed multiplier (0.5-2.0, where 1.0 = normal)
            output_path: Optional output path
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_data, sr = self.synthesize_speech(text, speaker_wav, language, output_path)
        
        # Apply speed adjustment if needed
        if speed != 1.0 and len(audio_data) > 0:
            try:
                import librosa
                logger.info(f"⚙️ Applying speed adjustment: {speed}x")
                audio_data = librosa.effects.time_stretch(audio_data, rate=speed)
                logger.info(f"✅ Speed adjustment applied")
            except Exception as e:
                logger.warning(f"⚠️ Speed adjustment failed: {str(e)}")
        
        return audio_data, sr
