"""
VoiceSpoof V2 - Main Streamlit Application
Beautiful UI for ASR + TTS Voice Spoofing Pipeline
"""
import streamlit as st
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import io
import soundfile as sf
import librosa

# Import handlers
from config import (
    SUPPORTED_LANGUAGES, DEVICE, ASR_MODEL_SIZE, TTS_MODEL_NAME,
    TEMP_AUDIO_DIR, OUTPUT_AUDIO_DIR, SAMPLE_RATE
)
from asr_handler import ASRHandler
from text_handler import TextHandler
from tts_handler import TTSHandler
from audio_handler import AudioHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="VoiceSpoof - ASR + TTS",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 16px;
            font-weight: 600;
            padding: 10px 20px;
        }
        .recording-indicator {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .success-box {
            padding: 1.5rem;
            border-radius: 8px;
            background-color: #0f3460;
            border-left: 4px solid #16c784;
            color: white;
        }
        .info-box {
            padding: 1.5rem;
            border-radius: 8px;
            background-color: #0f3460;
            border-left: 4px solid #3d9cff;
            color: white;
        }
        .error-box {
            padding: 1.5rem;
            border-radius: 8px;
            background-color: #0f3460;
            border-left: 4px solid #ff3d3d;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
@st.cache_resource
def initialize_handlers():
    """Initialize all handlers as cached resources"""
    logger.info("Initializing handlers...")
    asr = ASRHandler(model_size=ASR_MODEL_SIZE)
    text_handler = TextHandler()
    tts = TTSHandler(model_name=TTS_MODEL_NAME, device=DEVICE)
    audio = AudioHandler(sample_rate=SAMPLE_RATE)
    logger.info("All handlers initialized successfully")
    return asr, text_handler, tts, audio


def initialize_session_state():
    """Initialize session state variables"""
    if "language" not in st.session_state:
        st.session_state.language = None
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorded_audio" not in st.session_state:
        st.session_state.recorded_audio = None
    if "recorded_sr" not in st.session_state:
        st.session_state.recorded_sr = SAMPLE_RATE
    if "transcription" not in st.session_state:
        st.session_state.transcription = ""
    if "detected_language" not in st.session_state:
        st.session_state.detected_language = None
    if "text_choice" not in st.session_state:
        st.session_state.text_choice = "transcription"
    if "custom_text" not in st.session_state:
        st.session_state.custom_text = ""
    if "final_text" not in st.session_state:
        st.session_state.final_text = ""
    if "spoof_audio" not in st.session_state:
        st.session_state.spoof_audio = None
    if "spoof_sr" not in st.session_state:
        st.session_state.spoof_sr = 24000


def main():
    """Main application"""
    # Initialize
    initialize_session_state()
    asr, text_handler, tts, audio_handler = initialize_handlers()
    
    # Header
    st.markdown("# VoiceSpoof - ASR + TTS Pipeline")
    st.markdown("### Transform your voice with AI - Record → Transcribe → Generate Spoofed Speech")
    st.divider()
    
    # Sidebar - Language Selection
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.divider()
        
        selected_language = st.radio(
            " **Select Language:**",
            options=list(SUPPORTED_LANGUAGES.keys()),
            help="Choose the language for recording and speech synthesis"
        )
        
        st.session_state.language = SUPPORTED_LANGUAGES[selected_language]
        
        st.markdown(f"**Selected:** {selected_language}")
        st.markdown(f"**Language Code:** `{st.session_state.language}`")
        st.markdown(f"**Device:** `{DEVICE}`")
        st.divider()
        
        st.markdown("###  Pipeline Info")
        st.markdown("""
        **Step 1:**  Record your voice
        
        **Step 2:**  Transcribe with Whisper ASR
        
        **Step 3:**  Review/Edit text
        
        **Step 4:**  Process text
        
        **Step 5:**  Generate spoof with XTTSV2
        
        **Step 6:**  Download audio
        """)
    
    # Main content
    if st.session_state.language is None:
        st.warning("👈 Please select a language from the sidebar to begin!")
        return
    
    # Create tabs for workflow
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Record",
        " Transcribe",
        " Edit Text",
        " Generate",
        " Download"
    ])
    
    # TAB 1: Recording
    with tab1:
        st.markdown("##  Step 1: Record Your Voice")
        st.markdown(f"Recording in **{selected_language}** at {SAMPLE_RATE} Hz")
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Recording Controls")
            
            audio_data = st.audio_input(
                "🎙️ Click to record or upload an audio file:",
                label_visibility="collapsed",
                key="audio_input"
            )
            
            if audio_data is not None:
                # Convert audio bytes to numpy array using proper WAV parsing
                audio_bytes = audio_data.read()
                try:
                    # Use soundfile to properly parse WAV format from bytes
                    audio_buffer = io.BytesIO(audio_bytes)
                    audio_array, sr = sf.read(audio_buffer)
                    
                    # Ensure mono audio
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array.mean(axis=1)
                    
                    # Resample to target sample rate if needed
                    if sr != SAMPLE_RATE:
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
                    
                    st.session_state.recorded_audio = audio_array.astype(np.float32)
                    st.session_state.recorded_sr = SAMPLE_RATE
                    
                    logger.info(f"Audio parsed successfully: {len(audio_array)} samples at {SAMPLE_RATE} Hz")
                except Exception as e:
                    st.error(f" Failed to parse audio: {str(e)}")
                    logger.error(f"Audio parsing error: {str(e)}")
                    st.session_state.recorded_audio = None
                
                st.success(" Audio recorded successfully!")
        
        with col2:
            st.markdown("### Audio Info")
            if st.session_state.recorded_audio is not None:
                duration = audio_handler.get_audio_duration(
                    st.session_state.recorded_audio,
                    st.session_state.recorded_sr
                )
                st.metric("Duration", f"{duration:.2f}s")
                st.metric("Sample Rate", f"{st.session_state.recorded_sr} Hz")
                st.metric("Samples", len(st.session_state.recorded_audio))
        
        st.divider()
        
        if st.session_state.recorded_audio is not None:
            st.markdown("### Preview")
            st.audio(st.session_state.recorded_audio, sample_rate=st.session_state.recorded_sr)
    
    # TAB 2: Transcription
    with tab2:
        st.markdown("##  Step 2: Transcribe Audio")
        st.divider()
        
        if st.session_state.recorded_audio is None:
            st.warning(" Please record audio first in the **Record** tab!")
        else:
            if st.button(" Transcribe with Whisper ASR", use_container_width=True, key="transcribe_btn"):
                with st.spinner(" Transcribing audio..."):
                    try:
                        # Normalize audio
                        normalized_audio = audio_handler.normalize_audio(st.session_state.recorded_audio)
                        
                        # Transcribe
                        text, detected_lang = asr.transcribe_numpy_array(
                            normalized_audio,
                            sr=st.session_state.recorded_sr,
                            language=st.session_state.language
                        )
                        
                        st.session_state.transcription = text
                        st.session_state.detected_language = detected_lang
                        
                        st.success(" Transcription complete!")
                    
                    except Exception as e:
                        st.error(f" Error during transcription: {str(e)}")
                        logger.error(f"Transcription error: {str(e)}")
            
            if st.session_state.transcription:
                st.markdown("### Transcribed Text")
                st.info(f"**Language:** {st.session_state.detected_language}")
                st.text_area(
                    "Transcription Result:",
                    value=st.session_state.transcription,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
    
    # TAB 3: Edit Text
    with tab3:
        st.markdown("## Step 3: Review & Edit Text")
        st.divider()
        
        if not st.session_state.transcription:
            st.warning(" Please transcribe audio first in the **Transcribe** tab!")
        else:
            st.markdown("### Text Choice")
            
            choice = st.radio(
                "Choose text source:",
                options=["Use Transcription", "Enter Custom Text"],
                index=0 if st.session_state.text_choice == "transcription" else 1,
                horizontal=True
            )
            
            st.session_state.text_choice = "transcription" if choice == "Use Transcription" else "custom"
            
            if st.session_state.text_choice == "transcription":
                st.markdown("### Transcribed Text (Editable)")
                edited_text = st.text_area(
                    "Edit the transcribed text if needed:",
                    value=st.session_state.transcription,
                    height=200,
                    label_visibility="collapsed"
                )
                st.session_state.final_text = edited_text
            
            else:
                st.markdown("### Enter Custom Text")
                if selected_language == "Hindi":
                    st.info(" Enter text in Hindi")
                else:
                    st.info(" Enter text in English")
                
                custom_text = st.text_area(
                    "Enter custom text:",
                    value=st.session_state.custom_text,
                    height=200,
                    placeholder="Type your text here...",
                    label_visibility="collapsed"
                )
                st.session_state.custom_text = custom_text
                st.session_state.final_text = custom_text
    
    # TAB 4: Generate Spoof
    with tab4:
        st.markdown("## Step 4: Generate Spoofed Speech")
        st.divider()
        
        if not st.session_state.recorded_audio is None and st.session_state.final_text:
            st.markdown("### Text to Synthesize")
            st.info(st.session_state.final_text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                speed = st.slider("Speech Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            
            with col2:
                st.markdown("#### Speed Description")
                if speed < 1.0:
                    st.write(f"🐢 Slower ({speed}x)")
                elif speed > 1.0:
                    st.write(f"🐇 Faster ({speed}x)")
                else:
                    st.write("⏱️ Normal (1.0x)")
            
            st.divider()
            
            if st.button("✨ Generate Spoof Audio", use_container_width=True, key="generate_btn"):
                with st.spinner(" Generating spoofed speech... This may take a moment..."):
                    try:
                        # Save reference audio temporarily
                        temp_ref_path = str(TEMP_AUDIO_DIR / f"ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                        normalized_ref = audio_handler.normalize_audio(st.session_state.recorded_audio)
                        audio_handler.save_audio_file(normalized_ref, temp_ref_path, st.session_state.recorded_sr)
                        
                        # FOR ENGLISH SPEAKERS:
                        # Use English language model for better quality
                        # This ensures proper English pronunciation and natural accent
                        if selected_language == "English":
                            text_for_tts = st.session_state.final_text  # Keep English text as-is
                            tts_language = "en"  # Use English voice model for best quality
                            
                            logger.info(f"English Speaker - Superior Quality Mode:")
                            logger.info(f"   Transcription: {st.session_state.final_text}")
                            logger.info(f"   TTS Text: {text_for_tts}")
                            logger.info(f"   Voice Model: English (best quality for English text)")
                        
                        # For Hindi speakers, use transliteration approach
                        else:
                            text_for_tts = text_handler.get_processed_text_for_tts(
                                st.session_state.final_text,
                                target_language="hi",
                                original_language="hi"
                            )
                            tts_language = "hi"
                            logger.info(f"Hindi Speaker - Standard Mode:")
                            logger.info(f"   Text for TTS: {text_for_tts}")
                        
                        logger.info(f"Final TTS Parameters:")
                        logger.info(f"  Text: {text_for_tts[:100]}...")
                        logger.info(f"  Language: {tts_language}")
                        logger.info(f"  Speed: {speed}x")
                        
                        # Generate spoof audio
                        temp_output_path = str(TEMP_AUDIO_DIR / f"spoof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                        
                        spoof_audio, spoof_sr = tts.synthesize_with_prosody_control(
                            text=text_for_tts,
                            speaker_wav=temp_ref_path,
                            language=tts_language,
                            speed=speed,
                            output_path=temp_output_path
                        )
                        
                        st.session_state.spoof_audio = spoof_audio
                        st.session_state.spoof_sr = spoof_sr
                        
                        st.success(" Spoof audio generated successfully!")
                    
                    except Exception as e:
                        st.error(f" Error generating spoof: {str(e)}")
                        logger.error(f"Generation error: {str(e)}", exc_info=True)
            
            if st.session_state.spoof_audio is not None:
                st.markdown("### Generated Audio")
                st.audio(st.session_state.spoof_audio, sample_rate=st.session_state.spoof_sr)
        
        else:
            st.warning(" Please complete the previous steps (Record → Transcribe → Edit Text)")
    
    # TAB 5: Download
    with tab5:
        st.markdown("##  Step 5: Download Your Spoof Audio")
        st.divider()
        
        if st.session_state.spoof_audio is None:
            st.warning(" Please generate spoof audio first in the **Generate** tab!")
        else:
            st.markdown("### Your Generated Spoof Audio")
            st.audio(st.session_state.spoof_audio, sample_rate=st.session_state.spoof_sr)
            
            st.divider()
            st.markdown("### Download Options")
            
            # Create audio file
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, st.session_state.spoof_audio, st.session_state.spoof_sr, format='WAV')
            audio_buffer.seek(0)
            
            filename = f"spoof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            
            st.download_button(
                label=" Download Audio (WAV)",
                data=audio_buffer,
                file_name=filename,
                mime="audio/wav",
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duration", f"{len(st.session_state.spoof_audio) / st.session_state.spoof_sr:.2f}s")
            
            with col2:
                st.metric("Sample Rate", f"{st.session_state.spoof_sr} Hz")
            
            with col3:
                st.metric("Size", f"{len(audio_buffer.getvalue()) / (1024):.2f} KB")
    
    # Footer
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray; margin-top: 2rem;'>
            <p>🎙️ VoiceSpoof V2</p>
            <p><small>Made with ❤️ for voice research and testing</small></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
