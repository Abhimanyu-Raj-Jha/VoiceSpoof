"""
Text Handler - Handles text processing and transliteration
Based on reference pipeline implementation
Uses indic-transliteration library for proper ITRANS to Devanagari conversion
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Try to import indic_transliteration for proper transliteration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC_TRANSLITERATION = True
    logger.info("indic-transliteration library loaded successfully")
except ImportError:
    HAS_INDIC_TRANSLITERATION = False
    logger.warning("indic-transliteration not available, will use fallback method")


class TextHandler:
    """Handles text processing, transliteration (English to Hindi/Hinglish)"""
    
    def __init__(self):
        """Initialize text handler"""
        self.english_to_hindi_map = self._get_english_hindi_mapping()
    
    @staticmethod
    def _get_english_hindi_mapping() -> dict:
        """Get mapping of common English words to Hindi"""
        return {
            'hello': 'नमस्ते',
            'hi': 'नमस्ते',
            'how': 'कैसे',
            'are': 'हो',
            'you': 'आप',
            'thank': 'धन्यवाद',
            'thanks': 'धन्यवाद',
            'yes': 'हाँ',
            'no': 'नहीं',
            'ok': 'ठीक',
            'okay': 'ठीक',
            'bye': 'अलविदा',
            'good': 'अच्छा',
            'morning': 'सुबह',
            'night': 'रात',
            'please': 'कृपया',
            'what': 'क्या',
            'where': 'कहाँ',
            'when': 'कब',
            'why': 'क्यों',
            'who': 'कौन',
            'time': 'समय',
            'day': 'दिन',
            'today': 'आज',
            'tomorrow': 'कल',
        }
    
    @staticmethod
    def _english_to_hindi_phonetic(word: str) -> str:
        """
        Convert English to Hindi using character-by-character mapping
        Preserves word structure while converting to Devanagari script
        
        Args:
            word: English word to transliterate
        
        Returns:
            Phonetically transliterated Hindi text
        """
        word = word.lower()
        
        # Character-by-character mapping with multi-char patterns first
        # Order matters! Process longer patterns first
        replacements = [
            # Multi-character patterns (must come first)
            ('ch', 'च'),
            ('sh', 'श'),
            ('th', 'थ'),
            ('ph', 'फ'),
            ('gh', 'घ'),
            ('jh', 'झ'),
            ('kh', 'ख'),
            ('ng', 'ङ'),
            ('ck', 'क'),
            ('qu', 'क्व'),
            # Vowel combinations
            ('aa', 'आ'),
            ('ii', 'ई'),
            ('uu', 'ऊ'),
            ('ee', 'ई'),
            ('oo', 'ऊ'),
            ('ai', 'ऐ'),
            ('au', 'औ'),
            ('ou', 'ौ'),
            ('ea', 'ि'),
            # Single vowels
            ('a', 'अ'),
            ('e', 'े'),
            ('i', 'ि'),
            ('o', 'ो'),
            ('u', 'ु'),
            # Consonants
            ('b', 'ब'),
            ('c', 'क'),
            ('d', 'द'),
            ('f', 'फ'),
            ('g', 'ग'),
            ('h', 'ह'),
            ('j', 'ज'),
            ('k', 'क'),
            ('l', 'ल'),
            ('m', 'म'),
            ('n', 'न'),
            ('p', 'प'),
            ('q', 'क'),
            ('r', 'र'),
            ('s', 'स'),
            ('t', 'त'),
            ('v', 'व'),
            ('w', 'व'),
            ('x', 'क्स'),
            ('y', 'य'),
            ('z', 'ज'),
        ]
        
        result = word
        for eng, hindi in replacements:
            result = result.replace(eng, hindi)
        
        return result
    
    def transliterate_english_to_hindi(self, text: str) -> str:
        """
        Transliterate English text to Hindi using ITRANS to Devanagari conversion
        
        Args:
            text: English text (will be treated as ITRANS format)
        
        Returns:
            Hindi transliterated text in Devanagari script
        """
        try:
            if HAS_INDIC_TRANSLITERATION:
                # Use proper indic-transliteration library
                logger.info(f"Transliterating using indic-transliteration: {text}")
                result = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                logger.info(f"Transliteration result: {result}")
                return result
            else:
                # Fallback: use direct mapping and phonetic conversion
                logger.info(f"Using fallback phonetic transliteration: {text}")
                return self._fallback_transliterate(text)
        
        except Exception as e:
            logger.warning(f"Transliteration error: {str(e)}")
            return text
    
    def transliterate_hinglish_to_hindi(self, text: str) -> str:
        """
        Transliterate Hinglish (mixed English-Hindi) to pure Hindi
        Treats the input as ITRANS format and converts to Devanagari
        
        Args:
            text: Hinglish text
        
        Returns:
            Hindi transliterated text in Devanagari script
        """
        # For Hinglish, apply the same ITRANS transliteration logic
        return self.transliterate_english_to_hindi(text)
    
    def _fallback_transliterate(self, text: str) -> str:
        """
        Fallback transliteration using direct mapping and phonetic rules
        Used when indic-transliteration library is not available
        
        Args:
            text: English text
        
        Returns:
            Transliterated Hindi text
        """
        words = text.lower().split()
        transliterated_words = []
        
        for word in words:
            # Remove punctuation for processing
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            # Check if word exists in our direct mapping
            if clean_word in self.english_to_hindi_map:
                transliterated_words.append(self.english_to_hindi_map[clean_word])
            elif clean_word:
                # Use phonetic transliteration
                hindi_word = self._english_to_hindi_phonetic(clean_word)
                transliterated_words.append(hindi_word)
        
        result = ' '.join(transliterated_words)
        logger.info(f"Fallback transliterated: '{text}' -> '{result}'")
        return result
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra spaces, normalizing punctuation
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s+([।.!?])', r'\1', text)
        
        logger.info(f"Cleaned text: {text}")
        return text
    
    def get_processed_text_for_tts(self, text: str, target_language: str, 
                                   original_language: str = "en", 
                                   preserve_english_accent: bool = False) -> str:
        """
        Process text for TTS based on target language
        
        The key logic:
        - If preserve_english_accent=True and target_language='hi': Keep English text 
          but pass language='hi' to TTS. This keeps the English accent while using 
          Hindi voice model, avoiding bad transliteration.
        - If target_language is 'hi' and original_language is 'en': transliterate to Hindi
        - If target_language is 'hi' and original_language is 'hi': keep as Hindi
        - If target_language is 'en': keep in English
        
        Args:
            text: Input text
            target_language: Target language for TTS ('en' or 'hi')
            original_language: Original language of input ('en' or 'hi')
            preserve_english_accent: If True and target='hi', keep English text with Hindi voice
        
        Returns:
            Processed text ready for TTS
        """
        # Clean text first
        text = self.clean_text(text)
        
        # NEW APPROACH: For English speakers, keep English text but use Hindi voice
        # This preserves the natural English accent while using Hindi TTS
        if preserve_english_accent and target_language == 'hi' and original_language == 'en':
            logger.info(f"🎯 Preserving English accent: Keeping English text with Hindi voice model")
            logger.info(f"   Original English text: {text}")
            logger.info(f"   TTS Language: Hindi (but text remains English)")
            logger.info(f"   Result: English accent + Hindi voice = Natural spoof")
            return text  # Return English text as-is, TTS will handle with language='hi'
        
        # Original approach: If target is Hindi and input is English, transliterate
        elif target_language == 'hi' and original_language == 'en':
            processed_text = self.transliterate_english_to_hindi(text)
            logger.info(f"Processed for Hindi TTS (transliterated): {processed_text}")
            return processed_text
        
        # If target is Hindi and input is Hindi (or Hinglish), transliterate
        elif target_language == 'hi' and original_language == 'hi':
            processed_text = self.transliterate_hinglish_to_hindi(text)
            logger.info(f"Processed for Hindi TTS: {processed_text}")
            return processed_text
        
        # If target is English, keep as is
        else:
            logger.info(f"Keeping text in English for TTS")
            return text
