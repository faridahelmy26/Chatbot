import re
from typing import Optional
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for reproducibility
DetectorFactory.seed = 0


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect Arabic or English language
        
        Args:
            text: Input text
            
        Returns:
            'ar' for Arabic, 'en' for English
        """
        if not text:
            return "en"
        
        # Try using langdetect first
        try:
            lang = detect(text)
            if lang == "ar":
                return "ar"
            return "en"
        except LangDetectException:
            pass
        
        # Fallback to character detection
        if any("\u0600" <= c <= "\u06FF" for c in text):
            return "ar"
        
        return "en"
    
    @staticmethod
    def remove_diacritics(text: str) -> str:
        """
        Remove Arabic diacritics
        
        Args:
            text: Arabic text
            
        Returns:
            Text without diacritics
        """
        arabic_diacritics = re.compile(r"""
            ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ
        """, re.VERBOSE)
        
        return re.sub(arabic_diacritics, "", text)
    
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """
        Normalize Arabic text
        
        Args:
            text: Arabic text
            
        Returns:
            Normalized text
        """
        text = TextPreprocessor.remove_diacritics(text)
        
        replacements = {
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ى": "ي",
            "ؤ": "و",
            "ئ": "ي",
            "ة": "ه"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    @staticmethod
    def normalize_english(text: str) -> str:
        """
        Normalize English text
        
        Args:
            text: English text
            
        Returns:
            Normalized text
        """
        text = text.lower()
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
        text = text.replace("`", "'")
        
        return text
    
    @staticmethod
    def remove_special_characters(text: str) -> str:
        """
        Remove special characters
        
        Args:
            text: Input text
            
        Returns:
            Text with special characters removed
        """
        text = re.sub(r"[^\w\s']", " ", text)
        return text
    
    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """
        Remove extra spaces
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized spaces
        """
        return re.sub(r"\s+", " ", text).strip()
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Clean text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        language = TextPreprocessor.detect_language(text)
        
        if language == "ar":
            text = TextPreprocessor.normalize_arabic(text)
        else:
            text = TextPreprocessor.normalize_english(text)
        
        text = TextPreprocessor.remove_special_characters(text)
        text = TextPreprocessor.remove_extra_spaces(text)
        
        return text
    