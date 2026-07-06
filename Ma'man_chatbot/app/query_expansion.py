import re
from typing import Dict, List

class QueryExpansion:
    """Query expansion for better search results"""
    
    ARABIC_REPLACEMENTS = {
        "يعني ايه": "ما هو",
        "ايه هو": "ما هو",
        "ايه هي": "ما هي",
        "ما معنى": "ما هو",
        "عرفني": "ما هو",
        "عايز اعرف": "ما هو",
        "اريد معرفة": "ما هو",
        "ممكن اعرف": "ما هو",
        "لو سمحت": "",
        "من فضلك": "",
        "كيف": "طريقة",
        "كيفية": "طريقة",
        "شرح": "طريقة",
        "ايش": "ما",
        "شنو": "ما",
        "وين": "أين",
        "فين": "أين",
        "متى": "وقت",
        "امتى": "وقت",
        "ليش": "لماذا",
        "لماذا": "سبب",
        "اي": "ما",
        "منو": "من هو",
        "مين": "من هو",
    }
    
    ENGLISH_REPLACEMENTS = {
        "tell me about": "",
        "can you tell me": "",
        "i want to know": "",
        "please": "",
        "what's": "what is",
        "how to": "method",
        "how do i": "method",
        "what is": "definition",
        "why is": "reason",
        "where is": "location",
        "when is": "time",
        "who is": "person",
    }
    
    @staticmethod
    def expand(text: str, language: str) -> str:
        """
        Expand query with synonyms and variations
        
        Args:
            text: Input text
            language: 'ar' or 'en'
            
        Returns:
            Expanded query
        """
        if not text:
            return ""
        
        if language == "ar":
            replacements = QueryExpansion.ARABIC_REPLACEMENTS
        else:
            text = text.lower()
            replacements = QueryExpansion.ENGLISH_REPLACEMENTS
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    @staticmethod
    def get_keywords(text: str, language: str) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            language: 'ar' or 'en'
            
        Returns:
            List of keywords
        """
        # Remove stop words (simplified)
        stop_words = {
            'ar': ['و', 'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'كل', 'بعض', 'اي', 'اذا', 'ان'],
            'en': ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'our']
        }
        
        words = text.split()
        keywords = [w for w in words if w.lower() not in stop_words.get(language, [])]
        
        return keywords