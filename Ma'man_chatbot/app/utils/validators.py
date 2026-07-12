from typing import Tuple, Optional
import re


def validate_question(question: str) -> Tuple[bool, str]:
    """
    Validate user question
    
    Args:
        question: User's question text
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not question or not isinstance(question, str):
        return False, "السؤال يجب أن يكون نصاً غير فارغ"
    
    question = question.strip()
    
    if len(question) < 3:
        return False, "السؤال قصير جداً (أقل من 3 أحرف)"
    
    if len(question) > 1000:
        return False, "السؤال طويل جداً (أكثر من 1000 حرف)"
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'alert\(',
        r'<iframe',
        r'onclick=',
        r'onload=',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return False, "السؤال يحتوي على محتوى غير آمن"
    
    return True, ""


def validate_language(language: str) -> bool:
    """
    Validate language code
    
    Args:
        language: Language code (ar/en)
        
    Returns:
        True if valid, False otherwise
    """
    return language in ["ar", "en"]


def validate_similarity(similarity: float) -> bool:
    """
    Validate similarity score
    
    Args:
        similarity: Similarity score (0.0 - 1.0)
        
    Returns:
        True if valid, False otherwise
    """
    return isinstance(similarity, (int, float)) and 0.0 <= similarity <= 1.0


def sanitize_text(text: str) -> str:
    """
    Sanitize text to prevent XSS and SQL injection
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove potential XSS
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove potential SQL
    text = re.sub(r"['\";]", '', text)
    
    # Limit length
    text = text[:1000]
    
    return text.strip()


def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email address
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
