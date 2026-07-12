from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import Optional
from datetime import datetime, timedelta
import jwt
from dotenv import load_dotenv

load_dotenv()

security = HTTPBearer()

# =====================================================
# JWT Configuration
# =====================================================

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "123456")  # 👈 Default value for testing


def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token for admin authentication"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    return encoded_jwt


def decode_jwt_token(token: str) -> dict:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )


# =====================================================
# Main Authentication Dependency
# =====================================================

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify JWT token from Authorization header
    
    Supports:
    1. JWT Bearer Token: Authorization: Bearer <jwt-token>
    2. API Key: X-API-Key: <api-key> (fallback)
    """
    token = credentials.credentials
    
    # 🔥 IMPORTANT: Accept the token from /generate-token
    # If the token starts with 'eyJ', it's a JWT token
    if token.startswith('eyJ'):
        try:
            payload = decode_jwt_token(token)
            return token
        except HTTPException:
            # If JWT fails, try API Key
            if ADMIN_TOKEN and token == ADMIN_TOKEN:
                return token
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    # Try API Key
    if ADMIN_TOKEN and token == ADMIN_TOKEN:
        return token
    
    # Try JWT validation (fallback)
    try:
        payload = decode_jwt_token(token)
        return token
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


# =====================================================
# Helper Functions for Testing
# =====================================================

def get_test_token() -> str:
    """Generate a test token for development"""
    return create_jwt_token({"sub": "admin", "role": "admin"})


def get_admin_headers() -> dict:
    """Get headers with admin authentication for testing"""
    token = get_test_token()
    return {"Authorization": f"Bearer {token}"}