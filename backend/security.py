"""
Authentication & Security Module
JWT-based authentication for FastAPI.
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from backend.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

security = HTTPBearer()


class TokenData(BaseModel):
    api_key: Optional[str] = None
    expires_at: Optional[datetime] = None


def create_access_token(api_key: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode = {"api_key": api_key, "exp": expire}
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.jwt_secret, 
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """
    Verify JWT token in Authorization header.
    Expected header: Authorization: Bearer <token>
    """
    token = credentials.credentials
    
    try:
        payload = jwt.decode(
            token, 
            settings.jwt_secret, 
            algorithms=[settings.jwt_algorithm]
        )
        api_key: str = payload.get("api_key")
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        return TokenData(api_key=api_key)
    except JWTError as e:
        logger.error("JWT verification failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


# Predefined API keys for different tiers (in production, store in DB + hash)
VALID_API_KEYS = {
    "test_key_123": {"name": "Test User", "tier": "free"},
    "prod_key_456": {"name": "Production User", "tier": "premium"},
}


def get_api_key_info(api_key: str) -> Optional[dict]:
    """Look up API key metadata."""
    return VALID_API_KEYS.get(api_key)


class SecurityHeaders:
    """CORS and security headers middleware."""
    
    @staticmethod
    def get_headers() -> dict:
        """Security headers for responses."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        }
