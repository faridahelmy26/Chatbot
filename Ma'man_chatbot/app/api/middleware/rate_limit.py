from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import time
from collections import defaultdict

# In-memory rate limiting for specific endpoints
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limit = 60  # requests
        self.window = 60  # seconds
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        self.requests[key] = [t for t in self.requests[key] if now - t < self.window]
        
        if len(self.requests[key]) >= self.limit:
            return False
        
        self.requests[key].append(now)
        return True
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests"""
        now = time.time()
        self.requests[key] = [t for t in self.requests[key] if now - t < self.window]
        return max(0, self.limit - len(self.requests[key]))

rate_limiter = RateLimiter()
