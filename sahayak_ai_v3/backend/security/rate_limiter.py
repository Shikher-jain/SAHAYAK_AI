import os
import time
import logging
from upstash_redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Serverless Upstash Redis HTTP Client Initialization
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

redis = None
if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
    redis = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)

# Atomic Token Bucket Lua Script
# KEYS[1]: User Identifier (IP or ID)
# ARGV[1]: Capacity (Max Tokens)
# ARGV[2]: Refill Rate (Tokens per Second)
# ARGV[3]: Current Timestamp (Unix Seconds)
LUA_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

local data = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

if tokens == nil then
    -- First time seeing this user, fill bucket
    tokens = capacity
    last_refill = current_time
else
    -- Calculate tokens generated since last refill
    local elapsed = current_time - last_refill
    local generated = elapsed * refill_rate
    
    tokens = math.min(capacity, tokens + generated)
    last_refill = current_time
end

-- If enough tokens, consume one and accept
if tokens >= 1 then
    tokens = tokens - 1
    -- Save state and set a TTL of 1 hour to free memory for inactive users
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
    redis.call('EXPIRE', key, 3600)
    return {1, tokens}
else
    -- Reject request
    return {0, 0}
end
"""

class TokenBucketRateLimiter:
    """Asynchronous Token Bucket rate limiter using Upstash Redis over REST."""
    
    async def acquire(self, identifier: str, capacity: int = 10, refill_rate: float = 0.5) -> bool:
        """
        Attempts to consume a token. 
        Returns True if successful, False if rate limited.
        """
        if not redis:
            logger.warning("Upstash Redis not configured. Rate limiter is completely BYPASSED.")
            return True
            
        current_time = time.time()
        
        try:
            # Execute the atomic Lua script over REST
            result = await redis.eval(
                LUA_SCRIPT,
                keys=[f"ratelimit:{identifier}"],
                args=[capacity, refill_rate, current_time]
            )
            
            allowed = result[0] == 1
            if not allowed:
                logger.warning(f"Rate limit exceeded for identifier: {identifier}")
                
            return allowed
            
        except Exception as e:
            logger.error(f"Rate Limiter Redis Error: {e}")
            # Fail-open design: if Redis is down, don't break the app (unless strict is needed)
            return True

rate_limiter = TokenBucketRateLimiter()
