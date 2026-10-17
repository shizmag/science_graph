import time
import asyncio

class AsyncRateLimiter:
    """Controls frequency of asynchronous operations to fit rate-limits (RPM)."""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_call_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        """Suspends caller task if rate-limit interval hasn't been met yet."""
        if self.interval <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            target_time = max(now, self.last_call_time + self.interval)
            self.last_call_time = target_time
            sleep_duration = target_time - now

        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)

    def can_acquire(self) -> bool:
        """Checks if a call can be made immediately without waiting."""
        if self.interval <= 0:
            return True
        return (time.monotonic() - self.last_call_time) >= self.interval

    def has_capacity(self) -> bool:
        """Alias for can_acquire(). Checks if rate limit slot is available."""
        return self.can_acquire()

