"""
retry_utils.py — Shared retry-with-backoff wrapper for OpenAI chat-completion calls.

Discovered while running the HECA research-study evaluation: this OpenAI org has a
30,000 tokens-per-minute (TPM) rate limit for gpt-4o. Both app/generator.py and
app/language.py call client.chat.completions.create() with no retry logic, so a
transient 429 during normal (non-eval) traffic would surface to a real user as a
hard 502 from /query, and during evaluation it silently produced an empty ""
answer that then scored as a total failure in every downstream metric — a gap
that has nothing to do with the quality of the thing actually being measured.

call_with_retry() wraps any zero-arg callable (typically a lambda invoking
client.chat.completions.create(...)) and retries on RateLimitError with
exponential backoff, honouring the "please try again in Xms/Xs" hint OpenAI
includes in the error message when present.
"""

import re
import time
import logging

from openai import RateLimitError, APIConnectionError, APIError

logger = logging.getLogger(__name__)

_RETRY_HINT_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b")


def call_with_retry(fn, max_retries: int = 6, base_delay: float = 3.0, max_delay: float = 45.0):
    """Call fn() with no arguments; retry on rate limit / transient API errors."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError as exc:
            last_exc = exc
            wait = min(base_delay * (2 ** attempt), max_delay)
            m = _RETRY_HINT_RE.search(str(exc))
            if m:
                hinted = float(m.group(1)) / (1000.0 if m.group(2) == "ms" else 1.0)
                wait = max(wait, hinted + 0.5)
            logger.warning("Rate limited (attempt %d/%d) — waiting %.1fs", attempt + 1, max_retries, wait)
            time.sleep(wait)
        except (APIConnectionError, APIError) as exc:
            last_exc = exc
            wait = min(base_delay * (2 ** attempt), max_delay)
            logger.warning("Transient API error (attempt %d/%d): %s — waiting %.1fs", attempt + 1, max_retries, exc, wait)
            time.sleep(wait)
    # Exhausted retries — let the final attempt's exception propagate normally.
    raise last_exc
