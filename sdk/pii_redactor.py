"""
sdk/pii_redactor.py
-------------------
PII (Personally Identifiable Information) redaction utility.

Scrubs sensitive patterns from text before it is stored in
logs or databases. Applied to input_preview and output_preview.

Patterns handled:
  - Email addresses        → [EMAIL]
  - Phone numbers          → [PHONE]
  - Credit card numbers    → [CARD]
  - US Social Security #s  → [SSN]
  - IPv4 addresses         → [IP]
"""

import re

# ---------------------------------------------------------------------------
# Compiled regex patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(
    r"(?:\+?1[\s\-.]?)?"                   # optional country code
    r"(?:\(?\d{3}\)?[\s\-.]?)"             # area code
    r"\d{3}[\s\-.]?\d{4}",                 # number
)

_CARD_RE = re.compile(
    r"\b(?:4[0-9]{12}(?:[0-9]{3})?"        # Visa
    r"|5[1-5][0-9]{14}"                    # MasterCard
    r"|3[47][0-9]{13}"                     # Amex
    r"|3(?:0[0-5]|[68][0-9])[0-9]{11}"    # Diners
    r"|6(?:011|5[0-9]{2})[0-9]{12}"        # Discover
    r"|(?:2131|1800|35\d{3})\d{11})\b"     # JCB
)

_SSN_RE = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)

_IPV4_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

# Ordered list of (pattern, replacement_token)
_PATTERNS = [
    (_CARD_RE,  "[CARD]"),    # cards first — they match digit runs
    (_SSN_RE,   "[SSN]"),
    (_EMAIL_RE, "[EMAIL]"),
    (_PHONE_RE, "[PHONE]"),
    (_IPV4_RE,  "[IP]"),
]


def redact(text: str) -> str:
    """
    Redact all known PII patterns from *text* and return the sanitised string.

    Parameters
    ----------
    text : str
        Raw text that may contain sensitive information.

    Returns
    -------
    str
        Text with PII replaced by placeholder tokens.
    """
    if not text:
        return text

    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)

    return text
