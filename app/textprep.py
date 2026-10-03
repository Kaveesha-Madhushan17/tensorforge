"""Text preparation shared by training and serving. Any change here changes the model."""
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    return _WS.sub(" ", s).strip().lower()


def model_input(channel: str, subject: str, text: str) -> str:
    """Only channel, subject and text reach the model. Never ticket_id or language."""
    return f"[{channel}] {normalize(subject)} || {normalize(text)}"
