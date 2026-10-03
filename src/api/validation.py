"""Hand-written validation so we control exact error details (field, issue, index)."""
import re

from . import config
from src.model.labels import CHANNELS

_NON_WS = re.compile(r"\S")


def validate_ticket(obj, require_id: bool):
    """Return (clean_ticket, errors). errors is a list of {field, issue}."""
    errors = []
    if not isinstance(obj, dict):
        return None, [{"field": "ticket", "issue": "must be a JSON object"}]

    clean = {}

    # ticket_id
    if "ticket_id" in obj:
        tid = obj["ticket_id"]
        if not isinstance(tid, str):
            errors.append({"field": "ticket_id", "issue": "must be a string"})
        elif len(tid) > config.MAX_TICKET_ID_CHARS:
            errors.append({"field": "ticket_id", "issue": f"must be at most {config.MAX_TICKET_ID_CHARS} characters"})
        else:
            clean["ticket_id"] = tid
    elif require_id:
        errors.append({"field": "ticket_id", "issue": "is required"})

    # channel
    if "channel" not in obj:
        errors.append({"field": "channel", "issue": "is required"})
    elif not isinstance(obj["channel"], str) or obj["channel"] not in CHANNELS:
        errors.append({"field": "channel", "issue": "must be one of email, chat, call_transcript"})
    else:
        clean["channel"] = obj["channel"]

    # subject (optional, string only, null is not a string)
    subject = obj.get("subject", "")
    if "subject" in obj and not isinstance(subject, str):
        errors.append({"field": "subject", "issue": "must be a string"})
    elif len(subject) > config.MAX_SUBJECT_CHARS:
        errors.append({"field": "subject", "issue": f"must be at most {config.MAX_SUBJECT_CHARS} characters"})
    else:
        clean["subject"] = subject

    # text
    if "text" not in obj:
        errors.append({"field": "text", "issue": "is required"})
    else:
        text = obj["text"]
        if not isinstance(text, str):
            errors.append({"field": "text", "issue": "must be a string"})
        elif len(text) > config.MAX_TEXT_CHARS:
            errors.append({"field": "text", "issue": f"must be at most {config.MAX_TEXT_CHARS} characters"})
        elif not _NON_WS.search(text):
            errors.append({"field": "text", "issue": "must contain at least one non-whitespace character"})
        else:
            clean["text"] = text

    return (None if errors else clean), errors


def validate_single(body):
    clean, errors = validate_ticket(body, require_id=False)
    if errors and errors[0]["field"] == "ticket":
        errors = [{"field": "body", "issue": "must be a JSON object"}]
    return clean, errors


def validate_many(body, max_items: int):
    """Validate a {tickets: [...]} body. Atomic: return every failing item with its index."""
    if not isinstance(body, dict):
        return None, [{"field": "body", "issue": "must be a JSON object"}]
    if "tickets" not in body:
        return None, [{"field": "tickets", "issue": "is required"}]
    tickets = body["tickets"]
    if not isinstance(tickets, list):
        return None, [{"field": "tickets", "issue": "must be an array"}]
    if not 1 <= len(tickets) <= max_items:
        return None, [{"field": "tickets", "issue": f"must contain between 1 and {max_items} items"}]

    errors, clean_items, seen = [], [], set()
    for i, item in enumerate(tickets):
        clean, item_errors = validate_ticket(item, require_id=True)
        for e in item_errors:
            errors.append({"index": i, **e})
        if clean is not None:
            tid = clean["ticket_id"]
            if tid in seen:
                errors.append({"index": i, "field": "ticket_id", "issue": "duplicate of an earlier item"})
            seen.add(tid)
            clean_items.append(clean)
        elif isinstance(item, dict) and isinstance(item.get("ticket_id"), str):
            seen.add(item["ticket_id"])  # still catch duplicates of invalid items
    return (None if errors else clean_items), errors
