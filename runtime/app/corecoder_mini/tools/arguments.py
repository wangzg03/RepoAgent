"""One bounded normalization for completed Todo metadata, never executable code."""

import hashlib
import json


def decode_arguments(name, text):
    """Return arguments and optional audit metadata; caller must still validate schema.

    Some complete responses contain one whole object followed by one redundant
    closing brace. Only todo_write may recover this exact form. Never repair
    truncated objects, lists, concatenated objects, trailing prose or write tools.
    This function is called only after the provider completion boundary.
    """
    try:
        return json.loads(text), None
    except json.JSONDecodeError as error:
        if name != "todo_write" or error.msg != "Extra data" or len(text) > 48000:
            raise
        original_error = error
    stripped = text.lstrip()
    arguments, end = json.JSONDecoder().raw_decode(stripped)
    if not isinstance(arguments, dict) or stripped[end:].strip() != "}":
        raise original_error
    normalized = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return arguments, {"rule": "todo_single_redundant_closing_brace", "tool": name,
                       "original_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                       "original_chars": len(text), "normalized_chars": len(normalized)}
