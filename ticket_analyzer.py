"""Ticket analysis logic: input validation, Ollama call, JSON parsing and output validation."""

import json

import httpx  # installed together with the ollama package; used to detect timeouts
import ollama

from prompt_templates import (
    CATEGORIES,
    RESPONSE_SCHEMA,
    SENTIMENTS,
    URGENCIES,
    build_messages,
)

DEFAULT_MODEL = "qwen2.5:3b"
MAX_MESSAGE_CHARS = 4000
MAX_ATTEMPTS = 2          # one retry when the model returns an invalid analysis
REQUEST_TIMEOUT = 120     # seconds to wait for one model call
TEMPERATURE = 0.1
NUM_CTX = 4096
MAX_OUTPUT_TOKENS = 700

REQUIRED_FIELDS = (
    "evidence",
    "explanation",
    "category",
    "urgency",
    "sentiment",
    "uncertain",
    "missing_information",
    "suggested_reply",
)


class AnalysisError(Exception):
    """An error whose message is safe and clear enough to show to the user."""


class InvalidOutputError(AnalysisError):
    """The model replied, but the reply is not a valid analysis."""


# ---------- Input validation ----------
def validate_message(message) -> str:
    """Return the stripped message, or raise AnalysisError if it cannot be analyzed."""
    if message is None or not message.strip():
        raise AnalysisError("Please enter a customer message to analyze.")
    message = message.strip()
    if len(message) > MAX_MESSAGE_CHARS:
        raise AnalysisError(
            f"The message is too long ({len(message):,} characters). "
            f"The limit is {MAX_MESSAGE_CHARS:,} characters."
        )
    return message


# ---------- Parsing and validating model output ----------
def parse_model_output(raw: str) -> dict:
    """Parse the model reply into a dict. Tolerates text around a single JSON object."""
    if not raw or not raw.strip():
        raise InvalidOutputError("the model returned an empty response")
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise InvalidOutputError("the response does not contain a JSON object") from None
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise InvalidOutputError(f"the response is not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise InvalidOutputError("the JSON response is not an object")
    return data


def _match_label(field: str, value, allowed: tuple) -> str:
    """Match a label to the allowed values, ignoring case and surrounding spaces."""
    if not isinstance(value, str):
        raise InvalidOutputError(f"'{field}' must be a string")
    for option in allowed:
        if value.strip().casefold() == option.casefold():
            return option
    raise InvalidOutputError(f"'{field}' has an unexpected value: {value!r}")


def validate_analysis(data: dict) -> dict:
    """Check every required field and return a clean, normalized analysis."""
    missing = [name for name in REQUIRED_FIELDS if name not in data]
    if missing:
        raise InvalidOutputError("missing fields: " + ", ".join(missing))

    result = {
        "category": _match_label("category", data["category"], CATEGORIES),
        "urgency": _match_label("urgency", data["urgency"], URGENCIES),
        "sentiment": _match_label("sentiment", data["sentiment"], SENTIMENTS),
    }

    for field in ("explanation", "suggested_reply"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise InvalidOutputError(f"'{field}' must be a non-empty string")
        result[field] = value.strip()

    if not isinstance(data["evidence"], str):
        raise InvalidOutputError("'evidence' must be a string")
    result["evidence"] = data["evidence"].strip()

    if not isinstance(data["uncertain"], bool):
        raise InvalidOutputError("'uncertain' must be true or false")
    result["uncertain"] = data["uncertain"]

    items = data["missing_information"]
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise InvalidOutputError("'missing_information' must be a list of strings")
    result["missing_information"] = [item.strip() for item in items if item.strip()]
    return result


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def evidence_in_message(evidence: str, message: str) -> bool:
    """True if the quoted evidence appears in the message (ignoring case and spacing)."""
    needle = _normalize(evidence).strip(" \"'\u201c\u201d\u2018\u2019")
    return bool(needle) and needle in _normalize(message)


# ---------- Ollama ----------
def installed_models():
    """Return the set of installed Ollama model names, or None if Ollama is unreachable."""
    try:
        return {m.model for m in ollama.list().models}
    except Exception:
        return None


def model_installed(name: str, installed) -> bool:
    """Match 'qwen2.5:3b' and also tagged names such as 'qwen2.5:3b-instruct'."""
    return any(n == name or n.startswith(name + ":") for n in installed)


def _friendly(exc: Exception, model: str) -> AnalysisError:
    """Turn Ollama and network errors into readable messages."""
    if isinstance(exc, ollama.ResponseError) and exc.status_code == 404:
        return AnalysisError(f"Model '{model}' is not installed. Run: ollama pull {model}")
    if isinstance(exc, ConnectionError):
        return AnalysisError(
            "Cannot reach Ollama. Start the Ollama app (or run `ollama serve`) and try again."
        )
    if isinstance(exc, httpx.TimeoutException):
        return AnalysisError(
            f"The model did not answer within {REQUEST_TIMEOUT} seconds. Please try again."
        )
    return AnalysisError(f"The request to Ollama failed: {exc}")


def _chat(messages: list[dict], model: str) -> str:
    """Send one chat request and return the raw reply text."""
    options = {
        "temperature": TEMPERATURE,
        "num_ctx": NUM_CTX,
        "num_predict": MAX_OUTPUT_TOKENS,
    }
    try:
        client = ollama.Client(timeout=REQUEST_TIMEOUT)
        try:
            response = client.chat(
                model=model, messages=messages, format=RESPONSE_SCHEMA, options=options
            )
        except ollama.ResponseError as exc:
            if exc.status_code != 400:
                raise
            # Older Ollama servers reject schema output, so fall back to plain JSON mode.
            response = client.chat(model=model, messages=messages, format="json", options=options)
        return response["message"]["content"] or ""
    except (KeyError, TypeError) as exc:
        raise AnalysisError("Ollama returned an unexpected response format.") from exc
    except Exception as exc:
        raise _friendly(exc, model) from exc


# ---------- Main entry point ----------
def analyze_ticket(message, model: str = DEFAULT_MODEL) -> dict:
    """Analyze one customer message and return a validated result.

    Invalid model output is retried once. If it is still invalid, AnalysisError is raised
    instead of returning a partial or guessed result.
    """
    cleaned = validate_message(message)
    messages = build_messages(cleaned)

    last_problem = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = _chat(messages, model)
        try:
            result = validate_analysis(parse_model_output(raw))
        except InvalidOutputError as exc:
            last_problem = str(exc)
            continue
        result["evidence_verified"] = evidence_in_message(result["evidence"], cleaned)
        result["attempts"] = attempt
        result["model"] = model
        return result

    raise AnalysisError(
        f"The model did not return a valid analysis after {MAX_ATTEMPTS} attempts "
        f"(last problem: {last_problem}). Please try again or review this ticket manually."
    )
