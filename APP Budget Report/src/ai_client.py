"""
Claude API client setup, credentials, and the single streaming call.

Everything that talks to the API lives here. Two features now use Claude — the
chat (`ai_chat.py`) and the executive summary (`ai_summary.py`) — and both need
the same things: the same model settings, the same fallback handling, the same
translation of API errors into messages a user can read.

Rather than duplicate that in both, they share `stream_text()`. Each feature
supplies its own instructions and messages; this module owns the plumbing.

Never hard-code an API key. `.gitignore` already excludes `.env` and
`.streamlit/secrets.toml`; a key committed once stays in git history forever.
"""

import os
from typing import Any, Dict, Iterator, List, Mapping, Optional

import anthropic
from dotenv import load_dotenv

# The environment variable and secrets key we look for, in both places.
API_KEY_NAME = "ANTHROPIC_API_KEY"

# Model choice. Opus 5 is Anthropic's current flagship; the budget analysis
# itself is done in pandas, so what we need from the model is explanation
# quality, not arithmetic.
MODEL = "claude-opus-5"

# A cap, not a target — it exists so a runaway response cannot hang the app.
# Answers here are a few paragraphs, well under this.
MAX_TOKENS = 16000

# How hard the model works before answering. "medium" balances answer quality
# against latency; a budget question is reasoning, not research.
EFFORT = "medium"

# Opt-in server-side fallback. If Claude's safety classifiers decline a
# request, the API retries it on another model automatically instead of
# returning nothing. Unlikely to fire on budget questions, but it costs one
# parameter and turns a dead end into an answer.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class MissingAPIKeyError(RuntimeError):
    """Raised when no Claude API key can be found in any supported location."""


class ChatError(RuntimeError):
    """A user-presentable failure from the Claude API."""


def resolve_api_key(secrets: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """
    Find the Claude API key, checking each supported location in order.

    1. `secrets` — Streamlit's secrets store. This is how Streamlit Community
       Cloud supplies the key: App settings → Secrets. Nothing is written to
       disk in the repository.
    2. A `.env` file in the project root, for local development.
    3. A plain environment variable, for CI or a shell export.

    Returns None rather than raising, so the UI can show a friendly setup
    message instead of a stack trace.
    """
    if secrets is not None:
        try:
            key = secrets.get(API_KEY_NAME)
        except Exception:  # noqa: BLE001 - no secrets file configured at all
            key = None
        if key:
            return str(key).strip()

    # load_dotenv() is a no-op when there is no .env file, and never overrides
    # a variable that is already set in the environment.
    load_dotenv()

    key = os.environ.get(API_KEY_NAME)
    return key.strip() if key else None


def is_configured(secrets: Optional[Mapping[str, Any]] = None) -> bool:
    """True if a key is available. Used to decide which UI to show."""
    return resolve_api_key(secrets) is not None


def build_client(api_key: Optional[str] = None,
                 secrets: Optional[Mapping[str, Any]] = None) -> anthropic.Anthropic:
    """
    Create an Anthropic client.

    Raises:
        MissingAPIKeyError: no key was found in any supported location.
    """
    key = api_key or resolve_api_key(secrets)
    if not key:
        raise MissingAPIKeyError(
            "No Claude API key found. Set {} in a .env file locally, or under "
            "App settings → Secrets on Streamlit Community Cloud.".format(API_KEY_NAME)
        )
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# The shared streaming call
# ---------------------------------------------------------------------------
def _is_billing_error(message: str) -> bool:
    """
    True if a 400 is really "you have no credits".

    Worth its own message because it is the single most common first-run
    failure, and the fix has nothing to do with the code: API credits are
    bought separately from a Claude.ai subscription.
    """
    lowered = message.lower()
    return "credit balance" in lowered or "plans & billing" in lowered


def _is_fallback_error(message: str) -> bool:
    """True if a 400 is about the beta fallback parameter we opted into."""
    lowered = message.lower()
    return "fallback" in lowered or "beta" in lowered


def _request_kwargs(system: List[dict], messages: List[dict], effort: str,
                    with_fallback: bool) -> dict:
    """Build the arguments for one streaming request."""
    kwargs = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": system,
        "messages": messages,
        # Controls how hard the model works before answering. Both features
        # reason over a small brief, so "medium" is the sweet spot.
        "output_config": {"effort": effort},
    }
    if with_fallback:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"
    return kwargs


def stream_text(
    client: anthropic.Anthropic,
    system: List[dict],
    messages: List[dict],
    outcome: Optional[Dict] = None,
    effort: str = EFFORT,
) -> Iterator[str]:
    """
    Stream one response, yielding text as it arrives.

    Args:
        client:   An Anthropic client from build_client().
        system:   System prompt blocks, including any cache_control markers.
        messages: The conversation. The API is stateless — it has no memory of
                  previous calls — so anything it should "remember" has to be
                  in this list.
        outcome:  Optional dict, filled in with stop_reason / model / usage once
                  the stream finishes. A generator cannot return a value to a
                  caller that only iterates it, so the caller passes a dict in
                  and reads it afterwards.

    Raises:
        ChatError: any API failure, with a message safe to show a user.
    """
    outcome = outcome if outcome is not None else {}
    yielded_any = False

    for with_fallback in (True, False):
        try:
            kwargs = _request_kwargs(system, messages, effort, with_fallback)
            with client.beta.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yielded_any = True
                    yield text
                final = stream.get_final_message()

            outcome["stop_reason"] = final.stop_reason
            outcome["model"] = final.model
            outcome["usage"] = final.usage
            return

        except anthropic.BadRequestError as error:
            message = str(error)

            # A 400 can mean several unrelated things. Handle the ones a user
            # can actually act on before falling back to the raw message.
            if _is_billing_error(message):
                raise ChatError(
                    "Your Anthropic account has no API credits. The API is "
                    "billed separately from a Claude.ai subscription — add "
                    "credits at console.anthropic.com under Plans & Billing. "
                    "The report and charts work without credits."
                ) from error

            # The fallback parameter is a beta feature. If this account has not
            # been granted it, retry once without — better a working feature
            # with no safety net than no feature at all. The retry is only safe
            # because a rejected parameter fails before any text is streamed.
            #
            # Only retry when the error is actually about that parameter: a
            # blanket retry doubles the requests for every unrelated 400.
            if with_fallback and not yielded_any and _is_fallback_error(message):
                continue

            raise ChatError("Claude rejected the request: {}".format(error)) from error

        except anthropic.AuthenticationError as error:
            raise ChatError(
                "Your Claude API key was rejected. Check the key in your .env "
                "file, or in App settings → Secrets on Streamlit Cloud."
            ) from error

        except anthropic.RateLimitError as error:
            raise ChatError(
                "Rate limit reached. Wait a few seconds and try again."
            ) from error

        except anthropic.APIConnectionError as error:
            raise ChatError(
                "Could not reach the Claude API. Check your internet connection."
            ) from error

        except anthropic.APIStatusError as error:
            raise ChatError(
                "The Claude API returned an error ({}). Try again shortly.".format(
                    error.status_code
                )
            ) from error


def was_refused(outcome: Dict) -> bool:
    """
    True if Claude's safety classifiers declined the request.

    A refusal is a successful HTTP 200 with `stop_reason: "refusal"` and little
    or no text — not an exception. Code that assumes every 200 carries an
    answer will silently show an empty reply.
    """
    return outcome.get("stop_reason") == "refusal"


def describe_usage(outcome: Dict) -> Optional[str]:
    """
    One line of token accounting, for the UI's transparency panels.

    `cache_read_input_tokens` is the number worth watching: if it stays at zero
    across repeated requests, prompt caching is not working and every call is
    paying full price for the budget brief.
    """
    usage = outcome.get("usage")
    if usage is None:
        return None

    return (
        "{:,} input tokens · {:,} written to cache · {:,} read from cache · "
        "{:,} output tokens".format(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )
    )
