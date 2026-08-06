"""
Tests for the AI layer: key resolution, the budget brief, and the chat call.

Run from the project root:

    python tests/test_ai_chat.py

**No API key and no network access are needed.** The Anthropic client is
replaced with a fake that records what it was sent and replays a scripted
response. That is the standard way to test code that calls a paid external
service: we are testing *our* logic — what we send, how we handle what comes
back — not Anthropic's servers.
"""

import os
import sys
from contextlib import contextmanager

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic  # noqa: E402

from src.ai_chat import (  # noqa: E402
    SYSTEM_INSTRUCTIONS,
    build_system_blocks,
    stream_answer,
)
from src.ai_client import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    ChatError,
    MissingAPIKeyError,
    build_client,
    describe_usage,
    is_configured,
    resolve_api_key,
    was_refused,
)
from src.ai_summary import (  # noqa: E402
    SUMMARY_INSTRUCTIONS,
    stream_summary,
)
from src.ai_summary import build_system_blocks as build_summary_blocks  # noqa: E402
from src.report_export import build_markdown_report  # noqa: E402
from src.analyzer import ColumnMapping, analyze_budget  # noqa: E402
from src.budget_context import build_context, estimate_size  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class FakeUsage:
    input_tokens = 1200
    output_tokens = 300
    cache_creation_input_tokens = 900
    cache_read_input_tokens = 0


class FakeFinalMessage:
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.model = MODEL
        self.usage = FakeUsage()


class FakeStream:
    """Mimics the context manager returned by client.beta.messages.stream()."""

    def __init__(self, chunks, stop_reason="end_turn", error=None):
        self.chunks = chunks
        self.stop_reason = stop_reason
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    @property
    def text_stream(self):
        if self.error:
            raise self.error
        return iter(self.chunks)

    def get_final_message(self):
        return FakeFinalMessage(self.stop_reason)


class FakeClient:
    """Records every request, so tests can assert on what we actually send."""

    def __init__(self, chunks=("Engineering ", "overspent by $128,900."),
                 stop_reason="end_turn", errors=None):
        self.chunks = list(chunks)
        self.stop_reason = stop_reason
        self.errors = list(errors or [])   # one error per call, popped in order
        self.calls = []

        client = self

        class _Messages:
            def stream(self, **kwargs):
                client.calls.append(kwargs)
                error = client.errors.pop(0) if client.errors else None
                return FakeStream(client.chunks, client.stop_reason, error)

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


def _bad_request(message):
    """Build a real BadRequestError without touching the network."""
    request = anthropic._base_client.httpx.Request("POST", "https://example.test")
    response = anthropic._base_client.httpx.Response(400, request=request)
    return anthropic.BadRequestError(message, response=response, body=None)


@contextmanager
def no_api_key_anywhere():
    """
    Pretend the machine has no Claude API key.

    Two things have to be suppressed, not one: the environment variable, and
    `load_dotenv()` reading a real `.env` file back into the environment. Once
    a developer adds a key, a test that only pops the variable silently starts
    testing the opposite of what it claims to.
    """
    import src.ai_client as ai_client

    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    saved_loader = ai_client.load_dotenv
    ai_client.load_dotenv = lambda *args, **kwargs: False
    try:
        yield
    finally:
        ai_client.load_dotenv = saved_loader
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key


def _analysis():
    df = pd.DataFrame({
        "Department": ["Engineering", "Marketing", "Operations"],
        "Budget": [1000, 500, 300],
        "Actual": [1400, 450, 300],
    })
    mapping = ColumnMapping(budget="Budget", actual="Actual", department="Department")
    return analyze_budget(df, mapping)


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------
def test_secrets_take_priority_over_environment():
    os.environ["ANTHROPIC_API_KEY"] = "sk-from-env"
    try:
        assert resolve_api_key({"ANTHROPIC_API_KEY": "sk-from-secrets"}) == "sk-from-secrets"
        assert resolve_api_key(None) == "sk-from-env"
    finally:
        del os.environ["ANTHROPIC_API_KEY"]


def test_missing_key_is_reported_not_crashed():
    """
    Runs as if no key existed anywhere.

    Both the environment variable and the `.env` loader are neutralised —
    otherwise this test would pass or fail depending on whether the machine
    running it happens to have a key configured, which is not a test at all.
    """
    with no_api_key_anywhere():
        # A secrets object that raises, as Streamlit does with no secrets file.
        class Exploding:
            def get(self, _):
                raise FileNotFoundError("no secrets file")

        assert resolve_api_key(Exploding()) is None
        assert is_configured(Exploding()) is False

        try:
            build_client(secrets=Exploding())
        except MissingAPIKeyError as error:
            assert "ANTHROPIC_API_KEY" in str(error)
        else:
            raise AssertionError("Expected MissingAPIKeyError")


def test_keys_are_stripped_of_whitespace():
    assert resolve_api_key({"ANTHROPIC_API_KEY": "  sk-padded  "}) == "sk-padded"


# ---------------------------------------------------------------------------
# The budget brief
# ---------------------------------------------------------------------------
def test_context_contains_the_computed_figures():
    context = build_context(_analysis(), "budget.xlsx")

    assert "$1,800" in context      # total budgeted
    assert "$2,150" in context      # total actual
    assert "$350" in context        # total variance
    assert "Engineering" in context
    assert "budget.xlsx" in context


def test_context_states_the_sign_convention():
    """Without this the model can read every variance backwards."""
    context = build_context(_analysis())
    assert "POSITIVE variance means OVER budget" in context


def test_context_discloses_data_quality_warnings():
    df = pd.DataFrame({
        "Department": ["Sales", "TOTAL"],
        "Budget": [100, 100],
        "Actual": [150, 150],
    })
    analysis = analyze_budget(
        df, ColumnMapping(budget="Budget", actual="Actual", department="Department")
    )
    context = build_context(analysis)
    assert "total/subtotal row" in context


def test_context_is_deterministic():
    """
    Byte-identical output for identical input is what makes caching work.

    A timestamp or random id anywhere in the brief would change the prompt
    prefix on every question, silently disabling the cache and paying full
    price for the brief every single turn.
    """
    analysis = _analysis()
    assert build_context(analysis, "b.xlsx") == build_context(analysis, "b.xlsx")


def test_estimate_size_reports_something_usable():
    size = estimate_size(build_context(_analysis()))
    assert size["characters"] > 100
    assert size["approx_tokens"] > 0


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------
def test_system_blocks_put_the_cache_breakpoint_last():
    blocks = build_system_blocks("BUDGET BRIEF")

    assert len(blocks) == 2
    assert blocks[0]["text"] == SYSTEM_INSTRUCTIONS
    assert "cache_control" not in blocks[0]
    # The marker caches everything before it, so it belongs on the last block.
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == "BUDGET BRIEF"


def test_system_prompt_forbids_inventing_causes():
    """The single most important instruction in the whole milestone."""
    assert "WHAT happened, not WHY" in SYSTEM_INSTRUCTIONS
    assert "hypothesis" in SYSTEM_INSTRUCTIONS


# ---------------------------------------------------------------------------
# The streaming call
# ---------------------------------------------------------------------------
def test_stream_yields_text_and_records_the_outcome():
    client = FakeClient()
    outcome = {}
    text = "".join(stream_answer(client, [{"role": "user", "content": "hi"}],
                                 "CTX", outcome))

    assert text == "Engineering overspent by $128,900."
    assert outcome["stop_reason"] == "end_turn"
    assert outcome["model"] == MODEL
    assert was_refused(outcome) is False


def test_request_sends_the_whole_conversation():
    """The API is stateless — history is only 'remembered' by resending it."""
    client = FakeClient()
    history = [
        {"role": "user", "content": "Who overspent?"},
        {"role": "assistant", "content": "Engineering."},
        {"role": "user", "content": "By how much?"},
    ]
    list(stream_answer(client, history, "CTX"))

    sent = client.calls[0]["messages"]
    assert len(sent) == 3
    assert sent[-1]["content"] == "By how much?"


def test_request_uses_the_configured_model_and_caps():
    client = FakeClient()
    list(stream_answer(client, [{"role": "user", "content": "hi"}], "CTX"))

    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["max_tokens"] == MAX_TOKENS
    assert call["output_config"]["effort"] == "medium"
    assert call["fallbacks"] == "default"       # refusal safety net, on by default


def test_falls_back_when_the_beta_parameter_is_rejected():
    """
    An account without the fallback beta must still get a working chat.

    The retry is only safe because a rejected parameter fails before any text
    is streamed — so nothing is duplicated.
    """
    client = FakeClient(errors=[_bad_request("unsupported beta: fallbacks")])
    text = "".join(stream_answer(client, [{"role": "user", "content": "hi"}], "CTX"))

    assert len(client.calls) == 2
    assert "fallbacks" in client.calls[0]
    assert "fallbacks" not in client.calls[1]
    assert text == "Engineering overspent by $128,900."


def test_no_credits_gets_an_actionable_message_not_raw_json():
    """
    The most common first-run failure, caught against the real API.

    An account with a valid key but no credits returns a 400 whose raw body is
    a JSON blob. The user needs to know the fix is billing, not code — and that
    API credits are separate from a Claude.ai subscription.
    """
    client = FakeClient(errors=[_bad_request(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'Your credit balance is too low "
        "to access the Anthropic API. Please go to Plans & Billing to upgrade "
        "or purchase credits.'}}"
    )])

    try:
        list(stream_answer(client, [{"role": "user", "content": "hi"}], "CTX"))
    except ChatError as error:
        assert "no API credits" in str(error)
        assert "console.anthropic.com" in str(error)
        assert "{" not in str(error), "Raw JSON leaked into the user message"
    else:
        raise AssertionError("Expected ChatError")

    # A billing error must not trigger the fallback retry — that would double
    # the failed requests for something a retry cannot fix.
    assert len(client.calls) == 1


def test_unrelated_400_is_not_retried():
    """Only a fallback-parameter 400 justifies a second request."""
    client = FakeClient(errors=[_bad_request("messages: text content blocks must be non-empty")])

    try:
        list(stream_answer(client, [{"role": "user", "content": "hi"}], "CTX"))
    except ChatError:
        pass
    else:
        raise AssertionError("Expected ChatError")

    assert len(client.calls) == 1


def test_refusal_is_detected_rather_than_shown_as_an_answer():
    """A refusal is a successful HTTP 200 with no answer, not an exception."""
    client = FakeClient(chunks=[], stop_reason="refusal")
    outcome = {}
    text = "".join(stream_answer(client, [{"role": "user", "content": "hi"}],
                                 "CTX", outcome))

    assert text == ""
    assert was_refused(outcome) is True


def test_api_errors_become_friendly_messages():
    request = anthropic._base_client.httpx.Request("POST", "https://example.test")
    response = anthropic._base_client.httpx.Response(401, request=request)
    client = FakeClient(errors=[
        anthropic.AuthenticationError("bad key", response=response, body=None)
    ])

    try:
        list(stream_answer(client, [{"role": "user", "content": "hi"}], "CTX"))
    except ChatError as error:
        assert "API key" in str(error)
    else:
        raise AssertionError("Expected ChatError")


def test_describe_usage_surfaces_cache_reads():
    outcome = {"usage": FakeUsage()}
    line = describe_usage(outcome)

    assert "1,200 input tokens" in line
    assert "read from cache" in line
    assert describe_usage({}) is None


# ---------------------------------------------------------------------------
# Milestone 5: the executive summary
# ---------------------------------------------------------------------------
def test_summary_asks_for_the_four_required_sections():
    """
    A summary with no fixed shape cannot be trusted twice.

    If the structure varies, a reader cannot tell whether a missing "Risks"
    section means there were no risks or that the model skipped the analysis.
    """
    for heading in ("Overall position", "What drove it",
                    "Risks to watch", "Recommended next steps"):
        assert heading in SUMMARY_INSTRUCTIONS, "Missing section: {}".format(heading)


def test_summary_carries_the_same_safety_rules_as_the_chat():
    """Both features share the rules that stop invented figures and causes."""
    assert "WHAT happened, not WHY" in SUMMARY_INSTRUCTIONS
    assert "Use only the figures in the brief" in SUMMARY_INSTRUCTIONS
    assert "positive variance means over budget" in SUMMARY_INSTRUCTIONS


def test_summary_uses_its_own_cached_system_prompt():
    blocks = build_summary_blocks("BUDGET BRIEF")

    assert blocks[0]["text"] == SUMMARY_INSTRUCTIONS
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
    # Different instructions from the chat means a separate cache entry —
    # caching is a prefix match, so a different first block is a new prefix.
    assert blocks[0]["text"] != SYSTEM_INSTRUCTIONS


def test_summary_sends_one_user_turn_and_the_brief():
    client = FakeClient(chunks=["**Overall position** ", "The budget is over."])
    outcome = {}
    text = "".join(stream_summary(client, "CTX", outcome))

    call = client.calls[0]
    assert len(call["messages"]) == 1        # one-shot, not a conversation
    assert call["messages"][0]["role"] == "user"
    assert call["system"][1]["text"] == "CTX"
    assert call["model"] == MODEL
    assert text.startswith("**Overall position**")
    assert outcome["stop_reason"] == "end_turn"


def test_summary_shares_the_chat_error_handling():
    """Both go through ai_client.stream_text, so both fail the same way."""
    client = FakeClient(errors=[_bad_request("unsupported beta: fallbacks")])
    text = "".join(stream_summary(client, "CTX"))

    assert len(client.calls) == 2            # retried without the beta
    assert "fallbacks" not in client.calls[1]
    assert text != ""


# ---------------------------------------------------------------------------
# The downloadable report
# ---------------------------------------------------------------------------
def test_report_includes_figures_and_the_ai_summary():
    report = build_markdown_report(_analysis(), "budget.xlsx", "AI SUMMARY TEXT")

    assert "# Budget Report" in report
    assert "budget.xlsx" in report
    assert "AI SUMMARY TEXT" in report
    assert "$1,800" in report                # total budgeted
    assert "| Engineering |" in report       # breakdown table


def test_report_attributes_the_summary_to_claude():
    """A reader must be able to tell which sentences a model wrote."""
    report = build_markdown_report(_analysis(), "b.xlsx", "AI SUMMARY TEXT")
    assert "Written by Claude" in report
    assert "not by the model" in report


def test_report_stands_alone_without_an_ai_summary():
    """No API key must still produce a complete, useful report."""
    report = build_markdown_report(_analysis(), "b.xlsx", None)

    assert "## Summary" in report
    assert "over budget by" in report        # the rule-written baseline
    assert "$1,800" in report
    assert "Written by Claude" not in report


def test_report_discloses_method_and_assumptions():
    report = build_markdown_report(_analysis(), "b.xlsx", None)
    assert "positive variance means over budget" in report
    assert "Budgeted column: `Budget`" in report


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0

    for test in tests:
        try:
            test()
            print("  PASS  {}".format(test.__name__))
        except Exception as error:  # noqa: BLE001 - report and continue
            failures += 1
            print("  FAIL  {}: {}".format(test.__name__, error))

    print("\n{} passed, {} failed".format(len(tests) - failures, failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
