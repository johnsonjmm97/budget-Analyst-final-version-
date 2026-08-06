"""
The conversation layer: the chat's prompt and its message history.

The API call itself lives in ai_client.stream_text(), shared with the executive
summary. This module owns only what makes the *chat* a chat.

No Streamlit here either. `stream_answer()` is a plain generator of text
chunks, which is exactly what `st.write_stream()` consumes — but it would work
just as well in a terminal script.
"""

from typing import Dict, Iterator, List, Optional

import anthropic

from src.ai_client import EFFORT, stream_text

# Questions offered as one-click starters. These are the questions a budget
# owner actually asks, and they exercise different parts of the brief.
SUGGESTED_QUESTIONS = [
    "Which department overspent the most, and by how much?",
    "What are the biggest financial risks in this budget?",
    "Why did the largest overspend happen?",
    "Where did we underspend, and could that money be reallocated?",
    "Summarise this budget for a board meeting in five sentences.",
]

# ---------------------------------------------------------------------------
# The system prompt
#
# Three of these rules exist to prevent a specific, likely failure. Together
# they are what separates a useful analyst from a confident fabricator:
#
#   * "Only use the figures in the brief" stops invented numbers.
#   * "The data shows WHAT, not WHY" is the important one. A user will ask
#     "why did payroll exceed budget?" and the spreadsheet does not contain
#     the answer — it contains the amount, not the cause. Without this rule a
#     model will happily invent a plausible reason, and a plausible invented
#     reason is worse than no answer in a financial report.
#   * "Say when you cannot answer" makes the gap visible rather than papered
#     over.
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """\
You are a budget analyst assisting someone reviewing a specific budget. A brief
containing the analysed figures for that budget follows this message.

How to answer:

- Use only the figures in the brief. Never invent, estimate, or extrapolate a
  number that is not there. Quote the figures you rely on so the reader can
  check them.
- The data shows WHAT happened, not WHY. It records amounts, not causes. When
  asked why something happened, say plainly that the spreadsheet cannot answer
  that, then offer what the numbers *do* support: which line items drove the
  variance, how it compares to other areas, and what would be worth
  investigating. Label any suggested cause explicitly as a hypothesis to check,
  never as a finding.
- If the brief does not contain what is needed, say so directly and explain
  what extra data would answer the question. Do not pad an answer to seem
  complete.
- Remember the sign convention: a positive variance means over budget.
- Be concise and specific. A few short paragraphs or a tight list, not an
  essay. Lead with the answer, then the supporting figures.
- Write for an intelligent reader who is not an accountant. Explain financial
  terms the first time you use them.
- You are analysing a budget, not advising on investments. Do not give
  financial or investment advice.
"""


def build_system_blocks(context: str) -> List[dict]:
    """
    Assemble the system prompt as two blocks.

    The split matters. `cache_control` marks a caching breakpoint, and caching
    is a *prefix* match: everything up to the marker is cached together. So the
    stable instructions go first, the budget brief second, and the marker on
    the brief caches both.

    The result: the first question in a conversation pays for the whole brief,
    and every follow-up reads it from cache at a fraction of the cost. This is
    why the brief must be byte-identical between turns — a timestamp in it
    would silently invalidate the cache on every single question.
    """
    return [
        {"type": "text", "text": SYSTEM_INSTRUCTIONS},
        {
            "type": "text",
            "text": context,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def stream_answer(
    client: anthropic.Anthropic,
    messages: List[dict],
    context: str,
    outcome: Optional[Dict] = None,
    effort: str = EFFORT,
) -> Iterator[str]:
    """
    Stream one chat answer, yielding text as it arrives.

    Args:
        client:   An Anthropic client from ai_client.build_client().
        messages: The full conversation so far, ending with the user's new
                  question. The API is stateless, so "remembering" the
                  conversation just means resending all of it every time.
        context:  The budget brief from budget_context.build_context().
        outcome:  Optional dict, filled in with stop_reason / model / usage.

    Raises:
        ChatError: any API failure, with a message safe to show a user.
    """
    return stream_text(
        client=client,
        system=build_system_blocks(context),
        messages=messages,
        outcome=outcome,
        effort=effort,
    )
