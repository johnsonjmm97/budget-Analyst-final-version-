"""
The AI executive summary.

One request, one written summary of the whole budget — the thing a manager
reads instead of the spreadsheet.

This shares everything with the chat except its instructions: the same budget
brief, the same client, the same streaming call. The difference is the *job*.
The chat answers a question the reader already has; the summary decides what
the reader should be told before they know what to ask. That is a genuinely
harder task, and it is the clearest demonstration of what a language model adds
over the rule-written `headline_sentence()` from Milestone 3.
"""

from typing import Dict, Iterator, List, Optional

import anthropic

from src.ai_client import EFFORT, stream_text

# ---------------------------------------------------------------------------
# The summary instructions
#
# These carry the same three safety rules as the chat prompt — use only the
# figures given, never invent a cause, say when the data cannot answer — plus
# a required structure. Structure matters more here than in chat: a summary
# with no fixed shape is read once and never trusted again, because the reader
# cannot tell whether the absence of a "Risks" section means no risks or no
# analysis.
# ---------------------------------------------------------------------------
SUMMARY_INSTRUCTIONS = """\
You are a budget analyst writing an executive summary for senior management. A
brief containing the analysed figures follows this message.

Write the summary in Markdown with exactly these four sections:

**Overall position** — one or two sentences: is the budget over or under, by
how much, and by what percentage. Lead with the number.

**What drove it** — the two or three largest contributors to the variance,
each with its figure. Name the department or category.

**Risks to watch** — what in these numbers should concern management. A
department trending badly, a category far over in percentage terms, an
unusually large single variance. If the figures show no material risk, say so
rather than manufacturing one.

**Recommended next steps** — two or three concrete actions, each tied to a
figure above. Prefer "ask X why Y is Z% over" to vague advice.

Rules:

- Use only the figures in the brief. Never invent, estimate, or extrapolate a
  number that is not there. Quote the figures you rely on.
- The data shows WHAT happened, not WHY. It records amounts, not causes. Do not
  assert a reason for any variance. Where a cause would be useful, phrase it as
  a question to investigate, explicitly labelled as such.
- Remember the sign convention: a positive variance means over budget.
- Keep the whole summary under 300 words. It is a summary, not a report.
- Write for an intelligent reader who is not an accountant. Explain financial
  terms the first time you use them.
- Do not add a title, a preamble, or a closing sentence about being happy to
  help. Start directly with the first section heading.
- You are analysing a budget, not advising on investments. Do not give
  financial or investment advice.
"""

# The single user turn. The brief is in the system prompt, so this only has to
# state the task — which keeps the cacheable prefix stable across regenerations.
SUMMARY_REQUEST = "Write the executive summary for this budget."


def build_system_blocks(context: str) -> List[dict]:
    """
    Assemble the summary's system prompt.

    Same two-block shape as the chat: stable instructions first, the budget
    brief second, and the `cache_control` marker on the brief so both are
    cached together.

    Note the instructions differ from the chat's, which means the summary has
    its own cache entry — caching is a prefix match, so a different first block
    is a different prefix. That is fine and expected; the benefit here is that
    regenerating the summary, or generating it after using the chat, still
    reuses the cached brief.
    """
    return [
        {"type": "text", "text": SUMMARY_INSTRUCTIONS},
        {
            "type": "text",
            "text": context,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def stream_summary(
    client: anthropic.Anthropic,
    context: str,
    outcome: Optional[Dict] = None,
    effort: str = EFFORT,
) -> Iterator[str]:
    """
    Stream the executive summary, yielding text as it arrives.

    Args:
        client:  An Anthropic client from ai_client.build_client().
        context: The budget brief from budget_context.build_context().
        outcome: Optional dict, filled in with stop_reason / model / usage.

    Raises:
        ChatError: any API failure, with a message safe to show a user.
    """
    return stream_text(
        client=client,
        system=build_system_blocks(context),
        messages=[{"role": "user", "content": SUMMARY_REQUEST}],
        outcome=outcome,
        effort=effort,
    )
