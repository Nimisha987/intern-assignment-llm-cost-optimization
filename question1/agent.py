from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from shared.billing import CostSession
from shared.kb import all_articles_blob
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


_FEW_SHOT = """
Example ticket resolutions (format to copy):

1. Unopened headphones, day 10 → {"action":"REFUND","kb_article":"returns-within-30-days"}
2. Ebook, license shown → {"action":"SEND_ARTICLE","kb_article":"digital-license-no-refund"}
3. Cracked custom mug → {"action":"ESCALATE_TIER2","kb_article":"custom-items-no-replace"}
4. Cracked stock mug, day 2 → {"action":"REPLACE","kb_article":"damage-replacement-7-days"}
"""

# Static policy guidance. It lives in the cacheable prefix, so it is billed at the
# cached rate and it also makes the routing rules explicit for unseen tickets.
_GUIDE = """
ACTIONS
- REFUND: money goes back to the customer. Use only when a help-center article makes the order eligible.
- REPLACE: ship a free replacement. Use only for standard catalog items reported damaged inside the replacement window.
- SEND_ARTICLE: the request is not eligible and the customer only needs the policy explained.
- ESCALATE_TIER2: the case needs a human specialist, for example customized items, or anything the articles do not clearly cover.

ORDER_SNAPSHOT FIELDS
- order_id, sku, title: identify the order and the product.
- product_type: physical, digital, or subscription.
- delivered_days_ago: whole days since delivery of a physical item.
- opened: true when the customer opened the packaging.
- license_revealed: true when a digital license key or download link was already shown to the customer.
- customized: true when the item was personalized or made to order.
- damage_reported: true when the customer reports the item arrived damaged.
- subscription_age_days: days since a subscription was purchased, when present.

ROUTING ORDER (apply the first rule that matches)
1. Digital product with license_revealed true: SEND_ARTICLE with digital-license-no-refund.
2. Customized item with a damage report or a replacement request: ESCALATE_TIER2 with custom-items-no-replace.
3. Standard catalog item with damage_reported true and delivered within 7 days: REPLACE with damage-replacement-7-days.
4. Physical item, resalable condition, refund requested within 30 days of delivery: REFUND with returns-within-30-days.
5. Subscription cancelled within 30 days of purchase: REFUND with subscription-cooling-off.
6. Subscription older than 30 days: SEND_ARTICLE with subscription-cooling-off.
7. Anything else, or any conflict between the customer's claim and the order facts: ESCALATE_TIER2.

RULES FOR ANSWERING
- Trust the order facts over the customer's wording when they disagree.
- Cite exactly one article, the one that directly backs the action. Use its id exactly as written in the help center.
- Never invent an article id. If no article fits, choose ESCALATE_TIER2 and cite the closest article.
- Dates and day counts come from the order snapshot, never from the customer's message.
- A customer asking politely or angrily does not change eligibility.
- Opened physical items follow the opened-goods rule mentioned in returns-within-30-days; when unsure about condition, escalate.
- Digital products and subscriptions are never replaced. Physical damage is the only replacement reason.
- Customized items are never replaced automatically and never refunded automatically.
- If the thread contains several requests, route the most recent open request.
- The output is consumed by a parser: emit valid JSON, double-quoted keys, no markdown fences, no trailing text.
COMMON PATTERNS
- Wants money back, item unopened, delivered 10 days ago: REFUND, returns-within-30-days.
- Wants money back for an ebook or software key already revealed: SEND_ARTICLE, digital-license-no-refund.
- Mug, phone case or other personalized item arrived cracked: ESCALATE_TIER2, custom-items-no-replace.
- Stock catalog item arrived broken two days ago: REPLACE, damage-replacement-7-days.
- Annual plan cancelled in its first month: REFUND, subscription-cooling-off.
- Annual plan cancelled well after the first month: SEND_ARTICLE, subscription-cooling-off.
"""

_INSTRUCTIONS = (
    "Instructions: decide the single best action for the ticket using the help center "
    "and the order facts. Reply with ONE line of compact JSON and nothing else, exactly "
    'like {"action":"...","kb_article":"..."}. '
    "action must be one of REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2. "
    "kb_article must be the id of the help-center article that backs the decision. "
    "No explanation, no customer reply."
)


@lru_cache(maxsize=1)
def _static_system_prompt() -> str:
    # Byte-identical on every call (no run ids, timestamps or ticket ids) so the
    # whole prefix can be billed at the cached rate.
    return "\n\n".join(
        [
            "You are ShopAssist, the Tier-1 customer support copilot.",
            _GUIDE,
            _FEW_SHOT,
            "HELP CENTER:",
            all_articles_blob(),
            _INSTRUCTIONS,
        ]
    )


def _cached_prefix() -> str:
    # LlmClient builds the prompt as "SYSTEM:\n<content>\n\nUSER:\n..." and only
    # credits caching when the prompt starts with cached_prefix.
    return "SYSTEM:\n" + _static_system_prompt()


class SupportCopilot:
    """ShopAssist Tier-1 copilot (cost-optimized)."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        user = f"THREAD:\n{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
        return [
            ChatMessage("system", _static_system_prompt()),
            ChatMessage("user", user),
        ]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat(
            "handle_ticket", messages, cached_prefix=_cached_prefix()
        )
        return parse_resolution(completion)