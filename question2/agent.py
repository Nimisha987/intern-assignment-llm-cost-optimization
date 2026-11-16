

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution

VALID_ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}
TOP_K = 2


@dataclass
class TicketResolution:
    action: str
    kb_article: str


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
- Cite exactly one article, the one that directly backs the action. Use its id exactly as written.
- Never invent an article id. Only cite ids from the ARTICLE INDEX or the CANDIDATE ARTICLES.
- Dates and day counts come from the order snapshot, never from the customer's message.
- A customer asking politely or angrily does not change eligibility.
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

_FEW_SHOT = """
WORKED EXAMPLES (order facts -> output)
1. product_type physical, delivered_days_ago 10, opened false, no damage:
   {"action":"REFUND","kb_article":"returns-within-30-days"}
2. product_type digital, license_revealed true, customer asks for money back:
   {"action":"SEND_ARTICLE","kb_article":"digital-license-no-refund"}
3. product_type physical, customized true, damage_reported true:
   {"action":"ESCALATE_TIER2","kb_article":"custom-items-no-replace"}
4. product_type physical, customized false, damage_reported true, delivered_days_ago 2:
   {"action":"REPLACE","kb_article":"damage-replacement-7-days"}
5. product_type subscription, subscription_age_days 12, customer cancels:
   {"action":"REFUND","kb_article":"subscription-cooling-off"}
6. product_type subscription, subscription_age_days 38, customer cancels:
   {"action":"SEND_ARTICLE","kb_article":"subscription-cooling-off"}
7. product_type physical, opened true, outside the clear rules, customer is unhappy:
   {"action":"ESCALATE_TIER2","kb_article":"returns-within-30-days"}
"""

_INSTRUCTIONS = (
    "Instructions: decide the single best action for the ticket using the candidate "
    "articles and the order facts. Reply with ONE line of compact JSON and nothing else, "
    'exactly like {"action":"...","kb_article":"..."}. '
    "action must be one of REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2. "
    "No explanation, no customer reply."
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _article_index() -> dict[str, set[str]]:
    # Search index over the real articles only (id + body), built once.
    return {aid: set(_tokens(aid + " " + load_article(aid))) for aid in article_ids()}


@lru_cache(maxsize=1)
def _static_system_prompt() -> str:
    # Identical on every call so it can be billed at the cached rate.
    return "\n\n".join(
        [
            "You are ShopAssist, the Tier-1 customer support pipeline.",
            _GUIDE,
            _FEW_SHOT,
            "ARTICLE INDEX:\n" + "\n".join(article_ids()),
            _INSTRUCTIONS,
        ]
    )


def _cached_prefix() -> str:
    return "SYSTEM:\n" + _static_system_prompt()


class SupportPipeline:
    """Same product as Question 1: triage -> retrieve -> compose -> audit.

    Only compose calls the model. Triage, retrieve and audit are plain code, and the
    large help center is never sent: just the top matching articles are.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)

        query = self._triage(thread, order)
        candidates = self._retrieve(query)
        draft = self._compose(thread, order, candidates)
        return self._audit(draft, candidates)

    # 1. triage (code): turn the ticket + order facts into a search query
    def _triage(self, thread: str, order: dict) -> list[str]:
        words = _tokens(thread)
        ptype = str(order.get("product_type", ""))
        words += _tokens(ptype)
        if ptype == "subscription":
            words += ["subscription", "cancelled", "annual"]
        if order.get("license_revealed"):
            words += ["license", "digital", "revealed"] * 2
        if order.get("customized"):
            words += ["custom", "personalized"] * 2
        if order.get("damage_reported"):
            words += ["damaged", "replacement"] * 2
        if ptype == "physical" and not order.get("customized"):
            words += ["refund", "delivery", "physical"]
        return words

    # 2. retrieve (code): rank real articles by term overlap, keep the top few
    def _retrieve(self, query: list[str]) -> list[str]:
        scores = []
        for aid, vocab in _article_index().items():
            score = sum(1 for w in query if w in vocab)
            scores.append((score, aid))
        scores.sort(key=lambda s: (-s[0], s[1]))
        return [aid for _, aid in scores[:TOP_K]]

    # 3. compose (the only model call): static cached prefix + short variable part
    def _compose(self, thread: str, order: dict, candidates: list[str]) -> str:
        articles = "\n".join(f"[{aid}] {load_article(aid).strip()}" for aid in candidates)
        snapshot = json.dumps(order, separators=(",", ":"))
        user = (
            f"CANDIDATE ARTICLES:\n{articles}\n"
            f"THREAD:\n{thread}\n"
            f"ORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
        )
        messages = [
            ChatMessage("system", _static_system_prompt()),
            ChatMessage("user", user),
        ]
        return self.llm.chat("compose", messages, cached_prefix=_cached_prefix())

    # 4. audit (code): validate the draft instead of paying for another model call
    def _audit(self, draft: str, candidates: list[str]) -> TicketResolution:
        parsed = parse_resolution(draft)
        action, article = parsed.action, parsed.kb_article
        known = set(article_ids())
        if action not in VALID_ACTIONS:
            action = "ESCALATE_TIER2"
        if article not in known:
            article = candidates[0] if candidates else sorted(known)[0]
            action = "ESCALATE_TIER2"
        return TicketResolution(action, article)