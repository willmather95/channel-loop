"""The SIMULATED account: one fictional product (Fieldnote), two ad accounts, eight campaigns.

Fieldnote is the same fictional product as the landing-page-lab example brief. Account ids
are made up. These are the starting settings. Hidden performance (how each campaign really
responds) is drawn per seed inside `sim.market` and never exposed to the agent.
"""
from __future__ import annotations

from dataclasses import dataclass

from ads_guardrail import Entity, Platform
from ads_guardrail.models import usd

GOOGLE_ACCOUNT = "9990001111"
META_ACCOUNT = "act_000000000000042"

LANDING_PAGE = "/trial"


@dataclass(frozen=True)
class CampaignSpec:
    platform: Platform
    account_id: str
    entity_id: str
    name: str
    budget: float  # starting daily budget, dollars
    bid: float  # starting tCPA (Google) or cost-cap bid (Meta), dollars
    kind: str  # brand | search | social | retarget (sets the simulator's hidden ranges)


CAMPAIGNS: tuple[CampaignSpec, ...] = (
    CampaignSpec(Platform.GOOGLE_ADS, GOOGLE_ACCOUNT, "31000000001", "Search | Brand", 120, 18, "brand"),
    CampaignSpec(Platform.GOOGLE_ADS, GOOGLE_ACCOUNT, "31000000002", "Search | Reminder Software", 420, 30, "search"),
    CampaignSpec(Platform.GOOGLE_ADS, GOOGLE_ACCOUNT, "31000000003", "Search | No-Show Reduction", 300, 28, "search"),
    CampaignSpec(Platform.GOOGLE_ADS, GOOGLE_ACCOUNT, "31000000004", "Search | Competitor Terms", 220, 34, "search"),
    CampaignSpec(Platform.GOOGLE_ADS, GOOGLE_ACCOUNT, "31000000005", "Search | Clinic Scheduling", 260, 30, "search"),
    CampaignSpec(Platform.META, META_ACCOUNT, "23800000001", "Meta | Clinic Owners Prospecting", 380, 26, "social"),
    CampaignSpec(Platform.META, META_ACCOUNT, "23800000002", "Meta | Lookalike 2%", 260, 24, "social"),
    CampaignSpec(Platform.META, META_ACCOUNT, "23800000003", "Meta | Site Retargeting", 140, 16, "retarget"),
)


def key_of(spec: CampaignSpec) -> str:
    return f"{spec.platform.value}:{spec.account_id}:{spec.entity_id}"


def starting_entities() -> list[Entity]:
    """The account as it is on day 0, in ads-guardrail-agent's own Entity type."""
    out = []
    for c in CAMPAIGNS:
        ref = (f"customers/{c.account_id}/campaignBudgets/{c.entity_id}" if c.platform is Platform.GOOGLE_ADS
               else c.entity_id)
        out.append(Entity(platform=c.platform, account_id=c.account_id, entity_id=c.entity_id, name=c.name,
                          daily_budget_micros=usd(c.budget), bid_micros=usd(c.bid), learning=False,
                          budget_shared=False, target_cpa_micros=usd(c.bid), ref=ref))
    return out


def spec_by_key() -> dict[str, CampaignSpec]:
    return {key_of(c): c for c in CAMPAIGNS}
