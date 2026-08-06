from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from dotenv import load_dotenv
from typing import TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
import numpy as np
import os
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_openai_client = OpenAI()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-db")
def test_db():
    result = supabase.table("lead_lists").select("*").execute()
    return {"connected": True, "row_count": len(result.data)}


async def search_google(niche: str, location: str) -> list[dict]:
    """Searches Google Places for businesses matching niche + location.
    Fetches up to 2 pages (40 results) since Google caps each request at 20."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    query = f"{niche} in {location}"

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.internationalPhoneNumber,places.websiteUri,places.types,nextPageToken",
    }

    results = []
    body = {"textQuery": query}

    async with httpx.AsyncClient() as client:
        for _ in range(2):  # fetch up to 2 pages = up to 40 results
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()

            for place in data.get("places", []):
                results.append({
                    "business_name": place.get("displayName", {}).get("text"),
                    "address": place.get("formattedAddress"),
                    "phone": place.get("internationalPhoneNumber"),
                    "website": place.get("websiteUri"),
                    "category": place.get("types", [None])[0],
                    "source": "google",
                })

            next_token = data.get("nextPageToken")
            if not next_token:
                break

            # Google requires a short delay before the token becomes valid
            import asyncio
            await asyncio.sleep(2)
            body = {"textQuery": query, "pageToken": next_token}

    return results


async def search_instagram(niche: str, location: str) -> list[dict]:
    """Searches Instagram via a hashtag guess, calling Apify's REST API
    directly with httpx. Returns profile-shaped results — different
    fields than Google Places, since Instagram doesn't expose
    phone/address directly (uses bio instead)."""
    hashtag = f"{niche}{location}".replace(" ", "").lower()
    token = os.getenv("APIFY_API_TOKEN")

    run_url = (
        "https://api.apify.com/v2/acts/apify~instagram-hashtag-scraper"
        f"/run-sync-get-dataset-items?token={token}"
    )

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            run_url,
            json={"hashtags": [hashtag], "resultsLimit": 20},
        )
        response.raise_for_status()
        items = response.json()

    results = []
    seen_usernames = set()

    for item in items:
        owner = item.get("ownerUsername")
        if not owner or owner in seen_usernames:
            continue
        seen_usernames.add(owner)

        results.append({
            "business_name": owner,
            "address": "",
            "phone": "",
            "website": item.get("ownerWebsite", ""),
            "category": "instagram_profile",
            "bio": item.get("caption", "")[:200],
            "source": "instagram",
        })

    return results


async def search_facebook(niche: str, location: str) -> list[dict]:
    """Searches Facebook pages via Apify's Facebook Search Scraper —
    by keyword + location, matching real business pages. Same direct
    REST pattern as search_instagram (bypasses apify-client library)."""
    token = os.getenv("APIFY_API_TOKEN")

    run_url = (
        "https://api.apify.com/v2/acts/apify~facebook-search-scraper"
        f"/run-sync-get-dataset-items?token={token}"
    )

    run_input = {
        "queries": [niche],
        "locations": [location],
        "maxPages": 20,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(run_url, json=run_input)
        response.raise_for_status()
        items = response.json()

    # Safety net: log the raw shape of the first result so we can see
    # exactly what fields this actor actually returns, in case the
    # mapping below needs adjusting based on real output.
    if items:
        logger.info(f"[DEBUG] Facebook raw item sample: {items[0]}")

    results = []
    seen_names = set()

    for item in items:
        name = item.get("name") or item.get("pageName") or item.get("title")
        if not name or name in seen_names:
            continue
        seen_names.add(name)

        results.append({
            "business_name": name,
            "address": item.get("address", "") or item.get("location", ""),
            "phone": item.get("phone", "") or item.get("phoneNumber", ""),
            "website": item.get("website", ""),
            "category": item.get("category", "facebook_page"),
            "bio": item.get("about", "") or item.get("description", ""),
            "facebook_url": item.get("url", "") or item.get("pageUrl", ""),
            "source": "facebook",
        })

    return results


@app.get("/test-facebook")
async def test_facebook(niche: str, location: str):
    """Temporary isolated test route — remove once confirmed working."""
    try:
        results = await search_facebook(niche, location)
        return {"count": len(results), "results": results}
    except Exception as e:
        logger.exception("Facebook test failed")
        return {"error": str(e)}


@app.get("/search-leads")
async def search_leads(niche: str, location: str, sources: str = "google"):
    """Searches across the requested sources (google, instagram, facebook)
    and returns a combined, unified-shape result list. Any source that
    isn't yet connected or that errors is reported in unavailable_sources
    rather than silently ignored."""
    requested = [s.strip().lower() for s in sources.split(",")]
    unavailable = [s for s in requested if s not in ("google", "instagram", "facebook")]

    results = []

    if "google" in requested:
        try:
            results.extend(await search_google(niche, location))
        except Exception:
            logger.exception("Google search failed")
            unavailable.append("google (error)")

    if "instagram" in requested:
        try:
            results.extend(await search_instagram(niche, location))
        except Exception:
            logger.exception("Instagram search failed")
            unavailable.append("instagram (error)")

    if "facebook" in requested:
        try:
            results.extend(await search_facebook(niche, location))
        except Exception:
            logger.exception("Facebook search failed")
            unavailable.append("facebook (error)")

    return {
        "query": f"{niche} in {location}",
        "count": len(results),
        "results": results,
        "unavailable_sources": unavailable,
    }


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def dedup_leads(raw_leads: list[dict]) -> list[dict]:
    """Merges leads that are the same business under slightly different
    names, using OpenAI embedding similarity instead of exact string
    matching — catches 'Joe's Pizza' vs 'Joes Pizza LLC'."""
    if not raw_leads:
        return []

    texts = [
        f"{l['business_name']} {l.get('address', '')} {l.get('category', '')}"
        for l in raw_leads
    ]
    embeddings = np.array(get_embeddings(texts))

    keep = []
    used = set()

    for i, lead in enumerate(raw_leads):
        if i in used:
            continue
        keep.append(lead)
        used.add(i)
        for j in range(i + 1, len(raw_leads)):
            if j in used:
                continue
            similarity = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if similarity > 0.88:
                used.add(j)

    return keep


class LeadState(TypedDict):
    business_name: str
    address: str
    phone: str
    website: str
    category: str
    bio: str
    facebook_url: str
    extracted_signals: str
    score: str
    score_reason: str
    next: str


class SupervisorDecision(BaseModel):
    next: Literal["extract_signals", "score_directly"]


class ScoreResult(BaseModel):
    score: Literal["hot", "warm", "cold"]
    reason: str


def get_claude():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def lead_supervisor(state: LeadState) -> dict:
    llm = get_claude()
    structured = llm.with_structured_output(SupervisorDecision)

    decision = structured.invoke([
        SystemMessage(content=(
            "You supervise lead scoring. Given a business's raw data, decide: "
            "'score_directly' if the signals are already clear (e.g. has a "
            "website and looks like a big chain, or clearly has no website "
            "and looks independent — or, for Instagram/Facebook profiles, a "
            "bio that clearly indicates business type and activity level). "
            "'extract_signals' if it's ambiguous and needs a closer read of "
            "the available fields first."
        )),
        HumanMessage(content=(
            f"Business: {state['business_name']}\n"
            f"Category: {state['category']}\n"
            f"Website: {state['website'] or 'none'}\n"
            f"Address: {state['address'] or 'none'}\n"
            f"Bio: {state.get('bio') or 'none'}"
        )),
    ])
    return {"next": decision.next}


def signal_extractor(state: LeadState) -> dict:
    llm = get_claude()
    response = llm.invoke([
        SystemMessage(content=(
            "Look closely at this business's available data and note any "
            "inferable signals relevant to sales outreach — e.g. likely "
            "independent vs. chain, likely under-invested in online presence, "
            "likely price point. If this is a social profile (Instagram or "
            "Facebook, with no address/phone but has a bio), infer from the "
            "bio content instead. Be specific, 1-2 sentences, based only on "
            "what's given — don't invent facts not implied by the data."
        )),
        HumanMessage(content=(
            f"Business: {state['business_name']}\n"
            f"Category: {state['category']}\n"
            f"Website: {state['website'] or 'none'}\n"
            f"Address: {state['address'] or 'none'}\n"
            f"Bio: {state.get('bio') or 'none'}"
        )),
    ])
    return {"extracted_signals": response.content}


def scorer(state: LeadState) -> dict:
    llm = get_claude()
    structured = llm.with_structured_output(ScoreResult)

    context = f"Category: {state['category']}\nWebsite: {state['website'] or 'none'}"
    if state.get("address"):
        context += f"\nAddress: {state['address']}"
    if state.get("bio"):
        context += f"\nBio: {state['bio']}"
    if state.get("extracted_signals"):
        context += f"\nAdditional signal: {state['extracted_signals']}"

    result = structured.invoke([
        SystemMessage(content=(
            "Score this lead hot/warm/cold for sales outreach. Hot = strong "
            "opportunity (e.g. no website but clearly active business, or an "
            "active social profile with no linked website). Warm = some "
            "opportunity but less urgent. Cold = already well established "
            "online, low urgency. Give a short concrete reason."
        )),
        HumanMessage(content=f"Business: {state['business_name']}\n{context}"),
    ])
    return {"score": result.score, "score_reason": result.reason}


def route_from_supervisor(state: LeadState) -> str:
    return state["next"]


def build_lead_processor():
    graph = StateGraph(LeadState)
    graph.add_node("supervisor", lead_supervisor)
    graph.add_node("extract_signals", signal_extractor)
    graph.add_node("scorer", scorer)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"extract_signals": "extract_signals", "score_directly": "scorer"}
    )
    graph.add_edge("extract_signals", "scorer")
    graph.add_edge("scorer", END)

    return graph.compile()


_lead_processor = build_lead_processor()


class GuardrailResult(BaseModel):
    passes: bool
    reason: str


def draft_outreach_message(lead: dict) -> dict:
    llm = get_claude()

    draft = llm.invoke([
        SystemMessage(content=(
            "Write a short, genuine-sounding outreach message (2-3 "
            "sentences) to this business, referencing the specific reason "
            "they're a good lead. No generic sales language, no excessive "
            "enthusiasm, sound like a real person reaching out."
        )),
        HumanMessage(content=(
            f"Business: {lead['business_name']}\n"
            f"Why they're a good lead: {lead.get('score_reason', '')}"
        )),
    ]).content

    guardrail_llm = get_claude()
    structured_guardrail = guardrail_llm.with_structured_output(GuardrailResult)
    check = structured_guardrail.invoke([
        SystemMessage(content=(
            "Check this outreach message for generic/spammy phrasing "
            "('Hope this finds you well', 'I came across your business', "
            "excessive exclamation points, vague flattery). Fail it if "
            "present, note why."
        )),
        HumanMessage(content=draft),
    ])

    if not check.passes:
        draft = llm.invoke([
            SystemMessage(content=(
                f"Rewrite this message, avoiding: {check.reason}. "
                "Keep it short and genuine."
            )),
            HumanMessage(content=draft),
        ]).content

    return {"draft": draft}


@app.post("/process-leads")
def process_leads(raw_leads: list[dict]):
    deduped = dedup_leads(raw_leads)

    processed = []
    for lead in deduped:
        result = _lead_processor.invoke({
            "business_name": lead["business_name"],
            "address": lead.get("address", ""),
            "phone": lead.get("phone", ""),
            "website": lead.get("website", ""),
            "category": lead.get("category", ""),
            "bio": lead.get("bio", ""),
            "facebook_url": lead.get("facebook_url", ""),
            "extracted_signals": "",
            "score": "",
            "score_reason": "",
            "next": "",
        })
        processed.append({**lead, "score": result["score"], "score_reason": result["score_reason"]})

    return {"count": len(processed), "leads": processed}


@app.post("/draft-outreach")
def draft_outreach(lead: dict):
    return draft_outreach_message(lead)