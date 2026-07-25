from fastapi import FastAPI
from supabase import create_client
from dotenv import load_dotenv
from typing import TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph,START,END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI
from langchain_openai import ChatOpenAI
import numpy as np
import os


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

_openai_client = OpenAI()

app = FastAPI()

   
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-db")
def test_db():
    result = supabase.table("lead_lists").select("*").execute()
    return {"connected": True, "row_count": len(result.data)}

import httpx


@app.get("/search-leads")
async def search_leads(niche: str, location: str):
    """Searches Google Places for businesses matching niche + location.
    No AI, no cleanup yet — just proves real data flows in correctly."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    query = f"{niche} in {location}"

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.internationalPhoneNumber,places.websiteUri,places.types",
    }
    body = {"textQuery": query}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    results = []
    for place in data.get("places", []):
        results.append({
            "business_name": place.get("displayName", {}).get("text"),
            "address": place.get("formattedAddress"),
            "phone": place.get("internationalPhoneNumber"),
            "website": place.get("websiteUri"),
            "category": place.get("types", [None])[0],
        })

    return {"query": query, "count": len(results), "results": results}

def get_embeddings(texts: list[str])->list[list[float]]:
    response = _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def dedup_leads(raw_leads: list[dict])->list[dict]:
    """Merges leads that are the same business under slightly different
    names,using OpenAI embedding similarity instead of exact string
    matching-catches 'Joe's Pizza' vs 'Joes Pizza LLC'."""

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
    address:str
    phone:str
    website:str
    category:str
    extracted_signals:str
    score:str
    score_reason:str
    next: str

class SupervisorDecision(BaseModel):
    next : Literal["extract_signals","score_directly"]

class ScoreResult(BaseModel):
    score:Literal["hot","warm","cold"]
    reason : str

def get_claude():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def lead_supervisor(state:LeadState)-> dict:
    llm=get_claude()
    structured=llm.with_structured_output(SupervisorDecision)

    decision=structured.invoke([
        SystemMessage(content=(
            "You supervise lead scoring. Given a business's raw data, decide: "
            "'score_directly' if the signals are already clear (e.g. has a "
            "website and looks like a big chain, or clearly has no website "
            "and looks independent). 'extract_signals' if it's ambiguous and "
            "needs a closer read of the available fields first."
        )),
        HumanMessage(content=(
            f"Business: {state['business_name']}\n"
            f"Category: {state['category']}\n"
            f"Website: {state['website'] or 'none'}\n"
            f"Address: {state['address']}"

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
            "likely price point. Be specific, 1-2 sentences, based only on "
            "what's given — don't invent facts not implied by the data."
        )),
        HumanMessage(content=(
            f"Business: {state['business_name']}\n"
            f"Category: {state['category']}\n"
            f"Website: {state['website'] or 'none'}\n"
            f"Address: {state['address']}"
        )),
    ])
    return {"extracted_signals": response.content}

def scorer(state: LeadState) -> dict:
    llm = get_claude()
    structured = llm.with_structured_output(ScoreResult)

    context = f"Category: {state['category']}\nWebsite: {state['website'] or 'none'}"
    if state.get("extracted_signals"):
        context += f"\nAdditional signal: {state['extracted_signals']}"

    result = structured.invoke([
        SystemMessage(content=(
            "Score this lead hot/warm/cold for sales outreach. Hot = strong "
            "opportunity (e.g. no website but clearly active business). "
            "Warm = some opportunity but less urgent. Cold = already well "
            "established online, low urgency. Give a short concrete reason."
        )),
        HumanMessage(content=f"Business: {state['business_name']}\n{context}"),
    ])
    return {"score": result.score, "score_reason": result.reason}

def route_from_supervisor(state:LeadState)->str:
    return state["next"]

def build_lead_processor():
    graph=StateGraph(LeadState)
    graph.add_node("supervisor", lead_supervisor)
    graph.add_node("extract_signals", signal_extractor)
    graph.add_node("scorer", scorer)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"extract_signals":"extract_signals","score_directly":"scorer"}
    )
    graph.add_edge("extract_signals","scorer")
    graph.add_edge("scorer",END)

    return graph.compile()

_lead_processor=build_lead_processor()

class GuardrailResult(BaseModel):
    passes:bool
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




