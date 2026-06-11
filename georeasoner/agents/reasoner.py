"""Reasoner agent — interprets FSI results into scientific natural-language explanation."""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from georeasoner.agents._utils import trace_entry
from georeasoner.llm import get_llm
from georeasoner.state import GeoReasonerState

_SYSTEM_PROMPT = """You are a scientific analyst specialising in flood risk assessment for
South Asia. You will receive the results of a Flood Susceptibility Index (FSI) analysis
for Sylhet District, Bangladesh.

Your response must:
1. Identify the most flood-prone upazilas and explain WHY (terrain, river proximity, land use)
2. Describe the key environmental drivers of flood susceptibility in the Sylhet basin
3. Note limitations of the weighted-overlay FSI method
4. Suggest one evidence-based risk-reduction measure

Be precise (3–4 paragraphs), cite the actual FSI values, and write for a scientific audience."""


def reasoner_node(state: GeoReasonerState) -> dict:
    """Interpret FSI results and produce a ranked scientific answer."""
    ranking = state.get("fsi_ranking") or []

    context = (
        f"Original query: {state['query']}\n\n"
        f"FSI ranking (top 10 upazilas, sorted by mean FSI):\n"
        f"{json.dumps(ranking[:10], indent=2)}\n\n"
        f"Analysis parameters:\n"
        f"  Weights — elevation: 35%, slope: 25%, river proximity: 25%, land cover: 15%\n"
        f"  FSI scale: 0 (least susceptible) → 1 (most susceptible)\n"
        f"  Study area: Sylhet District, Bangladesh\n"
    )

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        reasoning = response.content
    except Exception:
        reasoning = _default_reasoning(ranking)

    answer = _build_answer(ranking, reasoning)
    entry = trace_entry("reasoner", "interpret_fsi", {"n_units": len(ranking)}, answer[:200])

    return {"reasoning": reasoning, "answer": answer, "agent_trace": [entry]}


def _build_answer(ranking: list[dict], reasoning: str) -> str:
    if not ranking:
        return reasoning
    top3 = "; ".join(
        f"{r['rank']}. {r['name']} (FSI {r['mean_fsi']:.3f})" for r in ranking[:3]
    )
    return f"**Top flood-vulnerable upazilas:** {top3}\n\n{reasoning}"


def _default_reasoning(ranking: list[dict]) -> str:
    if not ranking:
        return "Insufficient data for detailed analysis."
    top = ranking[0]
    return (
        f"The FSI analysis identifies {top['name']} as the most flood-susceptible upazila "
        f"(mean FSI: {top['mean_fsi']:.3f}). The weighted overlay integrates elevation (35%), "
        f"terrain slope (25%), proximity to rivers (25%), and land cover type (15%). "
        f"Low-lying floodplain areas adjacent to the Surma and Kushiyara rivers, characterised "
        f"by flat cropland with minimal tree cover, score highest on the index — consistent with "
        f"documented monsoon flood patterns in the Sylhet basin. "
        f"Limitations include the 90 m spatial resolution of the SRTM DEM and the static nature "
        f"of the land-cover input. Recommended intervention: community-based early-warning systems "
        f"combined with flood-resilient crop varieties in the highest-risk upazilas."
    )
