"""Optional Tavily company research."""

from __future__ import annotations

from interview_agent.graph.state import InterviewPrepState
from interview_agent.keys import tavily_api_key
from interview_agent.progress import log_node


def node_research_company(state: InterviewPrepState) -> dict:
    company = (state.get("company_name") or "").strip()
    if not company:
        log_node("company_research", "No company provided; skipping Tavily search.")
        return {"company_context": None}

    log_node("company_research", f"Searching Tavily for company context: {company}")
    try:
        from tavily import TavilyClient
    except ImportError:
        return {
            "errors": ["Install tavily-python for company research."],
            "company_context": None,
        }

    try:
        tv = TavilyClient(api_key=tavily_api_key())
        query = (
            f"{company}: what they do, main products/services, recent news, "
            f"culture/values if known, and hiring/interview-relevant public signals."
        )
        raw = tv.search(query=query, search_depth="advanced", max_results=8)
        chunks: list[str] = []
        for item in raw.get("results") or []:
            title = (item.get("title") or "").strip()
            body = (item.get("content") or item.get("snippet") or "").strip()
            url = (item.get("url") or "").strip()
            if body:
                ref = f"{title}\n{body}"
                if url:
                    ref += f"\nSource: {url}"
                chunks.append(ref.strip())
        text = "\n\n---\n\n".join(chunks).strip()
        if not text:
            log_node("company_research", "Tavily returned no usable company snippets.")
            return {
                "company_context": f"(No Tavily results returned for “{company}”.)",
            }
        log_node("company_research", f"Collected {len(chunks)} Tavily snippets.")
        return {"company_context": text}
    except Exception as exc:
        log_node("company_research", f"Failed: {exc}")
        return {
            "company_context": None,
            "errors": [f"Company research failed: {exc}"],
        }
