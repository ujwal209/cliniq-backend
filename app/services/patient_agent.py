import re
from typing import TypedDict, List
from datetime import datetime
import zoneinfo
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from app.services.ai_doctor import _serper_search, get_rotated_groq_keys

# ── Tool: ONLY web_search ──────────────────────────────────────────────────────

class SearchWebInput(BaseModel):
    query: str = Field(..., description="Clinical search query for guidelines, drug interactions, or research.")

@tool(args_schema=SearchWebInput)
def search_web(query: str) -> str:
    """Search the web for up-to-date clinical guidelines, drug interactions, and evidence-based research."""
    results = _serper_search(query)
    out = ""
    for r in results:
        out += f"Title: {r['title']}\nSnippet: {r['snippet']}\nSource: {r['url']}\n\n"
    return out or "No relevant clinical search results found."

PATIENT_TOOLS = [search_web]

# ── Cleanup: strip any leaked function-call syntax from LLM output ─────────────

def _strip_function_calls(text: str) -> str:
    """Remove any raw <function=...>...</function> or <function=.../> leaked tags."""
    # Remove <function=name {...}>...</function>
    text = re.sub(r'<\s*function\s*=\s*\w+[^>]*>.*?(?:</function>|(?=\n\n))', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove self-closing <function=name {...}/>
    text = re.sub(r'<\s*function\s*=\s*\w+[^>]*/?\s*>', '', text, flags=re.IGNORECASE)
    # Remove any markdown image wrapping pollinations/unsplash hallucinations
    text = re.sub(r'!\[.*?\]\(https?://image\.pollinations\.ai[^\)]*\)', '', text)
    text = re.sub(r'!\[.*?\]\(https?://[a-z]+\.unsplash\.com[^\)]*\)', '', text)
    # Clean up extra blank lines left behind
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── System Prompt ──────────────────────────────────────────────────────────────

PATIENT_SYSTEM_PROMPT = """You are Dr. HealthSync, a highly experienced AI physician specializing in internal medicine, clinical diagnosis, and evidence-based treatment.

Your primary role is to ASSESS THE PATIENT through a structured clinical consultation. Always start by understanding their symptoms fully before giving any diagnosis or treatment.

═══════════════════════════════════════════════════════════
  CONSULTATION FLOW — ALWAYS FOLLOW THIS ORDER:
═══════════════════════════════════════════════════════════
1. SYMPTOM INTAKE: If the patient hasn't described symptoms yet, ask targeted questions:
   - Chief complaint (what's wrong?)
   - Duration (how long?)
   - Severity (scale 1-10)
   - Associated symptoms (fever, pain, nausea, etc.)
   - Medical history, current medications, allergies
   - Only move to assessment AFTER you have enough information.

2. CLINICAL ASSESSMENT: Once symptoms are clear:
   - State your working diagnosis and differentials clearly
   - Use the format: ### Assessment, ### Differential Diagnosis

3. TREATMENT PLAN:
   - Name SPECIFIC medications with exact dose/frequency/duration (e.g., "Ibuprofen 400mg orally every 8 hours with food")
   - Include: drug class, mechanism, contraindications, expected effect
   - Use format: ### Prescription

4. INVESTIGATIONS: Recommend specific tests if needed (CBC, CRP, ECG, etc.)
   - Use format: ### Investigations

5. RED FLAGS: Always state symptoms that require immediate emergency care.
   - Use format: ### 🚨 Red Flags

═══════════════════════════════════════════════════════════
  ABSOLUTE PROHIBITIONS:
═══════════════════════════════════════════════════════════
- NEVER output any <function=...> tags or tool syntax in your text — this will corrupt the response.
- NEVER generate or include image URLs (pollinations, unsplash, etc.) in your response.
- NEVER say "here is an image of" or attempt to show visual content.
- NEVER give vague advice like "eat healthy" or "see a doctor" without specifics.
- ONLY use the search_web tool for looking up clinical data — not for images or illustrations.

You speak with calm clinical authority. Every response should feel like a consultation with a brilliant, caring physician."""

# ── Graph State ────────────────────────────────────────────────────────────────

class PatientGraphState(TypedDict):
    messages: List
    patient_email: str
    patient_context: str
    web_search_enabled: bool
    session_id: str

def patient_node(state: PatientGraphState):
    messages = state["messages"]
    patient_context = state.get("patient_context", "")

    pt_ctx = f"\n\n[PATIENT PROFILE — FACTOR INTO ALL RESPONSES]\n{patient_context}" if patient_context else ""
    system_content = PATIENT_SYSTEM_PROMPT + pt_ctx

    try:
        ist = zoneinfo.ZoneInfo("Asia/Kolkata")
        now_local = datetime.now(ist)
        datetime_context = f"Current date and time: {now_local.strftime('%A, %d %B %Y at %I:%M %p IST')}"
    except Exception:
        datetime_context = f"Current date and time (UTC): {datetime.utcnow().strftime('%A, %d %B %Y at %I:%M %p')}"

    system_content += f"\n\n[TEMPORAL CONTEXT] {datetime_context}."

    full_context = [SystemMessage(content=system_content)] + messages

    keys = get_rotated_groq_keys()
    models = ["llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"]

    llms = []
    for model in models:
        for key in keys:
            llm = ChatGroq(model=model, groq_api_key=key, temperature=0.1, max_retries=0)
            llm = llm.bind_tools(PATIENT_TOOLS)
            llms.append(llm)

    primary_llm = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]
    response = primary_llm.invoke(full_context)

    new_sources = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_results = []
        tool_map = {t.name: t for t in PATIENT_TOOLS}

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"Tool error: {e}"

            if tool_name == "search_web" and result:
                urls = re.findall(r'Source: (https?://\S+)', result)
                titles = re.findall(r'Title: (.*?)\n', result)
                for title, url in zip(titles, urls):
                    new_sources.append({"title": title.strip(), "url": url.strip()})

            tool_results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        followup = [SystemMessage(content=system_content)] + messages + [response] + tool_results

        bare_llms = []
        for model in models:
            for key in keys:
                bare_llms.append(ChatGroq(model=model, groq_api_key=key, temperature=0.1, max_retries=0))

        final_llm = bare_llms[0].with_fallbacks(bare_llms[1:]) if len(bare_llms) > 1 else bare_llms[0]
        final_response = final_llm.invoke(followup)

        # Strip any leaked function-call syntax before returning
        clean_content = _strip_function_calls(final_response.content)

        return {
            "messages": [response] + tool_results + [AIMessage(content=clean_content)],
            "sources": new_sources
        }

    # Strip leaked function-call syntax from direct response too
    clean_content = _strip_function_calls(response.content)
    return {"messages": [AIMessage(content=clean_content)], "sources": []}


workflow = StateGraph(PatientGraphState)
workflow.add_node("patient", patient_node)
workflow.set_entry_point("patient")
workflow.add_edge("patient", END)
patient_agent_app = workflow.compile()
