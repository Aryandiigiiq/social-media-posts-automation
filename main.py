import os, re
from datetime import datetime
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

load_dotenv()

# ==============================================================================
# INJECTED TOOL DOCUMENTATION, DATA & CODE SNIPPETS
# ==============================================================================
INJECTED_CONTEXT = """
### 1. Crawl4AI Technical Context
```python
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation import DefaultMarkdownGenerator

async def execute_crawl():
    # Configure browser properties (headless mode, user simulation)
    browser_cfg = BrowserConfig(headless=True)
    
    # Configure crawl-specific settings (bypass caching for fresh content)
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=DefaultMarkdownGenerator()
    )
    
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # Execute crawl asynchronously
        result = await crawler.arun(
            url="https://example.com",
            config=run_cfg
        )
        
        # Output programmatic findings (raw and fit markdown structures)
        print("Raw Markdown:", result.markdown[:100])
        if hasattr(result.markdown, 'fit_markdown'):
            print("Filtered Markdown:", result.markdown.fit_markdown[:100])

if __name__ == "__main__":
    asyncio.run(execute_crawl())
```
* Key Functions & API Call Hierarchy: AsyncWebCrawler (main context manager), BrowserConfig (browser action control), CrawlerRunConfig (runtime mechanics), arun_many(urls, ...) (concurrency engine with MemoryAdaptiveDispatcher).
* Key Parameters & Settings: CacheMode.BYPASS / CacheMode.ENABLED, stream=True (async generation stream), session_id, js_code, wait_for, js_only, kill_session().
* Data Types: Returns CrawlResult containing result.markdown and result.markdown.fit_markdown.
* Operational Trade-offs: PruningContentFilter with DefaultMarkdownGenerator improves text signal-to-noise ratio but adds ~50ms processing latency. LLM-free extraction (CSS/XPath selectors) has zero token cost vs LLM-based parsing API costs.
* Production Pipeline Use Case: Asynchronous web ingestion converting dynamic and static structures into clean LLM-readable Markdown.

### 2. OpenWebTrack Technical Context
```python
import requests

def query_openwebtrack_api():
    # Exposing raw metric aggregates via OpenWebTrack secure REST API
    api_url = "http://localhost:8424/api/analytics"  # Core container mapping
    headers = {
        "Authorization": "Bearer YOUR_SECURE_API_KEY",  # REST secure programmatic access
        "Content-Type": "application/json"
    }
    params = {
        "website_id": "main-app-site",
        "period": "7d",
        "metrics": "pageviews,visitors,revenue"  # Key analytical tracks
    }
    
    response = requests.get(api_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return None

if __name__ == "__main__":
    data = query_openwebtrack_api()
    print("Analytics Data Extracted:", data)
```
* Setup, Environment, & API Parameters: SvelteKit framework with local Docker postgres:17-alpine storage. Core env vars: DATABASE_URL, ORIGIN, AUTH_SECRET, DISABLE_REGISTER, CRON_SECRET, and email configurations (RESEND_API_KEY, Maileroo, SMTP).
* Exposed Metrics & Intelligence: Real-time traffic aggregates, entry/exit landing pages, visitor retention profiles (first/last seen), transactional e-commerce revenue by OS/geo, custom conversion funnels.
* Programmatic Interfaces: Secure REST API endpoints and Model Context Protocol (MCP) Server for direct AI assistant integration.
* Operational Trade-offs: Complete data ownership with zero third-party leakage vs engineering overhead of hosting, database replication, and SMTP maintenance.
* Production Pipeline Use Case: Self-hosted analytics stack for real-time application behavior monitoring and autonomous MCP AI agent querying.

### 3. Plausible Analytics Technical Context
```python
import requests

def fetch_plausible_breakdown():
    api_url = "https://plausible.io/api/v1/stats/breakdown"
    headers = {
        "Authorization": "Bearer YOUR_PLAUSIBLE_TOKEN",
        "Content-Type": "application/json"
    }
    
    # Query structure containing mandatory fields, dimensions, and filter rules
    payload = {
        "site_id": "dummy.site",
        "metrics": ["visitors", "pageviews", "bounce_rate"],
        "date_range": "7d",
        "dimensions": ["visit:country_name", "visit:city_name"],
        "filters": [
            ["is_not", "visit:country_name", [""]],  # Simple filter array
            ["contains", "event:page", ["blog"], {"case_sensitive": False}]  # Case-insensitive modifier
        ],
        "pagination": {
            "limit": 100,
            "offset": 0
        }
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        results = response.json().get("results", [])
        for row in results:
            print(f"Dimensions: {row['dimensions']} | Metrics: {row['metrics']}")

if __name__ == "__main__":
    fetch_plausible_breakdown()
```
* Required Query Parameters: site_id (domain string), date_range (custom ISO8601 array or intervals like "24h", "7d", "30d", "year").
* Analytical Dimensions: Event Dimensions (event:goal, event:page, event:hostname), Visit Dimensions (referrer, acquisition channel, UTM params, device, location), Time & Custom props.
* Filter Logic Syntax: Arrays [operator, dimension, clauses] with operators (is, is_not, contains, matches regex), supporting trailing modifiers like {"case_sensitive": false}.
* Operational Trade-offs: Cannot mix session-level metrics (bounce_rate, visit_duration) with event-level dimensions except event:page and event:hostname.
* Production Pipeline Use Case: Programmatic querying for automated marketing reports, campaign UTM tracking, and BI dashboard ingestion.

### 4. Python Feedparser Technical Context
```python
import feedparser

def parse_syndication_feed():
    feed_url = "https://feedparser.readthedocs.io/en/latest/"  # Documentation target
    
    # Programmatic parse execution
    parsed_result = feedparser.parse(feed_url)
    
    # Evaluate parsed structures and errors
    if parsed_result.bozo:
        print("Malformed feed detected. Exception:", parsed_result.bozo_exception)
    
    # Inspect metadata structures
    print(f"Feed Version: {parsed_result.version}")
    print(f"Feed Title: {parsed_result.feed.title}")
    
    # Ingest published feed entries
    for index, entry in enumerate(parsed_result.entries[:5]):
        print(f"\\nEntry {index + 1}: {entry.title}")
        print(f"Link: {entry.link}")
        print(f"Published Date: {entry.published}")

if __name__ == "__main__":
    parse_syndication_feed()
```
* Key Functionalities: feedparser.parse() accepts HTTP URLs, local files, or raw streams. Handles HTTP redirects, ETag/Last-Modified conditional GETs, HTML sanitization, and namespace resolution.
* Key Schemas: feed object (title, link, generator, language), entries list (title, link, summary, published_parsed, content), bozo flag and bozo_exception.
* Operational Trade-offs: Multi-format parsing (RSS, Atom, JSON) vs version variations; integrations should pin to active standard versions (e.g. 6.0.13).
* Production Pipeline Use Case: First-stage syndication parser polling RSS/Atom feeds, verifying HTTP headers, checking bozo flags, and standardizing ingest data.
"""

# ==============================================================================
# PYDANTIC STRUCTURED OUTPUT SCHEMAS
# ==============================================================================
class PostEvaluationSchema(BaseModel):
    humanness_score: int = Field(
        description="Score from 0-100 measuring natural human phrasing, absence of obvious AI buzzwords, and developer tone.",
        ge=0, le=100
    )
    hook_score: int = Field(
        description="Score from 0-100 grading the hook quality. Fast setups, problem statements, and technical entry are favored over generalities.",
        ge=0, le=100
    )
    technical_depth_score: int = Field(
        description="Score from 0-100 checking for API parameters, explicit configurations, exact metrics, and precise database/dependency naming.",
        ge=0, le=100
    )
    engagement_score: int = Field(
        description="Score from 0-100 checking the post's organic readability, syntax formatting, and community-specific vocabulary.",
        ge=0, le=100
    )
    detailed_critique: str = Field(
        description="Actionable and blunt technical feedback summarizing exactly why points were deducted or added."
    )

class SocialPostMetadata(BaseModel):
    topics_used: List[str] = Field(
        description="List of raw technical topics addressed in the post (e.g., ['Concurrency', 'Caching', 'REST API'])."
    )
    sources_used: List[str] = Field(
        description="Explicit tools or resources referenced from the context (e.g., ['Crawl4AI', 'Plausible Analytics'])."
    )
    hook_type: str = Field(
        description="Categorization of the post opening (e.g., 'Stat-driven', 'System Bottleneck', 'Direct Technical Setup')."
    )
    content_structure: str = Field(
        description="The logical layout of the post structure (e.g., 'Problem-Solution-Tradeoff', 'Standard Tutorial')."
    )
    closure_type: str = Field(
        description="How the post resolves at the end (e.g., 'Actionable CTA', 'Open Technical Discussion', 'Summary Quote')."
    )
    human_score: int = Field(
        default=85,
        description="The anticipated humanness score representing absence of robotic terminology.",
        ge=0, le=100
    )
    overall_effective_score: int = Field(
        default=85,
        description="Combined weight representing hook strength, clarity, and developer value.",
        ge=0, le=100
    )

class StructuredPost(BaseModel):
    platform: str = Field(description="Target social media network. Must be either 'LinkedIn' or 'X'.")
    post_text: str = Field(description="The complete written post body. Technical, pragmatic, and free of filler phrases.")
    metadata: SocialPostMetadata = Field(description="Metadata mapping properties and dimensions of this post.")

class SocialPostBatch(BaseModel):
    posts: List[StructuredPost] = Field(
        description="A list containing exactly 10 generated social posts (5 optimized for LinkedIn, 5 optimized for X)."
    )

# ==============================================================================
# LANGGRAPH STATE SCHEMA
# ==============================================================================
class GraphState(TypedDict):
    research: str
    draft_batch: List[Dict[str, Any]]
    evaluations: List[Dict[str, Any]]
    attempts: int

# Initialize Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

llm_evaluator = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)

# ==============================================================================
# GRAPH NODES
# ==============================================================================
def research_node(state: GraphState) -> Dict[str, Any]:
    """NODE A: Dual-pass search for technical topics and high-engagement post viral structures."""
    search_tool = DuckDuckGoSearchRun()
    
    # Pass 1: Target technical topics
    try:
        topic_results = search_tool.run("latest technical software engineering AI data pipelines robotics systems 2026")
    except Exception as e:
        topic_results = f"Topic search fallback: {e}"
        
    # Pass 2: High-engagement post structures on LinkedIn and X
    try:
        structure_results = search_tool.run("viral technical engineering posts structure hooks LinkedIn Twitter X engagement patterns")
    except Exception as e:
        structure_results = f"Structure search fallback: {e}"

    combined_research = (
        f"=== PASS 1: TECHNICAL TREND RESEARCH ===\n{topic_results}\n\n"
        f"=== PASS 2: VIRAL STRUCTURE & ENGAGEMENT PATTERNS ===\n{structure_results}\n\n"
        f"=== INJECTED TOOL CONTEXT & CODE SNIPPETS ===\n{INJECTED_CONTEXT}"
    )
    return {"research": combined_research, "attempts": 0}

def writer_node(state: GraphState) -> Dict[str, Any]:
    """NODE B: Structured Generation Engine producing exactly 10 posts (5 LinkedIn, 5 X)."""
    attempts = state.get("attempts", 0) + 1
    
    # Extract historical evaluation critiques if rewriting
    critique_context = ""
    if attempts > 1 and state.get("evaluations"):
        critique_context = "\nPREVIOUS RUN CRITIQUES & EVALUATIONS:\n" + "\n".join(
            f"Post {idx + 1} Critique: {ev.get('detailed_critique', 'N/A')}"
            for idx, ev in enumerate(state["evaluations"])
        )

    generator = llm.with_structured_output(SocialPostBatch)
    
    prompt = f"""You are a Senior Infrastructure & AI Systems Engineer writing high-signal technical posts for developers and engineering leaders.

RESEARCH & VIRAL STRUCTURE PLAYBOOK:
{state['research']}
{critique_context}

STRICT GENERATION LAWS:
1. Generate EXACTLY 10 distinct posts inside the SocialPostBatch schema:
   - EXACTLY 5 posts for LinkedIn (long-form, deep system architectures, parameters, trade-offs, technical findings).
   - EXACTLY 5 posts for X / Twitter (short-form or multi-tweet thread format, punchy, high-signal).
2. NO UNNECESSARY CODE BOILERPLATE:
   - Do NOT dump long Python boilerplate. Keep code or parameter references concise and focused on real operational trade-offs.
3. HUMANIZATION & STRICT BANNED WORDS:
   - Strictly BAN robotic AI buzzwords: "delve", "game-changer", "unleash", "tapestry", "testament", "in today's fast-paced world", "furthermore", "moreover", "beacon", "landscape".
   - Write like a battle-tested engineer who has built production systems.
4. METADATA MAPPING:
   - Accurately populate the SocialPostMetadata for each post (topics_used, sources_used, hook_type, content_structure, closure_type).

Generate the batch of 10 structured posts now:"""

    batch_res = generator.invoke(prompt)
    draft_dict_list = [post.model_dump() for post in batch_res.posts]
    return {"draft_batch": draft_dict_list, "attempts": attempts}

def evaluator_node(state: GraphState) -> Dict[str, Any]:
    """NODE C: Multi-metric LangChain Pydantic Evaluator Sub-Graph."""
    evaluator = llm_evaluator.with_structured_output(PostEvaluationSchema)
    evaluations = []
    
    updated_draft_batch = []
    
    for idx, post_dict in enumerate(state.get("draft_batch", [])):
        post_text = post_dict.get("post_text", "")
        
        prompt = f"""You are a strict, senior technical content reviewer auditing dev-focused social media content.

POST CONTENT TO EVALUATE (Post #{idx + 1}):
---
{post_text}
---

Grade the post across the four provided dimensions (0-100 each). Deduct points heavily if you spot generic intros,
marketing slogans, AI fluff words, or vague explanations lacking engineering trade-offs.
"""
        eval_result = evaluator.invoke(prompt)
        eval_dict = eval_result.model_dump()
        evaluations.append(eval_dict)
        
        # Calculate overall effective post score as weighted average
        effective_score = int(
            (eval_result.hook_score * 0.3) + 
            (eval_result.technical_depth_score * 0.4) + 
            (eval_result.engagement_score * 0.3)
        )
        
        # Update metadata scores inside draft batch
        post_copy = dict(post_dict)
        if "metadata" in post_copy and isinstance(post_copy["metadata"], dict):
            post_copy["metadata"]["human_score"] = eval_result.humanness_score
            post_copy["metadata"]["overall_effective_score"] = effective_score
        
        updated_draft_batch.append(post_copy)
        
    return {"evaluations": evaluations, "draft_batch": updated_draft_batch}

def saver_node(state: GraphState) -> Dict[str, Any]:
    """NODE D & E: Formatter & Saver generating timestamped posts_YYYYMMDD_HHMMSS.txt file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"posts_{timestamp}.txt"
    
    eval_list = state.get("evaluations", [])
    draft_batch = state.get("draft_batch", [])
    
    avg_humanness = sum(e.get("humanness_score", 85) for e in eval_list) / len(eval_list) if eval_list else 85
    avg_effective = sum(p.get("metadata", {}).get("overall_effective_score", 85) for p in draft_batch) / len(draft_batch) if draft_batch else 85

    output_lines = [
        "======================================================================",
        "SYSTEM EVALUATION SUMMARY",
        "======================================================================",
        f"Average Humanness Score: {avg_humanness:.1f}%",
        f"Average Overall Effective Post Score: {avg_effective:.1f}%",
        f"Total Generation Attempts: {state.get('attempts', 1)}",
        f"Total Posts Generated: {len(draft_batch)} (5 LinkedIn, 5 X/Twitter)",
        "======================================================================\n"
    ]
    
    for idx, post in enumerate(draft_batch, start=1):
        meta = post.get("metadata", {})
        platform = post.get("platform", "Social").upper()
        post_text = post.get("post_text", "").strip()
        
        topics_str = ", ".join(meta.get("topics_used", [])) or "Technical Systems"
        sources_str = ", ".join(meta.get("sources_used", [])) or "Injected Docs"
        
        analyser_report = f"""======================================================================
POST {idx} OF {len(draft_batch)} [{platform}]
======================================================================
ANALYSER REPORT:
• Human Score: {meta.get('human_score', 85)}%
• Overall Effective Post Score: {meta.get('overall_effective_score', 85)}%
• Topics Used: {topics_str}
• Sources Used: {sources_str}
• Hook Type: {meta.get('hook_type', 'N/A')}
• Content Structure Type: {meta.get('content_structure', 'N/A')}
• Closure Type: {meta.get('closure_type', 'N/A')}
----------------------------------------------------------------------

{post_text}
"""
        output_lines.append(analyser_report)
        
    final_content = "\n".join(output_lines)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_content)
        
    print(f"\n[+] Processing Complete!")
    print(f"[+] Average Humanness Score: {avg_humanness:.1f}% (Attempts: {state.get('attempts', 1)})")
    print(f"[+] Output written to: {filename}")
    return {}

# ==============================================================================
# CONDITIONAL EVALUATION ROUTER
# ==============================================================================
def route_evaluation(state: GraphState) -> str:
    """Cyclic routing logic: route to saver if avg humanness >= 85 or attempts >= 3."""
    attempts = state.get("attempts", 0)
    evals = state.get("evaluations", [])
    
    if attempts >= 3:
        print(f"[->] Max attempts ({attempts}) reached. Routing to Saver Node...")
        return "save"
        
    if evals:
        avg_humanness = sum(e.get("humanness_score", 0) for e in evals) / len(evals)
        if avg_humanness >= 85:
            print(f"[->] Avg Humanness ({avg_humanness:.1f}%) >= 85%. Routing to Saver Node...")
            return "save"
        print(f"[<-] Avg Humanness ({avg_humanness:.1f}%) < 85%. Cycling back to Writer Node for revision...")
        return "rewrite"
        
    return "save"

# ==============================================================================
# WORKFLOW GRAPH ASSEMBLY
# ==============================================================================
workflow = StateGraph(GraphState)

workflow.add_node("research_node", research_node)
workflow.add_node("writer_node", writer_node)
workflow.add_node("evaluator_node", evaluator_node)
workflow.add_node("saver_node", saver_node)

workflow.set_entry_point("research_node")
workflow.add_edge("research_node", "writer_node")
workflow.add_edge("writer_node", "evaluator_node")

workflow.add_conditional_edges(
    "evaluator_node",
    route_evaluation,
    {
        "save": "saver_node",
        "rewrite": "writer_node"
    }
)

workflow.add_edge("saver_node", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({})