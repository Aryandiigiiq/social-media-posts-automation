# DIGIiq Solution Hybrid AI Content Engineering Platform

An autonomous, self-correcting multi-agent content generation platform built inside `main.py`, engineered for **DIGIiq Solution Private Limited** ([LinkedIn Profile](https://www.linkedin.com/company/digiiq-solution-private-limited/posts/?feedView=all)).

The platform features an 11-stage stateful workflow powered by **LangGraph**, **LangChain**, **Google Gemini**, **LlamaIndex**, **Crawl4AI**, **spaCy**, **TextStat**, **CrewAI**, **DeepEval**, **RAGAS**, **TruLens**, and **Prometheus Telemetry**.

---

## 1. Integrated Technologies & Official Documentation Links

| Technology | Role in System | Documentation Link |
| :--- | :--- | :--- |
| **LangGraph** | Stateful DAG workflow orchestration across 11 stages | [LangGraph Docs](https://python.langchain.com/docs/langgraph/) \| [LangGraph GitHub](https://langchain-ai.github.io/langgraph/) |
| **LangChain Core** | LLM output structuring (`with_structured_output`), prompt engineering | [LangChain Docs](https://python.langchain.com/docs/) |
| **LangChain Google GenAI** | Gemini API integration wrapper (`ChatGoogleGenerativeAI`) | [LangChain Google GenAI Docs](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) |
| **Google Gemini API** | Core LLM engine (`gemini-2.5-flash`) for research, generation & evaluation | [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs) \| [Google GenAI Python SDK](https://github.com/google-gemini/generative-ai-python) |
| **LlamaIndex Core** | RAG vector store indexing and semantic retrieval over post corpora | [LlamaIndex Docs](https://docs.llamaindex.ai/en/stable/) |
| **Crawl4AI** | Headless async web crawler (`AsyncWebCrawler`) for live research extraction | [Crawl4AI Documentation](https://docs.crawl4ai.com/) |
| **DeepSearcher** | Multi-source web research orchestration over AI & automation domains | [DeepSearcher GitHub](https://github.com/zilliztech/deepsearcher) |
| **OpenManus** | Autonomous tool loop execution for developer infrastructure synthesis | [OpenManus GitHub](https://github.com/mannaandpoem/OpenManus) |
| **spaCy NLP** | Named Entity Recognition (NER) analysis (`en_core_web_sm`) | [spaCy Documentation](https://spacy.io/usage) |
| **TextStat** | Readability scoring (Flesch Reading Ease & Flesch-Kincaid Grade) | [TextStat PyPI](https://pypi.org/project/textstat/) |
| **CrewAI** | Multi-agent persona review simulation (Executive, Architect, Growth) | [CrewAI Documentation](https://docs.crewai.com/) |
| **DeepEval** | Metric testing for humanness, groundedness, and technical depth | [DeepEval Docs](https://docs.confident-ai.com/) |
| **RAGAS** | RAG evaluation framework (context precision, recall, faithfulness) | [RAGAS Documentation](https://docs.ragas.io/) |
| **TruLens** | Evaluation triad tracking and LLM feedback functions | [TruLens Documentation](https://www.trulens.org/) |
| **Prometheus Client** | Prometheus metric exposition port `8000` & real-time telemetry | [Prometheus Python Client](https://prometheus.github.io/client_python/) |
| **Pydantic v2** | Data validation, strict type enforcement & JSON schemas | [Pydantic Docs](https://docs.pydantic.dev/) |
| **Python Dotenv** | Environment variable configuration (`GEMINI_API_KEY`, `GOOGLE_API_KEY`) | [Python-Dotenv Docs](https://saurabh-kumar.com/python-dotenv/) |
| **DuckDuckGo Search** | Multi-tiered fallback web search engine | [duckduckgo-search PyPI](https://pypi.org/project/duckduckgo-search/) |

---

## 2. Platform Architecture & 11-Stage LangGraph Workflow

```
                                  +-----------------------+
                                  | STAGE 1: Hybrid Topic |
                                  |   Discovery Engine    |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  STAGE 2: Dynamic     |
                                  |   Audience Planning   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  STAGE 3: Research    |
                                  |        Planning       |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | STAGE 4: Adaptive     |
                                  | Crawl4AI/DeepSearcher |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  STAGE 5: spaCy &     |
                                  | TextStat Indexing     |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | STAGE 5.5: LlamaIndex |
                                  | Reference Analysis    |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | STAGE 6: Insight      |
                                  |  Extraction Graph     |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | STAGE 7: Content      |
                                  |     Brief Builder     |
                                  +-----------+-----------+
                                              |
                                              v
                 +----------------------------+----------------------------+
                 |                                                         |
                 v                                                         v
   +---------------------------+                             +---------------------------+
   |   STAGE 8LI: LinkedIn     |                             |  STAGE 8X: X / Twitter    |
   | DIGIiq + Aidan Writer     |                             |       Writer Engine       |
   +-------------+-------------+                             +-------------+-------------+
                 |                                                         |
                 v                                                         v
   +---------------------------+                             +---------------------------+
   | STAGE 9LI: Aidan Style    |                             |   STAGE 9X: X Polish      |
   |    Paragraph Polish       |                             |         Engine            |
   +-------------+-------------+                             +-------------+-------------+
                 |                                                         |
                 v                                                         v
   +---------------------------+                             +---------------------------+
   | STAGE 10LI: CrewAI &      |                             |   STAGE 10X: Multi-       |
   | Anti-AI Gate Evaluator    |                             |   Framework Evaluator     |
   +-------------+-------------+                             +-------------+-------------+
                 |                                                         |
                 +----------------------------+----------------------------+
                                              |
                                              v
                                  +-----------------------+
                                  | STAGE 11: Analytics & |
                                  | Prometheus Saver Node |
                                  +-----------------------+
```

---

## 3. Core Generation Rules & Brand Mandates

1. **DIGIiq Solution Account Perspective**:
   - Every post is written from the authentic DIGIiq team perspective (`"Here at DIGIiq, we..."`, `"At DIGIiq, we constantly try to be better at..."`, `"This was our learning at DIGIiq..."`, `"We implemented this directly into DIGIiq core..."`).
2. **Hybrid Topic Blending**:
   - Intersects DIGIiq core pillars (`#aichatbot`, `#businessautomation`, `#customerexperience`, `#promptengineering`, `#digitaltransformation`) with real-time trending AI concepts (agentic streaming memory, zero-defect prompt guardrails, production RAG, sub-300ms voice AI).
3. **Dynamic Research-Driven Hook Generation**:
   - Opening hooks (Line 1, strictly $<10$ words) are dynamically synthesized from research findings, industry case studies, and topic friction points.
4. **Signature Founder-Led Style**:
   - Line 1 standalone hook -> contrarian twist -> concrete metrics ($18,000 saved, 40% retention jump, 30 hours weekly saved) -> technical teardown -> reflective networking CTA.
5. **100% Dynamic Direct Evaluator Scoring**:
   - Humanness, accessibility, and overall effectiveness scores are generated dynamically by n-gram entropy, Wikipedia AI sign audits, and CrewAI persona evaluators without hardcoded fallbacks.

---

## 4. Setup & Execution

### Prerequisites
- Python 3.10+
- Google Gemini API Key

### Environment Setup
Create a `.env` file in the project root:
```env
GEMINI_API_KEY="your_google_gemini_api_key_here"
```

### Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Execution
```bash
python main.py
```

Output posts will be automatically saved to a timestamped file `posts_YYYYMMDD_HHMMSS.txt` in the root directory.
