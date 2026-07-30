# Pragmatic Multi-Stage AI Content Engineering Platform

An autonomous, self-correcting 11-stage multi-agent pipeline built inside a single, unified `main.py` file powered by **LangGraph**, **Google Gemini**, **Crawl4AI**, **LlamaIndex**, **spaCy**, **TextStat**, **DeepEval**, **RAGAS**, **TruLens**, and **Prometheus**.

---

## 1. Overview

The platform automates technical trend discovery, engineer audience blueprinting, deep web research, semantic knowledge indexing, insight synthesis, content brief creation, post generation, writing polishing, multi-framework evaluation, and telemetry analytics. All data flow between stages is strictly enforced using Pydantic models.

---

## 2. Architecture & Pipeline

```
  [ STAGE 1: Trending Topics ] ──► (TrendingTopic)
               │
               ▼
  [ STAGE 2: Audience Planning ] ──► (AudienceBlueprint)
               │
               ▼
  [ STAGE 3: Research Planning ] ──► (ResearchPlan)
               │
               ▼
  [ STAGE 4: Adaptive Research ] ──► (ResearchArtifact)  [Crawl4AI / DeepSearcher / OpenManus]
               │
               ▼
  [ STAGE 5: Knowledge Indexing ] ──► (KnowledgeDocument) [spaCy NER + TextStat Readability + LlamaIndex]
               │
               ▼
  [ STAGE 6: Insight Extraction ] ──► (InsightGraph)
               │
               ▼
  [ STAGE 7: Content Brief ] ──► (ContentBrief)
               │
               ▼
  [ STAGE 8: AI Writer ] ──► (GeneratedContent) [Gemini 2.5 Flash Structured Output]
               │
               ▼
  [ STAGE 9: Writing Polish ] ──► (GeneratedContent) [Anti-AI Buzzword Filtering]
               │
               ▼
  [ STAGE 10: Evaluation ] ──► (EvaluationReport) [DeepEval GEval + Ragas + TruLens]
               │
               ├──► [ Avg Humanness < 85% & Attempts < 3 ] ──► (Loop back to STAGE 8)
               │
               ▼
  [ STAGE 11: Analytics ] ──► (AnalyticsRecord) [Prometheus Telemetry & Report File Saver]
```

---

## 3. Installation & Setup

### Environment Configuration
Create a `.env` file in the repository root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. How to Run

Execute the main pipeline:
```bash
python main.py
```

The system will execute all 11 stages asynchronously and save the final report to a timestamped file (`posts_YYYYMMDD_HHMMSS.txt`).

---

## 5. External Libraries

- **LangGraph**: Stateful 11-stage multi-agent orchestration runtime with retry and cyclic edge routing.
- **Crawl4AI**: Asynchronous headless web crawler generating fit-markdown.
- **LlamaIndex**: Dense vector store indexing and semantic retrieval.
- **spaCy**: Industrial-strength Named Entity Recognition (`en_core_web_sm`).
- **TextStat**: Syntactic readability indices (Flesch Reading Ease, Flesch-Kincaid Grade, Reading Time).
- **DeepEval**: Custom G-Eval metrics for qualitative tracks (Humanness, Hook Quality, Technical Depth, Engagement).
- **RAGAS & TruLens**: Faithfulness and Answer Relevancy cross-referencing against context.
- **Prometheus**: Gauge and Counter performance telemetry metrics.
- **DeepSearcher**: Deep search query expansion and multi-pass retrieval.
- **OpenManus**: Agentic tool orchestration loop wrapper.

---

## 6. Future Improvements

- Support for automated direct posting to LinkedIn and X APIs.
- Persistent pgvector PostgreSQL database backend integration.
- Distributed worker queues via Celery or Redis for concurrent document ingestion.

---

## Implemented Architecture

### New Classes
- `SystemConfig`: Global system configuration parameters.
- `Crawl4AIService`: Service wrapper executing headless web scraping via Crawl4AI.
- `DeepSearcherService`: Service wrapper executing multi-pass deep query expansion.
- `OpenManusAgentService`: Service wrapper running autonomous multi-tool agentic loops.
- `LinguisticAnalysisService`: Service wrapper running spaCy NER and tokenization.
- `ReadabilityService`: Service wrapper calculating TextStat syntactic readability metrics.
- `SemanticIndexService`: Service wrapper managing LlamaIndex vector indexing and document retrieval.
- `MultiFrameworkEvaluatorService`: Service wrapper running DeepEval GEval metrics, RAGAS Faithfulness, and TruLens tracing.
- `PrometheusTelemetryService`: Service wrapper recording Prometheus Counter and Gauge metrics.

### New Pydantic Models
- `TrendingTopic`: Discovery trend data payload.
- `AudienceBlueprint`: Engineer persona, pain points, and value proposition payload.
- `ResearchPlan`: Multi-query and target URI plan payload.
- `ResearchArtifact`: Ingested page markdown and scrape performance payload.
- `KnowledgeDocument`: Enriched document with spaCy NER and TextStat metrics payload.
- `InsightGraph`: Synthesized technical trade-offs and code snippet payload.
- `ContentBrief`: Post generation outline and technical depth requirements payload.
- `SocialPostMetadata`: Detailed 6-metric qualitative and groundedness post metadata.
- `StructuredPost`: Individual post payload targeting LinkedIn or X.
- `SocialPostBatch` / `GeneratedContent`: Batch container of 10 structured posts.
- `EvaluationReport`: Step-by-step multi-framework evaluation report payload.
- `AnalyticsRecord`: System aggregate analytics and telemetry record.

### New Workflow
- 11-stage LangGraph workflow (`build_content_pipeline_graph`) containing `trend_discovery_node`, `audience_planning_node`, `research_planning_node`, `adaptive_research_node`, `knowledge_indexing_node`, `insight_extraction_node`, `content_brief_node`, `ai_writer_node`, `writing_polish_node`, `evaluation_node`, and `analytics_node` connected via the `route_evaluation` cyclic edge gate.

### New Integrations
- **LangGraph**: Core 11-node cyclic state graph engine.
- **Crawl4AI**: Headless web crawling and fit-markdown filtering.
- **LlamaIndex**: Vector store document indexing and semantic retrieval.
- **spaCy**: Named Entity Recognition (`en_core_web_sm`).
- **TextStat**: Flesch Reading Ease and Kincaid Grade level evaluation.
- **DeepEval & RAGAS & TruLens**: Multi-metric qualitative and groundedness assessment.
- **Prometheus**: Gauge and Counter real-time metrics telemetry.
- **DeepSearcher & OpenManus**: Multi-pass query expansion and tool orchestration.
