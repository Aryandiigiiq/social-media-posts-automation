# Pragmatic Multi-Stage AI Content Engineering Platform (Decoupled X & LinkedIn Pipelines)

An autonomous, self-correcting multi-agent pipeline built inside a single, unified `main.py` file powered by **LangGraph**, **Google Gemini**, **Crawl4AI**, **LlamaIndex**, **spaCy**, **TextStat**, **DeepEval**, **RAGAS**, **TruLens**, and **Prometheus**.

---

## 1. Overview

The platform automates technical trend discovery, engineer audience blueprinting, deep web research, semantic knowledge indexing, insight synthesis, content brief creation, platform strategy formulation, post generation, writing polishing, multi-framework evaluation, and telemetry analytics.

The system features **decoupled generation and evaluation pipelines for LinkedIn and X (Twitter)**:
- **X Pathway**: 100% untouched production-ready pipeline generating 5 punchy, high-signal, code-and-parameter-dense technical threads.
- **LinkedIn Pathway**: Dedicated story-driven pipeline generating 5 first-person business-impact posts tailored for Engineering Managers, CTOs, Staff Engineers, and Product Leaders.

---

## 2. Architecture & Decoupled Workflow

```
                                  [STAGE 7: Content Brief Builder]
                                                 │
                                                 ▼
                             [STAGE 7.5: LinkedIn Strategy Planning]
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   │                                                           │
                   ▼                                                           ▼
         [STAGE 8X: X Writer]                                   [STAGE 8LI: LinkedIn Writer]
                   │                                                           │
                   ▼                                                           ▼
         [STAGE 9X: X Polish]                                   [STAGE 9LI: LinkedIn Polish]
                   │                                                           │
                   ▼                                                           ▼
        [STAGE 10X: X Evaluation]                              [STAGE 10LI: LinkedIn Evaluation]
        (DeepEval GEval & Groundedness)                        (Readability Gate & Business Impact)
                   │                                                           │
                   ├──► [ Score < 85 & Attempts < 3 ]                          ├──► [ Readability / Score < 85
                   │    (Loop back to STAGE 8X)                                │      & Attempts < 3 ]
                   │                                                           │      (Loop back to STAGE 8LI)
                   │ (Passes Gate)                                             │ (Passes Gate)
                   └─────────────────────────────┬─────────────────────────────┘
                                                 ▼
                                     [STAGE 11: Analytics & Saver]
```

---

## 3. LinkedIn vs X Strategy Comparison

| Strategy Dimension | X (Twitter) Pathway (Untouched) | LinkedIn Pathway (Dedicated) |
| :--- | :--- | :--- |
| **Target Audience** | Infrastructure, AI Systems Engineers | Senior Engineers, EMs, CTOs, Staff Engineers, VPs, Product Leaders |
| **Communication Goal** | High-throughput technical depth & parameters | Business value, engineering leadership, storytelling, practical lessons |
| **Narrative Structure** | Problem $\rightarrow$ Code $\rightarrow$ Trade-offs | Why care? $\rightarrow$ Problem $\rightarrow$ What changed? $\rightarrow$ What was learned? $\rightarrow$ Why it matters $\rightarrow$ Open discussion CTA |
| **Writing Tone** | Direct, implementation-first, parameter-dense | First-person ("I noticed...", "We tried..."), high sentence burstiness, conversational |
| **Markdown Rules** | Standard code blocks & parameters | Natural paragraphs (No bullet dumps `• • •`) |
| **Evaluation Metrics** | GEval Humanness, Technical Depth, Faithfulness | TextStat Flesch Reading Ease ($\ge 55$), Kincaid Grade ($\le 10$), Business Impact Gate |

---

## 4. How to Run

Execute the main pipeline:
```bash
python main.py
```

The system will execute the parallel pipeline asynchronously and save 10 posts (5 LinkedIn, 5 X) to a timestamped file (`posts_YYYYMMDD_HHMMSS.txt`).

---

## 5. Implemented Architecture

### New Classes & Models
- `PlatformContentStrategy`: Pydantic strategy model defining platform-specific audience targets, communication goals, narrative styles, hook preferences, and readability targets.
- `LinkedInPostMetadata`, `LinkedInStructuredPost`, `LinkedInPostBatch`: Isolated Pydantic schemas capturing business value focus, story narrative types, TextStat readability scores, and grade difficulty levels.
- `LinkedInEvaluatorService`: Dedicated evaluator enforcing hard readability limits (Flesch Ease $\ge 55$, Grade $\le 10$), business-impact-over-code audits, and actionable rewrite feedback.

### New Workflow Nodes
- `linkedin_strategy_node`: Formulates `PlatformContentStrategy` for LinkedIn.
- `linkedin_ai_writer_node`: Prompts Gemini for 5 story-driven, first-person LinkedIn posts optimized for business value.
- `linkedin_writing_polish_node`: Applies anti-AI filters and verifies natural conversational flow.
- `linkedin_evaluation_node`: Executes `LinkedInEvaluatorService` checks.
- `route_evaluation_linkedin`: Independent cyclic router gate for LinkedIn retries.
