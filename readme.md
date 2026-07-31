# Pragmatic Multi-Stage AI Content Engineering Platform (Aidan Nguyen Tran Signature Engine & X Pipeline)

An autonomous, self-correcting multi-agent content generation platform built inside a single, unified `main.py` file powered by **CrewAI Multi-Persona Review**, **LlamaIndex Vector Engine**, **LangGraph**, **Google Gemini**, **Crawl4AI**, **spaCy**, **TextStat**, **DeepEval**, **RAGAS**, **TruLens**, and **Prometheus**.

---

## 1. Overview

The platform features **decoupled, specialized generation and evaluation pipelines for LinkedIn and X (Twitter)**:
- **X Pathway**: 100% untouched production-ready pipeline generating 5 punchy, high-signal, code-and-parameter-dense technical threads.
- **LinkedIn Stream (Mandatory Aidan Nguyen Tran Signature Style Engine)**: 5 authentic, founder-led story posts mandating Aidan Nguyen Tran's signature writing style (`https://www.linkedin.com/in/aidan-nguyen-tran-277a3a258/`) with 100% dynamic direct evaluator score outputs.

---

## 2. Mandatory Aidan Nguyen Tran Signature Style (`style` Metric)

Every generated LinkedIn post strictly embodies Aidan Nguyen Tran's signature founder-led content engineering style:
- **Upfront Hook**: First sentence strictly $< 10$ words, framing AI, robotics, real-world systems, automation, and marketing as a systems/memory engineering problem.
- **Paragraph Cadence**: Ultra-skimmable 1–2 sentence paragraphs separated by `\n\n` for maximum visual breathing room.
- **Content Teardown Architecture**: Teardown / Build-in-Public format (Hook -> System Pain -> 3-Step Fix -> Outcome -> Reflective Question).
- **Subtle Soft Marketing**: Softly positions automated growth & content systems without hard selling.

### Post Style Mixes:
1. **Post 1**: `Aidan Nguyen Tran Signature Style + Corporate Real-World`
2. **Post 2**: `Aidan Nguyen Tran Signature Style + Fun Conversational`
3. **Post 3**: `Aidan Nguyen Tran Signature Style + Technical Story`
4. **Post 4**: `Aidan Nguyen Tran Signature Style + Mechanical Business`
5. **Post 5**: `Aidan Nguyen Tran Signature Style + Corporate Discussion`

---

## 3. 100% Dynamic Direct Evaluator Scoring

- **Anti-AI Humanness Score**: Directly pulled from `LinkedInEvaluatorService` (`AIPredictabilityAnalyzerService` n-gram entropy $H$ & Wikipedia signs penalty) as an un-truncated, precise floating-point number (e.g. `87.4%`).
- Zero static fallbacks or hardcoded values in text reports.

---

## 4. How to Run

Execute the main pipeline:
```bash
python main.py
```

The system will generate 10 total posts (5 LinkedIn Aidan Nguyen Tran Signature posts, 5 X/Twitter threads) saved to a timestamped file (`posts_YYYYMMDD_HHMMSS.txt`).
