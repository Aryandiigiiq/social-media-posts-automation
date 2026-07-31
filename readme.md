# Pragmatic Multi-Stage AI Content Engineering Platform (LinkedIn Typefully/Taplio Suite & X Pipeline)

An autonomous, self-correcting multi-agent content generation platform built inside a single, unified `main.py` file powered by **LangGraph**, **LangChain Tools**, **Google Gemini**, **Crawl4AI**, **LlamaIndex**, **spaCy**, **TextStat**, **DeepEval**, **RAGAS**, **TruLens**, and **Prometheus**.

---

## 1. Overview

The platform features **decoupled, specialized generation and evaluation pipelines for LinkedIn and X (Twitter)**:
- **X Pathway**: 100% untouched production-ready pipeline generating 5 punchy, high-signal, code-and-parameter-dense technical threads.
- **LinkedIn Stream (LangChain Typefully & Taplio Tool Suite)**: 5 story-driven, high-clarity posts formatted via open-source tools inspired by Typefully and Taplio for maximum skimmability, mobile breathing room, and executive reader engagement.

---

## 2. Open-Source LangChain Typefully & Taplio Tools

1. **`typefully_taplio_formatting_tool` (`@tool`)**:
   - **Hook Line Optimization**: Ensures opening sentences are short ($< 12$ words), intriguing, and open a curiosity gap.
   - **White-Space Breathing Room**: Formats posts into clean 1–3 sentence paragraphs separated by `\n\n` for mobile skimmability.
   - **CTA Rotation**: Rotates closures across the 5 posts (`Reflection`, `Lesson Learned`, `Business Takeaway`, `Recommendation`, `Discussion Invitation`).
2. **`taplio_engagement_predictor_tool` (`@tool`)**:
   - Predicts **Skimmability Index (0–100)**, **Comment Trigger Score (0–100)**, and **Mobile Readability Balance**.

---

## 3. Anti-AI Predictability & Wikipedia Signs Audit (`AIPredictabilityAnalyzerService`)

- **N-Gram Transition Predictability Entropy ($H$)**: Calculates 2-gram and 3-gram token transition entropy across text. AI text exhibits predictable low-entropy token sequences; human writing exhibits unpredictable token transitions.
- **Wikipedia 14 Signs of AI Writing Audit**: Detects overused AI vocabulary ("delve", "tapestry", "testament", "game-changer", "landscape", "pivotal", "foster", "garner", "vibrant"), formulaic closures, excessive em-dashes, and passive abstractions.

---

## 4. Post-Level Verification & `[UNFIT]` Header Marking

- **Individual Post Gate**: The evaluation router inspects **each post individually** (`r["anti_ai_score"] >= 82.0` and `r["passed"] == True`).
- **`[UNFIT]` Header Marking**: If after 3 attempts a post's Anti-AI score remains below threshold ($< 82.0$), it is output in the final report marked with a clear header:
  `[UNFIT - FAILED ANTI-AI HUMANNESS GATE]`.

---

## 5. How to Run

Execute the main pipeline:
```bash
python main.py
```

The system will generate 10 total posts (5 LinkedIn Typefully/Taplio optimized, 5 X/Twitter) saved to a timestamped file (`posts_YYYYMMDD_HHMMSS.txt`).
