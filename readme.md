# Pragmatic Multi-Stage AI Content Engineering Platform (history.txt & post.py Selection Engine)

An autonomous, self-correcting multi-agent content generation platform built inside `main.py`, engineered for **DIGIiq Solution Private Limited** ([LinkedIn Profile](https://www.linkedin.com/company/digiiq-solution-private-limited/posts/?feedView=all)).

The platform features an 11-stage stateful workflow powered by **LangGraph**, **LangChain**, **Google Gemini**, **LlamaIndex**, **Crawl4AI**, **spaCy**, **TextStat**, **CrewAI**, **DeepEval**, **RAGAS**, **TruLens**, and **Prometheus Telemetry**.

---

## 1. Overview

The platform features **decoupled, specialized generation and evaluation pipelines for LinkedIn and X (Twitter)**:

- **X Pathway**: 100% untouched production-ready pipeline generating 5 punchy, high-signal, code-and-parameter-dense technical threads.
- **LinkedIn Stream (Mandatory Aidan Nguyen Tran Signature Style Engine & history.txt Context)**: 5 authentic, founder-led story posts mandating Aidan Nguyen Tran's signature writing style (`https://www.linkedin.com/in/aidan-nguyen-tran-277a3a258/`) with **history.txt non-duplication enforcement** and **100% dynamic direct evaluator score outputs**.

---

## 2. Historical Post Repository (`history.txt`)

`history.txt` acts as the persistent memory store for all published or selected posts:

- **Preserves DIGIiq Brand Perspective**: Pre-populated with DIGIiq Solution Private Limited's actual historical post corpus.
- **Non-Duplication & Non-Contradiction**: `main.py` reads `history.txt` to guarantee that generated posts on one day never repeat opening hooks, metric claims, or core anecdotes, and never contradict previous brand stances.
- **Manual Copy-Pasting**: You can paste text of current/past posts directly into `history.txt` at any time.

---

## 3. Post Selection Utility Script (`post.py`)

[`post.py`](file:///c:/Users/aryan/OneDrive/Desktop/DIGIiq/WORK/social%20media%20Posts%20Automations/post.py) is a standalone selection tool to manage post history:

### Interactive Usage:

```bash
python post.py
```

Displays all generated posts from the latest timestamped output file (`posts_YYYYMMDD_HHMMSS.txt`) and prompts you to select which post numbers (e.g. `1, 3, 7` out of 10) were chosen for posting. Selected posts are automatically formatted and appended to `history.txt`.

### CLI Command Usage:

```bash
python post.py --file posts_20260801_001907.txt --select 1,3,7
python post.py --latest --select 2
```

---

## 4. Mandatory Aidan Nguyen Tran Signature Style

- **Upfront Hook**: First sentence strictly $< 10$ words, framing AI, robotics, real-world systems, automation, and marketing as a systems/memory engineering problem.
- **Paragraph Cadence**: Ultra-skimmable 1–2 sentence paragraphs separated by `\n\n` for maximum visual breathing room.
- **Teardown Architecture**: Upfront Hook -> System Bottleneck -> 3-Step Tactical Solution -> Business Outcome -> Reflective Question.
- **DIGIiq Perspective**: Written from DIGIiq Solution Private Limited's authentic team perspective.

---

## 5. How to Run

1. Execute main pipeline to generate 10 posts:
   ```bash
   python main.py
   ```
2. Pick chosen posts and append to history:
   ```bash
   python post.py
   ```
