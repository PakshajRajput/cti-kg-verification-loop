# Verification-in-the-Loop: CTI Knowledge Graph Pipeline

> **"Verification-in-the-Loop: Reducing Hallucination in LLM-Based Cyber Threat Intelligence Knowledge Graph Construction"**

A research pipeline that extracts entity-relation triplets from CTI reports using few-shot LLM prompting, verifies them against MITRE ATT&CK via hybrid RAG + cross-encoder reranking, and stores verified triplets in a Neo4j knowledge graph. Measures hallucination reduction vs. extraction-only baseline.

---

## Architecture

```
CTI Report Text
      │
      ▼
┌─────────────┐     few-shot ICL      ┌─────────────────────┐
│  Extractor  │ ──────────────────▶  │  Raw Triplets JSON  │
│   Agent     │                       └──────────┬──────────┘
└─────────────┘                                  │
                                                 ▼
                                       ┌──────────────────┐
                                       │  Canonicalizer   │  ← embedding cluster + LLM merge
                                       └────────┬─────────┘
                                                │
                                                ▼
                              ┌─────────────────────────────────┐
                              │        Verifier Agent           │
                              │  BM25 + ChromaDB + Reranker     │  ← MITRE ATT&CK RAG
                              │  LLM verdict: valid/fix/remove  │
                              └──────┬──────────────────────────┘
                                     │
                        ┌────────────┴─────────────┐
                        │ needs re-extraction?       │
                    Yes │ (attempts < 2)             │ No
                        ▼                           ▼
                  Extractor (loop)          ┌──────────────┐
                                            │  Neo4j KG    │
                                            │  Writer      │
                                            └──────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Neo4j)
- OpenAI API key

### 1. Clone & Install

```bash
git clone <this-repo>
cd verification-in-the-loop-cti-kg

# Install dependencies (use a venv)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -e .
```

### 2. Configure Environment

```bash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux

# Edit .env and set:
#   OPENAI_API_KEY=sk-...
#   NEO4J_PASSWORD=yourpassword
```

### 3. Start Neo4j

```bash
docker run -d \
  --name neo4j-cti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:latest
```

Browser UI available at: http://localhost:7474

### 4. Download & Index Data

```bash
python scripts/setup_data.py
```

This will:
- Clone the CTINexus dataset (~150 annotated reports)
- Download MITRE ATT&CK STIX2 enterprise bundle
- Parse and save to `data/processed/`
- Build ChromaDB vector index + BM25 index over ATT&CK entries

### 5. Smoke Test (Run First!)

```bash
python scripts/test_pipeline.py
```

Runs the full pipeline on 3 sample reports. If this passes, everything is wired correctly.

### 6. Run Full Evaluation

```bash
# Baseline only (extractor, no verification)
python -m src.evaluation.run_evaluation --subset 20 --condition baseline

# Full pipeline (extractor + verifier)
python -m src.evaluation.run_evaluation --subset 20 --condition full

# Both conditions, generate comparison plots
python -m src.evaluation.run_evaluation --subset 20 --condition both
```

Results saved to `results/evaluation_results.json`. Plots saved to `results/figures/`.

### 7. Start the API Server

```bash
uvicorn src.api.main:app --reload
```

Swagger UI: http://localhost:8000/docs

```bash
# Example: extract from a CTI snippet
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "APT29 used Cobalt Strike to compromise a financial institution via spear-phishing emails."}'
```

---

## Project Structure

```
├── data/
│   ├── ctinexus/           ← CTINexus dataset (auto-downloaded)
│   ├── mitre/              ← MITRE ATT&CK STIX entries
│   ├── processed/          ← Parsed reports.jsonl, demonstrations.json
│   └── chroma/             ← ChromaDB vector store (auto-built)
├── results/
│   ├── evaluation_results.json
│   └── figures/            ← P/R/F1 comparison plots
├── scripts/
│   ├── setup_data.py       ← One-time data setup
│   ├── test_pipeline.py    ← Smoke test (run first!)
│   └── demo_query.py       ← Demo Neo4j Cypher queries
├── docs/
│   └── architecture.md     ← Mermaid architecture diagram
└── src/
    ├── ingestion/          ← Data loading & models
    ├── rag/                ← ChromaDB + BM25 + Reranker
    ├── agents/             ← Extractor, Canonicalizer, Verifier nodes
    ├── pipeline/           ← LangGraph graph wiring
    ├── kg/                 ← Neo4j writer
    ├── evaluation/         ← P/R/F1 evaluation scripts
    └── api/                ← FastAPI endpoints
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Triplet Precision | Fraction of predicted triplets matching ground truth |
| Triplet Recall | Fraction of ground-truth triplets recovered |
| Triplet F1 | Harmonic mean of Precision and Recall |
| Hallucination Rate | Fraction of predicted triplets rejected by verifier |

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{ctinexus2024,
  title={CTINexus: Automatic Cyber Threat Intelligence Knowledge Graph Construction Using Large Language Models},
  author={Gao, Peng and others},
  journal={arXiv preprint arXiv:2410.21060},
  year={2024}
}
```
