# Pipeline Architecture

## Full System Diagram

```mermaid
flowchart TD
    A([CTI Report Text]) --> B

    subgraph PIPELINE["LangGraph StateGraph"]
        direction TB
        B["🔍 Extractor Agent\nFew-shot ICL • GPT-4o\nJSON triplet extraction"]
        B --> C["🔗 Canonicalizer Agent\nEmbedding clustering\nAlias resolution • Dedup"]
        C --> D["✅ Verifier Agent\nHybrid RAG retrieval\nLLM verdict: valid/fix/remove"]
        D -->|"needs_reextraction\nand iteration < 2"| B
        D -->|"accepted"| E["🏗️ KG Builder\nFinalize triplets"]
    end

    subgraph RAG["RAG Stack"]
        direction TB
        F["ChromaDB\nDense Embeddings\nall-MiniLM-L6-v2"]
        G["BM25 Index\nSparse Lexical\nrank-bm25"]
        H["Cross-Encoder\nReranker\nms-marco-MiniLM"]
        F & G --> |"RRF Fusion"| H
    end

    subgraph KNOWLEDGE["MITRE ATT&CK KB"]
        I["enterprise-attack.json\nSTIX2 bundle\n~700+ techniques\n~140 groups\n~700+ software"]
    end

    I --> F
    I --> G
    RAG --> D

    E --> J[("Neo4j KG\nEntity nodes • Typed edges\nCypher queries")]
    E --> K["FastAPI\nPOST /extract\nGET /graph/stats"]

    subgraph EVAL["Evaluation"]
        L["Baseline Run\nExtractor only\nno verifier"]
        M["Full Pipeline Run\nExtractor + Verifier"]
        N["P / R / F1\nHallucination Rate\nComparison Plots"]
        L & M --> N
    end
```

## Component Details

### Extractor Agent (Stage 2)

| Property | Value |
|---|---|
| LLM | GPT-4o (default) |
| Prompt style | System prompt + few-shot ICL (k=5 retrieved demos) |
| Demo retrieval | Cosine similarity over training report embeddings |
| Output format | JSON array of `{entity1, entity1_type, relation, entity2, entity2_type, confidence}` |
| Retry logic | 3 retries with exponential backoff on API errors |

### Canonicalizer (Stage 3)

| Property | Value |
|---|---|
| Clustering | Embedding cosine similarity, threshold 0.85 |
| Alias resolution | LLM selects canonical from each cluster |
| Deduplication | Hash-based on (entity1, relation, entity2) after lowercasing |

### Verifier Agent (Stage 4) — Core Contribution

| Property | Value |
|---|---|
| Retrieval strategy | Hybrid: BM25 + ChromaDB dense → RRF fusion → cross-encoder rerank |
| Context per triplet | Top-5 ATT&CK entries (techniques, groups, software) |
| LLM verdicts | `valid` / `corrected` / `hallucinated` |
| Feedback loop | If >50% rejected and iteration < 2 → re-extract |
| Knowledge base | MITRE ATT&CK Enterprise STIX2 (~1,500 entries indexed) |

### Knowledge Graph Schema (Neo4j)

```
(:Entity {name, type, normalized_name})
  types: Malware | ThreatActor | Tool | Vulnerability | Target | Technique | Tactic | Campaign | IOC

(:CTIReport {id, source, title})

(:Entity)-[:USES|:EXPLOITS|:TARGETS|:DROPS|...{relation, report_id, confidence}]->(:Entity)
(:CTIReport)-[:CONTAINS]->(:Entity)
```

## LangGraph State Transitions

```mermaid
stateDiagram-v2
    [*] --> Extractor
    Extractor --> Canonicalizer: raw_triplets
    Canonicalizer --> Verifier: canonical_triplets
    Verifier --> Extractor: needs_reextraction AND iteration < 2
    Verifier --> KGBuilder: accepted (iteration >= 2 OR rejection_rate < 50%)
    KGBuilder --> [*]: final_triplets → Neo4j
```

## Evaluation Conditions

```
Condition A (Baseline):
  CTI Text → Extractor → Canonicalizer → [final triplets, NO verification]

Condition B (Full Pipeline):
  CTI Text → Extractor → Canonicalizer → Verifier ⟲ → [verified triplets]

Metrics:
  • Triplet Precision = TP / (TP + FP)
  • Triplet Recall    = TP / (TP + FN)  
  • Triplet F1        = 2·P·R / (P + R)
  • Hallucination Rate = |rejected| / |total_extracted|  [full pipeline only]

Ground truth: CTINexus dataset annotations (150 reports, MIT licensed)
```
