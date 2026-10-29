# Independent Research Evaluation & Reproducibility Audit

**Repository:** `PakshajRajput/cti-kg-verification-loop`  
**Working branch:** `research/evaluation-audit`  
**Baseline audit commit:** `1b7936e30e2b64906253901ba4bf018bc8d5c774`  
**Remediation commit:** `2b42305becccf40774e9f4233783ba3bf1e69c42`  
**Latest follow-up test-restoration commit:** `7c2c3ab264bc58f3633ecf413054a102165c12b3`  
**Publication verdict:** **NOT READY FOR PUBLICATION — conditionally remediable**

## Executive finding

The initial audit found that the evaluation mathematics were broadly consistent, but the dataset construction invalidated the benchmark design: annotation and demo trees were recursively ingested together, so exact report clones and their gold triplets could enter the few-shot demonstration pool for held-out reports. A model could reproduce the ground truth from its prompt rather than demonstrate generalizable extraction.

No valid end-to-end research benchmark was executed during that audit. No model-performance result, hallucination-reduction claim, or confidence interval should be inferred from the current repository state.

## Findings and disposition

### CRIT-1 — Few-shot train/test data leakage

**Evidence supplied by the independent audit**
- The loader recursively parsed JSON from both the CTINexus annotation and demo trees.
- The audit reported 308 cached reports, while the intended annotation-only dataset contains 148 unique reports.
- In the seeded split examined by the audit, 53 of 61 test reports had exact copies in the demonstration pool (86.89%), with nearest-neighbor similarity 1.0000.
- The existing preflight checked counts and index consistency but did not reject duplicated report text.

**Remediation committed**
- `_find_json_files` now requires an annotation directory and scans only within that directory; it fails closed if the directory is absent.
- The dataset loader deduplicates normalized source text. Identical copies are dropped only when their annotation signatures agree; conflicting annotations for the same ID or source text raise an error.
- Cached report data is deduplicated and rewritten before splitting.
- Demonstration retrieval excludes the query ID, non-train rows, normalized exact-text clones, and near-identical embedding matches at the configured threshold.
- The research preflight rejects missing/duplicate IDs, empty report text, and duplicated normalized report text.
- Regression tests cover annotation-only discovery, absent annotation directories, clone deduplication, duplicate-ID conflicts, duplicate cache rejection, and explicit synthetic-data opt-in rules.

**Remaining verification**
- The real local cache and demonstrations must be rebuilt with the updated code.
- Confirm the resulting annotation-only count against the source distribution and run the preflight on the rebuilt local data/indexes before any paid benchmark.
- This remote repository connection cannot inspect the user's local `.env`, local indexes, or local generated cache.

### CRIT-2 — Incompatible default model IDs

**Evidence supplied by the independent audit**
- Default Groq model IDs `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` returned HTTP 404 `model_not_found` for the audited account.
- The audit reported a successful connection using `openai/gpt-oss-120b`.

**Remediation committed**
- Default primary/fallback IDs were changed to `openai/gpt-oss-120b` and `openai/gpt-oss-20b` across the LLM client and manual benchmark workflow, with the evaluation CLI primary default aligned.
- Environment-based overrides remain available.

**Remaining verification**
- Confirm primary and fallback availability for the actual provider account immediately before a benchmark. Model availability can change and must not be inferred from the current default strings alone.

### MED-1 — Verifier lacked source-document grounding

**Evidence supplied by the independent audit**
- The verifier was given a predicted triplet and top retrieved ATT&CK entries, but not the original CTI report.
- That design risks accepting a plausible relationship because it appears in ATT&CK, or rejecting a faithful zero-day/victim/campaign extraction because the subject is not indexed.

**Remediation committed**
- The verifier call now receives the source report text.
- The verifier prompt makes source evidence primary and ATT&CK contextual. Absence from ATT&CK alone must not justify a hallucination verdict; report text is treated as untrusted evidence, not instructions.

**Remaining verification**
- Add/complete focused unit tests for source-evidence verdict behavior and evaluate the verifier against independently adjudicated examples. Supplying source text improves grounding but does not by itself prove verifier accuracy.

### HIGH-3 — Pytest discovered vendored third-party tests

**Evidence supplied by the independent audit**
- Bare `python -m pytest -q` traversed `data/ctinexus_raw/ctinexus-main/tests` and failed on missing third-party dependencies.

**Remediation committed**
- `pyproject.toml` sets `testpaths = ["tests"]` and excludes downloaded data and build/environment directories from pytest discovery.

**Remaining verification**
- Run bare `python -m pytest -q` in a clean checkout/environment to ensure the project-wide default path works, not just the CI's explicit test-file list.

### Metric naming

Verifier rejection rate is a process/filtering metric, not a measured hallucination rate. The legacy `PipelineResult.hallucination_rate` remains only as a compatibility alias to the explicitly named `rejection_rate`. Publication claims about hallucination reduction require independently adjudicated source-ground-truth labels and a clearly stated evaluation protocol.

## Validation state

- The targeted CI run for remediation commit `2b42305becccf40774e9f4233783ba3bf1e69c42` completed successfully: **33 tests passed in 0.40 seconds**, and `python -m compileall -q src tests` succeeded.
- Follow-up work restored prior dataset-cache regression coverage instead of replacing those tests. CI for follow-up commit `7c2c3ab264bc58f3633ecf413054a102165c12b3` must be checked before treating that change as validated.
- The local real-data preflight has not been run from this GitHub session.
- A genuine paired baseline/full LLM benchmark has **not** been executed.
- No experimental metrics or confidence intervals are reported here.

## Required release gate

1. Use the updated branch code to delete and rebuild `data/processed/reports.jsonl` and `data/processed/demonstrations.json`.
2. Run `python scripts/validate_research_inputs.py data`; verify the unique annotation count and that ATT&CK, BM25, and Chroma indexes are derived from matching snapshots.
3. Run bare `python -m pytest -q` after the updated CI passes.
4. Run the paired baseline/full evaluation only after the data gate passes, with identical held-out IDs, split seed, and model/provider configuration.
5. Archive per-report audit records, LLM call audit, experiment manifest, plots, and run environment details.
6. Independently adjudicate verifier decisions before making any claim about hallucination reduction.

## Repository state

The work remains isolated on `research/evaluation-audit` in draft PR #1. It has not been merged into `main`. Publication readiness remains **NOT READY** until the local real-data gate and genuine experiment have been completed and their outputs reviewed.
