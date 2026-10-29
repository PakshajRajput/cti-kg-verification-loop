# Evaluation protocol and research safeguards

## What the current metrics mean

- **Triplet precision/recall/F1** use exact normalized (subject, relation, object) matches. Matching is performed inside each report, so a prediction for one report cannot be credited because it appears in another report's annotations.
- **Entity precision/recall/F1** evaluate the set of normalized entity strings per report.
- **Verifier rejection rate** is the fraction of triplets rejected in the final verifier pass. It is a process metric, not a validated hallucination rate: a verifier may reject a correct relation or accept an incorrect one. Baseline runs report this metric as not applicable because no verifier is run.
- **Micro metrics** pool report-local true-positive, prediction, and gold counts. **Macro metrics** average per-report scores, giving each report equal weight. Both are emitted in the evaluation JSON.

The legacy `hallucination_rate` field remains for compatibility. New analysis should use `rejection_rate` and must not describe it as a measured hallucination rate without independent adjudication.

## Dataset integrity

The evaluator requires at least 10 annotated reports by default and raises an error rather than silently mixing synthetic examples into the research dataset. `--allow-synthetic-data` explicitly permits synthetic supplementation for demonstration/debugging only. Do not use results from that mode as benchmark evidence.

The results JSON records the model/configuration, split seed and fraction, report IDs, elapsed runtime, and hashes of the dataset content, source prompts, and key implementation files. The sibling `experiment_manifest.json` includes provider, configured primary/fallback model IDs, train/test report IDs, a dataset fingerprint (hash only; report text is not copied), prompt hashes, implementation hashes, Python version, and decoding limits. The `--model` CLI override is passed into extractor, canonicalizer, verifier, and JSON-repair calls; the manifest records the requested primary model and configured fallback separately. Save this manifest with aggregate results and `per_report_audit.jsonl` for every experiment. The CLI accepts `--seed` and `--test-split` to reproduce the split. If the provider falls back after an error, inspect its logs and record the actual model per request before making fine-grained model-comparison claims.

For reproducibility, archive the exact dataset version/hash, ATT&CK/STIX snapshot, environment/dependency lock, model and embedding model identifiers, prompt versions, seed/split, and raw per-report predictions. Do not publish only aggregate metrics.

## Required experimental design

1. Use a frozen test split and keep all examples used for few-shot demonstrations out of the test split.
2. Run baseline extraction and the full pipeline on the same test report IDs.
3. Add ablations: no verifier, verifier without re-extraction, full verification loop; then compare lexical retrieval, dense retrieval, hybrid retrieval, and reranking.
4. Have independent annotators or an adjudication process label a representative sample of verifier decisions. Report false-accept and false-reject rates against these labels.
5. Report micro and macro precision/recall/F1, per-report results, confidence intervals or paired bootstrap intervals, latency, token/API cost, and failures.
6. Do not claim improvement until the same split, prompts, models, and evaluation code are used across conditions and the outputs have been inspected.

## Current limitations

- The legacy result model includes `hallucination_rate`; it is retained as a compatibility alias for rejection rate.
- The audit JSONL captures model-level verifier decisions and ATT&CK context IDs, not the full retrieved documents or complete source report text.
- Extractor and verifier API failures now fail the report/evaluation instead of being counted as empty predictions or valid verdicts. Review the resulting error record before rerunning.
- The connector can trigger CI but cannot execute the full project environment with real dataset/index/model credentials; run an end-to-end smoke test before treating metrics as final.
