## Title
<!-- e.g., "feat(pipeline): RAGpack v1.1 (manifest + paragraph offsets + validator)" -->

## What & Why
- **What**: <!-- Short description of modules touched: chunker / embedder / indexer / writer / schemas / cli / notebooks -->
- **Why**: <!-- DeepSearch support / quality / performance / compatibility -->

## Scope
- [ ] chunker  (token/sentence, offsets)
- [ ] embedder (model meta: name/version/hash/dim)
- [ ] indexer  (doc_id/title/path/page/line/timestamp, bm25)
- [ ] writer   (pack.manifest.json, chunks.json, citations.jsonl, metadata.json, embeddings.npy/csv)
- [ ] schemas  (pack.manifest.v1_1, citations.jsonl line)
- [ ] cli      (`nn-pack build|validate|show-manifest`)
- [ ] notebook (build_ragpack.ipynb)
- [ ] docs     (README, examples)
- [ ] tests    (unit/integration/perf)

## Linked Issue
Closes #<ISSUE_NUMBER>

## How to Test
### Build a sample pack
```bash
nn-pack build \
  --input ./corpus \
  --output ./exported/my_pack_v1_1 \
  --chunk-size 512 --chunk-overlap 50 \
  --embedder e5-small-gguf --para-offsets char --bm25-stats on

Validate

nn-pack validate ./exported/my_pack_v1_1
# expect exit 0 and a readable summary (pack_version=1.1, counts, warnings=0)

Acceptance (DoD)
	•	pack.manifest.json (v1.1 fields present)
	•	citations.jsonl has {para_id,start,end,snippet} (≥95% coverage)
	•	validate exits 0 for good pack, non‑zero with clear errors for bad pack
	•	notebooks/build_ragpack.ipynb produces v1.1 end‑to‑end
	•	Backward compat: v1.0 warns “Limited citations” (no crash)
	•	Performance within baseline ±10%

Artifacts
	•	Attach or link a tiny toy pack under exported/ for reviewers
	•	Paste validator summary here

Risks / Migration
	•	Breaking?  No / Yes (describe)
	•	Migration notes: 

Checklist
	•	Conversations resolved
	•	CI green (validator runs on sample pack)
	•	No merge commits (Squash)
	•	Signed or Squash‑verified commit
	•	Code comments in English
	•	Docs updated (README + examples)