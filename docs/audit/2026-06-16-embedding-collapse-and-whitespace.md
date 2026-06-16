# Audit — embedding collapse + whitespace loss (root cause)

> **CORRECTION (2026-06-16, measured — supersedes the collapse causal claim below).**
> Empirical follow-up (see the fix PR "fix(extraction): replace whitespace-stripping
> pypdf …") **disproved** the hypothesis that whitespace loss *causes* the vector
> collapse. Measured on the real source PDF + the production GGUF:
> - The broken (pypdf-default) extraction reproduces the baseline at 768-dim:
>   mean_norm **0.893** / off-diag cos **0.797** / eff-dim **72.1**. ✓
> - Re-embedding the **whitespace-fixed** text gives **0.861 / 0.741 / 42.6** — it
>   does **not** move the metrics to a "healthy" regime.
> - A control of **10 maximally-unrelated sentences** embeds at mean_norm **0.754**,
>   eff-dim **8.4/768** — so the high cosine / low eff-dim is **intrinsic nomic
>   anisotropy**, not a whitespace artefact and not a pipeline bug.
> - **Mean-centering** removes it (fixed pack off-diag **0.741 → −0.002**); this is
>   what the NoesisNoema app's mean-centering layer already does.
>
> The whitespace defect is **real and worth fixing**, but its harm is **unreadable
> citations and broken retrieval** (the broken pack returns the same glued chunk for
> every query; the fixed pack returns exactly-relevant passages), **not** the
> anisotropy metrics. Defect 1 ("collapse") and Defect 2 (whitespace) are
> **independent**: fixing extraction does not — and cannot — decollapse raw nomic
> vectors. Read the rest of this document with that correction in mind.

- **Date:** 2026-06-16
- **Scope:** two confirmed defects in a shipped RAGpack v1.2 (nomic-embed-text-v1.5.Q5_K_M, llama.cpp, mean pooling, L2).
- **Method:** read the embedding and extraction code paths end-to-end; no fixes applied. File/line references below are to `main` at the time of audit.
- **Evidence from the shipped pack:** doc-embedding mean-vector norm `0.893`, effective dim `71.9/768`, intra-pack cos `~0.79`; `chunks.json` contains spaceless text e.g. `"Spinozaisoneofthosegreatmen"`.

## TL;DR verdict

| # | Question | Verdict |
|---|----------|---------|
| 1 | `search_document: ` prefix present at embed time? | **YES** — `embedder/llamacpp_embedder.py:270`, present since the embedder was introduced (PR #13). |
| 1b | Pooling include/exclude prefix tokens correctly? | **Correct.** Prefix tokens are mean-pooled in, which is nomic's *documented* contract — not a bug. |
| 2 | Whitespace lost in extraction or detokenization? | **Extraction.** `pypdf` `page.extract_text()` in notebook cell 11 (`build_ragpack_v1_2.ipynb:219`). The chunker only *slices* text (`token_chunker.py:185`); it never detokenizes. |
| 3 | Were embeddings computed on spaceless text? | **YES** — `cli/build_ragpack.py:398-399` embeds the same spaceless `text` field that lands in `chunks.json`. **Regeneration is mandatory.** |

**Headline finding (contradicts the initial hypothesis):** Defect 1 is **not** a missing-prefix bug. The prefix is present and applied correctly. The collapse is a **downstream symptom of Defect 2** — the vectors were computed on spaceless token-soup, which nomic maps to near-identical directions, and the constant prefix contribution then dominates the weak per-document signal. **Fixing the extraction (Defect 2) and regenerating is the actual fix for both defects.**

---

## Defect 1 — document embeddings collapsed onto a common direction

### 1.1 Is `search_document: ` prepended at embed time? — YES

The v1.2 build path is `run_pipeline_v12` → `LlamaCppEmbedder.embed_texts`:

`cli/build_ragpack.py:397-399`
```python
embedder = LlamaCppEmbedder(str(gguf_path))
texts = [c["text"] for c in chunks_with_metadata]
embeddings = embedder.embed_texts(texts)
```

`embedder/llamacpp_embedder.py:63` and `:269-271`
```python
DOCUMENT_TASK_PREFIX: str = "search_document: "
...
for text in texts:
    response = self._model.create_embedding([DOCUMENT_TASK_PREFIX + str(text)])
    rows.append(self._to_matrix(response, expected_count=1))
```

The prefix is concatenated onto **every** text immediately before the llama.cpp call. The probe path used to resolve dimension also prefixes (`:295`). Callers pass raw text and must **not** pre-apply the prefix (module docstring `:26-33`).

**History check:** the prefix has existed since the embedder was first committed — `git log -S 'search_document' -- embedder/llamacpp_embedder.py` returns only `ec03c68` (original add) and `4617d72` (PR #13 merge). There is no commit that removed and re-added it. So the prefix **was present when the shipped pack was built.** A missing prefix cannot be the cause.

### 1.2 Query-side counterpart contract — matches

The app embeds queries with `search_query: ` (stated in the module docstring `:28-33`). This pipeline only embeds documents, so it never emits `search_query: ` itself — but the two prefixes are the matched nomic pair (`search_document: ` for the corpus, `search_query: ` for queries). Both sides are consistent with the nomic-embed-text-v1.5 task contract. **No mismatch.**

### 1.3 Does prefix-token pooling dominate? — No

Mean pooling over the sequence includes the prefix tokens. This is **correct and intended**: nomic-embed-text-v1.5 was trained with the task prefix included in the mean pool, and reference implementations (sentence-transformers / nomic SDK) pool over prefix + body. `_to_matrix` (`:299-331`) mean-pools whatever llama.cpp returns; `pooling_type=LLAMA_POOLING_TYPE_MEAN` is set at load (`:155`). For a normally-spaced ~512-token chunk the prefix is ~5 tokens (~1% of the sequence) and cannot dominate.

The prefix only *appears* to dominate **when the document body carries almost no signal** — which is exactly what spaceless token-soup produces. With a constant prefix vector added to every weak/similar body vector, all rows acquire a shared bias → high mean-vector norm (`0.893`), low effective dim (`71.9/768`), high intra-pack cosine (`~0.79`). The lever is the body content, not the prefix.

### 1.4 Conclusion

Defect 1 has **no independent root cause in the embedding code.** It is the embedding-space manifestation of Defect 2. Once the source text has real word boundaries, the per-document signal is strong enough that the prefix no longer dominates and the collapse disappears.

---

## Defect 2 — chunk text has all whitespace stripped

### 2.1 The chunker does NOT detokenize — it slices

`chunker/token_chunker.py:185`
```python
chunk_text = text[start_pos:end_pos].strip()
```

Every chunk is a **substring slice of the original `text`** (`chunk_text_with_offsets`, `:136-219`). The gpt2 tokenizer is used **only** to *count* tokens and to binary-search a character offset for a target token count:

- `_count_tokens` (`:94-100`) → `len(self.tokenizer.encode(text))`
- `_estimate_char_position_for_tokens` (`:119-134`) → binary search over `text[:mid]`

The tokenizer is **never** used to reconstruct text, so there is **no detokenization step that could drop spaces.** The "detokenizer joins gpt2 tokens without spaces" hypothesis is **false** — confirmed by reading the module. If the input `text` has spaces, the chunks have spaces; if it doesn't, they don't.

### 2.2 The whitespace is already gone in the `.txt` — pypdf extraction

The CLI only accepts `.txt`/`.md` (`cli/build_ragpack.py:52`). PDFs are pre-converted to `.txt` by the **notebook**, which is the only PDF-extraction path in the repo:

`notebooks/build_ragpack_v1_2.ipynb` cell 11 (`:219`)
```python
text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
```

This is `pypdf`'s default extraction mode. `pypdf` reconstructs text from PDF text-positioning operators, and for many PDFs (those that place each word/glyph run with explicit coordinates and **no** inter-word space glyph) the default mode emits adjacent runs with **no separating space** → `"Spinozaisoneofthosegreatmen"`. This is a well-known `pypdf` limitation, not a corruption introduced by this repo's code.

**Root cause location: `notebooks/build_ragpack_v1_2.ipynb` cell 11, `page.extract_text()` with default extraction mode.** The spaceless `.txt` it writes is then faithfully sliced by the chunker and carried, unmodified, into `chunks.json`.

### 2.3 Were the embeddings computed on the spaceless text? — YES (regeneration mandatory)

The exact same `text` field is used for **both** the embedding input and the `chunks.json` payload:

- Embedding input — `cli/build_ragpack.py:398` `texts = [c["text"] for c in chunks_with_metadata]` → `:399` `embedder.embed_texts(texts)`
- `chunks.json` payload — `writer/pack_writer.py:167` / `:120` `chunks_json = [chunk['text'] for chunk in chunks_with_metadata]`

There is no normalization or re-spacing between the two. Therefore the shipped embeddings were computed on spaceless text and are **doubly compromised**: fixing only `chunks.json` would leave the vectors wrong. **The pack must be fully regenerated after the extraction fix.**

---

## Defect 3 — manifest cannot detect prefix/quality mismatch

Today the manifest `embedder` block (`embedder/deterministic_embedder.py:127-151`, schema `schemas/manifest_v1_2.json`) records `embedding_model`, `embedding_version`, `embedding_dimension`, `model_hash`, `dtype`, `pooling`, `l2_normalized`, `runtime` — but **nothing about the task prefixes**, and **no embedding-quality signal.** A pack built with the wrong prefix, or a collapsed pack, imports **silently**. The app has no field to check.

**Plan: record the prefix contract and a quality flag so the app can refuse/correct loudly** (details in the fix plan below).

---

## Fix plan

> Per scope, fixes are **not** implemented in this PR. Defect 2 is the real root cause; Defects 1 and 3 follow from it.

### Fix A — PDF extraction whitespace (Defect 2) — **primary fix**
- **File:** `notebooks/build_ragpack_v1_2.ipynb`, cell 11 (`convert_pdfs_to_txt`).
- **Change:** replace `page.extract_text()` with a spacing-preserving extraction:
  - Minimal: `page.extract_text(extraction_mode="layout")` (pypdf ≥ 3.x).
  - More robust: switch the extractor to `pdfplumber` or `pymupdf` (`fitz`), which preserve word boundaries far more reliably; keep `pypdf` only as a fallback.
- **Add a spaceless-text guard** right after extraction: compute the whitespace ratio (`text.count(" ") / max(1, len(text))`) and **refuse + warn** if it is implausibly low (e.g. `< 0.05` for Latin-script prose). This converts the silent failure into a loud one at ingestion time.
- **Optional defense-in-depth (cheap, isolated — candidate follow-up PR):** add the same whitespace-ratio guard inside `run_pipeline_v12` (`cli/build_ragpack.py`) right after `text = raw_bytes.decode(...)` (`:370`), so any spaceless `.txt` — however produced — fails the build instead of shipping.

### Fix B — embeddings (Defect 1) — regenerate, no embedder code change
- **No change** to `embedder/llamacpp_embedder.py`: the prefix and pooling are correct.
- **Action:** after Fix A, **regenerate the pack** from the re-extracted, properly-spaced text. Verify recovery with the same metrics: mean-vector norm should drop well below `0.893`, effective dim rise well above `71.9`, intra-pack cosine fall well below `0.79`.

### Fix C — manifest contract (Defect 3)
- **`embedder/deterministic_embedder.py`** (`EmbedderMetadata` `:116-151`): add `query_prefix: str = ""` and `doc_prefix: str = ""` fields; emit them in `to_dict()`.
- **`embedder/llamacpp_embedder.py`** (`:175-184`): populate `doc_prefix=DOCUMENT_TASK_PREFIX` (`"search_document: "`) and `query_prefix="search_query: "` when constructing `EmbedderMetadata`.
- **Quality block:** after embedding, compute and record an `embedding_quality` diagnostic (mean-vector L2 norm, effective dimension, mean intra-pack cosine) plus a boolean `mean_centered`/`collapsed` flag derived from thresholds. Natural home: compute in `run_pipeline_v12` and pass through `indexer`/a new `quality` block; surface it via `ragpack/manifest_builder.build_manifest_v1_2`.
- **`schemas/manifest_v1_2.json`**: add optional `query_prefix`, `doc_prefix` to the `embedder` block; add an optional `embedding_quality` object (`mean_vector_norm`, `effective_dim`, `intra_pack_mean_cosine`, `collapsed`).
- **App side (NoesisNoema):** on import, (a) verify `doc_prefix`/`query_prefix` match the app's own embedding contract and **refuse** on mismatch; (b) **warn/refuse** when `embedding_quality.collapsed` is true — making both classes of defect loud instead of silent.

## Appendix — exact code path for the shipped pack

```
notebooks/build_ragpack_v1_2.ipynb  cell 11  page.extract_text()        ← whitespace lost here (pypdf)
        ↓ writes spaceless  <doc>.txt
cli/build_ragpack.py:370            text = raw_bytes.decode(...)         ← spaceless text enters pipeline
cli/build_ragpack.py:376            chunker.chunk_text_with_offsets(...)
chunker/token_chunker.py:185        chunk_text = text[start_pos:end_pos] ← pure slice, spaces NOT re-added
cli/build_ragpack.py:398-399        embed_texts([c["text"] ...])         ← embeddings computed on spaceless text
embedder/llamacpp_embedder.py:270   "search_document: " + text           ← prefix correct; body is the problem
writer/pack_writer.py:167/120       chunks_json = [chunk['text'] ...]    ← same spaceless text into chunks.json
```
