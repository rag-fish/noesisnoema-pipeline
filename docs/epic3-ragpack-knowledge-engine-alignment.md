# EPIC3 RAGPack Knowledge Engine Alignment

**Repository:** `noesisnoema-pipeline`  
**Branch:** `feature/epic3-ragpack-knowledge-engine`  
**Date:** 2026-03-06

## Purpose

This document aligns EPIC3 planning for this repository with the current RAGFish doctrine and the actual branch state before any implementation work begins.

It is intentionally **design-only**. It does **not** authorize code changes beyond the future implementation plan described below.

---

## 1. EPIC3 Scope Clarification

### What EPIC3 means in the current architecture

In this repository, **EPIC3** should be interpreted as the work required to turn `noesisnoema-pipeline` into a **deterministic knowledge-artifact generator** for RAGPack production.

That means:
- ingesting source documents in a controlled way,
- chunking them deterministically,
- generating reproducible embeddings,
- preserving evidence lineage,
- packaging outputs into a stable RAGPack contract,
- and verifying that later retrieval against those artifacts is deterministic.

### What this repository is

This repository is **not** the Noesis/Noema runtime.

It should be treated as a **generation-time toolchain** in the Knowledge Layer. Its job is to create immutable, inspectable artifacts that later runtime components can consume.

### What this repository is not

Per current architecture and RAGFish doctrine, this repository is **not** responsible for:
- routing user invocations,
- enforcing the runtime invocation boundary at request time,
- session handling,
- memory lifecycle enforcement for live conversations,
- response generation,
- runtime fallback behavior,
- or UI/client rendering.

### Generation-time concerns vs runtime concerns

**Generation-time concerns for this repo**
- source inventory and normalization,
- chunking policy and chunk metadata,
- embedding generation and reproducibility,
- manifest construction,
- evidence lineage,
- output hashing and packaging,
- offline verification of artifact integrity and deterministic retrieval expectations.

**Runtime concerns for client/runtime layers**
- `NoemaQuestion` / `NoemaResponse` handling,
- router decisions,
- privacy-level enforcement,
- invocation logging,
- fallback policy,
- session-scoped memory,
- user-visible execution logs,
- synchronous invocation lifecycle.

### Alignment implication

The doctrine documents on invocation boundary, execution flow, memory lifecycle, observability, and security still matter here, but **indirectly**:
- this repo must produce artifacts that are safe and traceable for later runtime use,
- this repo must avoid hidden side effects and undeclared persistence,
- and this repo must not drift into runtime-agent responsibilities.

---

## 2. Current Repository Assessment

### Current folder/file structure

Current branch contents are centered around a small Python package surface:

- `chunker/`
  - `token_chunker.py`
  - `README.md`
  - `__init__.py`
- `writer/`
  - `pack_writer.py`
  - `__init__.py`
- `schemas/`
  - `manifest_v1_1.json`
- `tests/`
  - `test_token_chunker.py`
  - `test_token_chunker_v1_1.py`
- root utilities
  - `create_ragpack.py`
  - `validate_chunker.py`
  - `README.md`
  - `requirements.txt`
  - `.github/PULL_REQUEST_TEMPLATE.md`

### What currently exists

Present in code today:
- **Chunking implementation** via `chunker/TokenChunker`
- **Chunk metadata support** including offsets, paragraph boundaries, and snippets
- **Pack writing** via `writer/PackWriter`
- **Manifest schema draft** in `schemas/manifest_v1_1.json`
- **Chunker-focused tests**
- **Sample pack generation script** in `create_ragpack.py`

Important detail: the current `PackWriter` can write embeddings **if they are already provided**, but there is **no in-repo embedding engine** that computes them from source documents.

### What was previously removed

The current branch history explicitly shows that a large part of the older surface area was removed in commit `115cf3f` (`Pipeline purification: deterministic chunk-only generator (#11)`).

That commit deleted, among other things:
- `embed/`
- `index/`
- `retriever/`
- `nn-pack`
- `nn-retriever`
- `notebooks/`
- `exported/`
- retriever/CLI tests and workflow tests
- sample/demo artifacts and demo scripts

This is not just “missing code.” It is evidence that the repository was intentionally narrowed from a broader workflow into a more limited generator-oriented codebase.

### Current repo vs documented expectations: explicit conflicts

The current repository state conflicts with several documented expectations:

1. **`README.md` still describes notebook-driven and CLI-adjacent workflows**
   - refers to `notebooks/`
   - refers to Colab helpers
   - describes producing full RAGPack outputs end-to-end
   - implies embedder and validator workflows that are not present in the current branch

2. **`.github/PULL_REQUEST_TEMPLATE.md` still assumes modules that do not exist in the branch**
   - `embedder`
   - `indexer`
   - `cli`
   - `notebook`
   - BM25 / validator / backward compatibility flows

3. **Current code is narrower than the docs**
   - `create_ragpack.py` produces a toy zip with synthetic/random-seeded embeddings and `metadata.json`
   - `writer/pack_writer.py` writes `manifest.json`, `citations.jsonl`, `embeddings.npy`, and `embeddings.csv`
   - there is no end-to-end build pipeline from source documents to embeddings to verified retrieval output

### Status of key EPIC3 capabilities

| Capability | Status | Notes |
|---|---|---|
| Chunking | **Present** | `TokenChunker` exists with configurable size/overlap and offset metadata |
| Embedding generation | **Partial / effectively absent** | Pack schema and writer expect embeddings, and `create_ragpack.py` fabricates deterministic sample embeddings, but there is no real embedder module in the current repo |
| Retrieval workflow | **Absent** | No retriever, indexer, BM25, ANN, or verification harness in the current branch |
| Notebook workflow | **Absent** | Notebooks were previously present and were explicitly removed in commit `115cf3f` |
| CLI workflow | **Absent** | Historical references exist, but no current `nn-pack` CLI remains |
| Manifest/schema support | **Partial** | Schema exists; pack writing exists; deterministic packaging and richer lineage are not yet complete |
| Evidence traceability | **Partial** | Current chunk metadata has char offsets and paragraph boundaries, but source lineage is not yet complete enough for EPIC3 |

### Additional current-state observations

- `requirements.txt` currently signals a narrowed intent: “Deterministic RagPack Generator (no retrieval, no embedding)”.
- `PackWriter` currently introduces non-determinism through `uuid.uuid4()` and `datetime.now()`.
- `create_ragpack.py` is a sample/demo producer, not an authoritative production pipeline.
- There is no current source ingestion layer for PDFs, markdown, text corpora, or document inventories.

---

## 3. RAGFish Alignment

### How this repo fits the overall Noesis / Noema architecture

RAGFish doctrine separates:
- **runtime invocation behavior** from
- **artifact preparation and knowledge packaging**.

This repository belongs on the **artifact preparation** side.

It should not implement the runtime execution loop described in `execution-flow.md`. Instead, it should prepare the knowledge assets that runtime components can consume later under the invocation boundary.

### Mapping this repo to the Knowledge Layer

This repository should be treated as the **Knowledge Layer build tool** for RAGPack generation.

Its responsibility is to transform source material into a stable artifact set such as:
- canonical chunk records,
- embeddings,
- lineage/citation records,
- manifest metadata,
- verification outputs,
- packaged archives or directories.

### Relationship to doctrine constraints

Even though this repo is not a runtime executor, it should align with doctrine in the following ways:

- **Invocation Boundary alignment:** no hidden background behavior, no undeclared side effects, no runtime autonomy.
- **Observability alignment:** artifact generation must be reconstructable from configuration, source hashes, and manifest metadata.
- **Security alignment:** no hidden persistence, no undeclared network behavior in the core path, clear trust boundaries for external models/assets.
- **Evaluation alignment:** structural compliance and determinism come before subjective notions of retrieval “quality.”
- **Memory alignment:** this repo must not become a persistent conversational memory store or vector accumulation service outside explicit artifact generation.

### How generated artifacts are consumed later

The intended downstream flow is:
1. This repository generates a deterministic RAGPack.
2. The pack is published or copied into a client/runtime environment.
3. Later, at user invocation time, a client/runtime layer loads the pack.
4. Retrieval occurs inside the runtime boundary using the generated artifacts.
5. Runtime layers use lineage metadata to surface evidence and citations back to the user.

That means this repo is **upstream of runtime retrieval**, not the retrieval runtime itself.

---

## 4. DoD Interpretation

EPIC3 DoD must be translated into repository-specific, generation-side targets.

### Chunking configurable

**Done in this repository means:**
- chunking parameters are explicit inputs, not hidden constants,
- chunk size, overlap, boundary policy, tokenizer identity, and normalization policy are recorded in manifest metadata,
- chunk ordering is deterministic,
- chunk IDs are stable for identical source + config,
- tests prove that identical inputs/config produce identical chunk records.

### Embedding reproducible

**Done in this repository means:**
- a real embedding module exists,
- model identity, version, artifact hash, dimension, backend, dtype, and configuration are captured in the manifest,
- embedding generation uses deterministic execution settings,
- identical source chunks and identical embedder configuration reproduce the same serialized embedding outputs within a defined tolerance,
- reproducibility is validated in automated tests.

### Retrieval deterministic

Because this repo is **not** the runtime retriever, “retrieval deterministic” must be scoped carefully.

**Done in this repository means:**
- the repo defines a deterministic retrieval verification harness for generated artifacts,
- verification uses fixed queries, a fixed similarity metric, fixed tie-break rules, and fixed top-k expectations,
- verification can prove that a generated pack yields the same ranking behavior under the same conditions,
- the repo does not need to become a serving retriever or runtime search service.

### Evidence traceable

**Done in this repository means:**
- every chunk has a stable identity,
- every chunk points back to a source document identity,
- offsets/pages/sections/paragraph references are preserved where source type allows,
- embeddings are linked to the exact chunk IDs they represent,
- the manifest captures lineage and processing metadata needed to reconstruct provenance,
- retrieval verification outputs can identify which source evidence should be surfaced later by runtime layers.

---

## 5. Proposed Target Architecture

This is the desired internal architecture for EPIC3. It is intentionally design-only.

### Proposed modules and responsibilities

| Module | Responsibility |
|---|---|
| `chunker/` or `chunking/` | Deterministic text normalization, segmentation, chunk IDs, chunk metadata, boundary policy |
| `embedding/` | Reproducible embedding generation, model metadata capture, deterministic backend configuration |
| `manifest/` | Manifest assembly, schema versioning, config capture, content hashes, artifact inventory |
| `lineage/` or `evidence/` | Source references, offsets/pages/sections, source hashes, chunk-to-source mapping, embedding-to-chunk mapping |
| `verification/` | Deterministic retrieval verification, golden query fixtures, ranking assertions, artifact integrity checks |
| `writer/` | Canonical serialization, packaging, stable zip/directory output, hash emission |
| `cli/` | Explicit user entrypoints such as build/verify/show-manifest; no hidden background workflows |
| `sources/` or `ingest/` | Canonical source discovery, stable ordering, document metadata extraction, normalization preconditions |

### Recommended internal flow

1. **Source inventory**
   - enumerate inputs deterministically
   - assign stable `source_id`
   - compute source hashes

2. **Chunk generation**
   - normalize content
   - apply chunking policy
   - assign stable `chunk_id`
   - attach offsets and section metadata

3. **Embedding generation**
   - embed chunk text in stable order
   - capture embedder metadata and model hash
   - bind output vectors to chunk IDs

4. **Manifest and lineage assembly**
   - capture configuration versions
   - store file inventory, counts, hashes, timestamps/policies
   - store source lineage summary

5. **Deterministic verification**
   - run pack integrity checks
   - run retrieval determinism checks against fixed fixtures

6. **Packaging**
   - write canonical files
   - produce deterministic pack hash
   - emit final package as directory and/or zip

### Architectural principle

The architecture should stay **build-oriented and explicit**. No notebooks, background jobs, hidden caching, or runtime-agent behavior should be treated as the primary system design.

---

## 6. Determinism Strategy

Determinism is a first-class requirement for EPIC3.

### Chunk generation

Determinism should be guaranteed by:
- canonical input ordering,
- canonical path handling,
- canonical text normalization rules,
- explicit tokenizer/version pinning,
- explicit chunk configuration capture,
- stable chunk ID derivation,
- and tests that compare full chunk records, not only chunk counts.

Recommended controls:
- normalize line endings and whitespace policy explicitly,
- record tokenizer name and tokenizer version/hash,
- sort sources before processing,
- derive chunk IDs from stable inputs such as `source_id + normalized offsets + chunk_text_hash + chunking_config_hash`.

### Embedding generation

Determinism should be guaranteed by:
- pinned model version and artifact hash,
- fixed backend and dtype,
- fixed batch ordering,
- fixed seeds where applicable,
- documented hardware/precision constraints,
- and reproducibility tests with clear tolerance rules.

Recommended controls:
- record model name, version, artifact hash, dimensions, backend, device, dtype,
- avoid hidden remote model updates,
- prefer a reproducible local artifact or pinned immutable remote reference,
- define acceptable numeric tolerance if exact bitwise identity is not portable across environments.

### Retrieval verification

Determinism should be guaranteed by:
- fixed verification queries,
- a fixed scoring method,
- exact tie-break rules,
- no approximate ANN in the verification path,
- and stored expected rankings for regression checking.

Recommended controls:
- use cosine or dot-product consistently and record it in manifest/verification metadata,
- sort ties by stable `chunk_id`,
- keep a small golden verification corpus in tests,
- separate “verification retriever” from any future production runtime retriever.

### Output packaging

Determinism should be guaranteed by:
- canonical JSON serialization,
- stable file ordering,
- explicit file hashes,
- deterministic zip entry ordering and timestamps,
- and manifest metadata that fully describes the build.

Recommended controls:
- sort JSON keys where practical,
- serialize with stable indentation/encoding policy,
- compute SHA-256 for source files and generated artifacts,
- include generator version, dependency versions, schema version, and config hash in the manifest,
- avoid random UUIDs and wall-clock timestamps as primary identifiers for deterministic builds.

---

## 7. Evidence Traceability Strategy

EPIC3 should preserve evidence traceability from source document to chunk to embedding to verification result.

### Required lineage at chunk level

Each chunk should preserve at least:
- `chunk_id`
- `source_id`
- `source_path` or source URI equivalent
- source content hash
- chunk ordinal within source
- start/end character offsets
- token count
- page number(s) when source type supports paging
- section/heading references when source type supports structure
- snippet/preview text for inspection

### Required lineage at embedding level

Each embedding record should preserve at least:
- `chunk_id`
- embedding vector index position
- embedder name/version/hash
- embedding dimensions
- dtype / backend metadata
- embedding generation config hash

### Required lineage at manifest level

The manifest should preserve at least:
- pack version
- build/generator version
- source inventory with hashes
- chunking configuration
- embedding configuration
- artifact file list with hashes
- verification metadata
- provenance summary sufficient for deterministic reconstruction

### Traceability rule

A future runtime layer should be able to answer all of the following from generated artifacts:
- Which source produced this chunk?
- Where in the source did it come from?
- Which embedder created this vector?
- Which pack version and config generated it?
- Which evidence should be shown to the user if this chunk is retrieved later?

### Important constraint

Traceability should be **explicit metadata**, not inferred later from filenames, ordering accidents, or undocumented conventions.

---

## 8. Zero-Scope / Non-Goals

EPIC3 will **not** do the following in this phase:

- no runtime agent logic
- no invocation router implementation
- no `NoemaQuestion` / `NoemaResponse` runtime execution path
- no UI
- no server execution
- no session or conversational memory system
- no hidden persistence beyond intended artifact generation
- no background autonomous behavior
- no automatic runtime fallback logic
- no notebook-first workflow as the primary interface unless explicitly reinstated by design
- no ad-hoc demos as substitutes for production contracts
- no speculative retrieval service or hosted vector database
- no hidden telemetry or network behavior in the core generation path

If any of the above becomes necessary, it should be treated as a separate architectural decision, not as incidental EPIC3 scope creep.

---

## 9. Implementation Plan

Implementation should proceed in small, verifiable stages.

### Stage 1 — Freeze the artifact contract
- Define the EPIC3 source-of-truth package layout and manifest contract.
- Decide the canonical chunk record shape and lineage fields.
- Decide deterministic ID and hashing rules.
- Validation: schema tests and fixture-based manifest serialization tests.

### Stage 2 — Make chunking fully deterministic and traceable
- Extend chunk generation to emit stable IDs and richer source lineage.
- Add source inventory and normalization rules.
- Preserve offsets/pages/sections where available.
- Validation: repeat-run chunk snapshot tests across fixed fixtures.

### Stage 3 — Introduce a real reproducible embedding layer
- Add embedder abstraction and pinned model metadata capture.
- Implement deterministic embedding generation and serialization.
- Bind embeddings explicitly to chunk IDs.
- Validation: reproducibility tests with defined tolerance and stable metadata assertions.

### Stage 4 — Build canonical manifest and packaging
- Centralize manifest assembly.
- Add file hashes, config hashes, dependency versions, and generator version.
- Make directory and zip packaging canonical.
- Validation: repeat-run pack hash or manifest-hash comparison tests.

### Stage 5 — Add deterministic retrieval verification
- Introduce a verification module with fixed queries and stable ranking expectations.
- Keep it as an offline validation tool, not a runtime service.
- Validation: golden ranking tests and negative tests for drift detection.

### Stage 6 — Add explicit CLI entrypoints
- Add a minimal explicit CLI such as `build`, `verify`, and `show-manifest`.
- Ensure commands are transparent, non-autonomous, and loggable.
- Validation: CLI integration tests on a tiny fixture corpus.

### Stage 7 — Documentation and branch hardening
- Update README and contributor docs to match the actual architecture.
- Remove stale references that conflict with the purified branch scope.
- Add a branch-level acceptance checklist for EPIC3.
- Validation: documentation review against current repo structure and test commands.

---

## Key Architectural Decisions

1. **EPIC3 is a Knowledge Layer build effort, not a runtime-agent effort.**
2. **This repository should generate deterministic RAGPack artifacts, not execute Noesis/Noema invocations.**
3. **Current README/template expectations are broader than the actual code and must not be treated as source of truth.**
4. **Embedding and retrieval are not currently implemented here; they must be reintroduced deliberately, not assumed to exist.**
5. **Traceability and determinism must be designed into IDs, manifests, and packaging from the start.**

## Recommended First Implementation Step

The recommended first implementation step is:

**Freeze the canonical chunk/lineage contract before adding any new embedding or retrieval code.**

Concretely, start by defining:
- stable `source_id` and `chunk_id` rules,
- the canonical chunk metadata shape,
- required lineage fields,
- and the manifest fields needed to record chunking deterministically.

This is the correct first step because chunk identity and lineage are the foundation for all later EPIC3 requirements: reproducible embeddings, deterministic retrieval verification, and evidence traceability.

