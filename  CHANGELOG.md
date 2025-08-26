

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# ## [v1.1] - 2025-08
### Added
- Precise Citations: Paragraph boundaries, character offsets, and optional span-level source mapping.
- Rich Metadata: Embedder version, chunker parameters, indexing timestamps, and source diversity metrics.
- Preview Support: Snippet extraction with context for DeepSearch UI and API.
- Validation: Built-in CLI validation with `nn-pack validate` including schema checks.
- Backward Compatible: Automatically handles v1.0 RAGpacks with clear deprecation warnings.

## [v0.1.0] - 2025-08-16

### Added
- RAGpack builder for streamlined RAG dataset creation.
- Colab notebooks for easy experimentation.
- GGUF utilities (optional).
- Validation helpers for data and format checking.
- Example datasets using philosophy PDFs.

### Changed
- Specification simplified for easier use.
- CSV is now the default format; `.npy` is optional.
- Removed tokenizer-specific code for broader compatibility.

### Notes
- `.gguf` files are currently excluded from the release.
- Initial public release on GitHub.
- References:
  - [NoesisNoema](https://github.com/NoesisNoema)
  - [RAGfish](https://github.com/NoesisNoema/ragfish)
  - [rag.fish](https://rag.fish)