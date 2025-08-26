# noesisnoema-pipeline

[![GitHub release](https://img.shields.io/github/v/release/raskolnikoff/noesisnoema-pipeline)](https://github.com/raskolnikoff/noesisnoema-pipeline/releases)
[![Platform](https://img.shields.io/badge/platform-Colab%20%7C%20CLI-blue)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-yellow)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview (Updated 2025-06)
**noesisnoema-pipeline** is a minimal, practical pipeline for:

1) **Fetching GGUF LLMs via the Hugging Face CLI** – to run with llama.cpp–compatible runtimes on iOS/desktop/server.
2) **Building a RAGpack (chunks + embeddings)** – split documents, embed them, and ship as a `.zip` your apps can load.

---

## What you can do here

🎥 **Demo video**: [Watch on YouTube](https://youtu.be/XT_cp066NRE)

- Safely download **GGUF** (often quantized) community models from Hugging Face.
- Produce a **RAGpack v1.1** (`chunks.json`, `embeddings.npy`, `citations.jsonl`, `manifest.json`).
- (Optional) Execute the same workflow on **Google Colab** using our helper notebook.

### NEW: RAGpack v1.1 Features
- **Precise Citations**: Paragraph boundaries, character offsets, and optional span‑level source mapping for highlighting.
- **Rich Metadata**: Embedder version, chunker parameters, indexing timestamps, and source diversity metrics.
- **Preview Support**: Snippet extraction with context for DeepSearch UI and API.
- **Validation**: Built‑in CLI validation with `nn-pack validate` including schema checks.
- **Backward Compatible**: Automatically handles v1.0 RAGpacks with clear deprecation warnings.

---

## Step‑by‑step

### 0) Requirements
- macOS / Linux (Windows works best via WSL)
- Python 3.10+ (CLI usage also works on 3.8+)
- `git`

### 1) Hugging Face account & access token
1. Create an account: https://huggingface.co/join  
2. Issue a token: **Settings → Access Tokens → New token**  
   - **Role**: *Read*  
   - Prefer **Fine‑grained** and enable **Gated repos: Read** (required for Meta Llama and other gated repos).
3. For gated models, visit the model page and **Accept** the license/usage policy.

### 2) Install the CLI and log in
```bash
python -m pip install -U "huggingface_hub[cli]"
# or, if you prefer pipx
# pipx install 'huggingface_hub[cli]'

huggingface-cli login    # paste your token when prompted
huggingface-cli whoami   # sanity check
```

> For faster downloads, enable the HF Transfer extension:
> ```bash
> python -m pip install -U hf_transfer
> export HF_HUB_ENABLE_HF_TRANSFER=1
> ```

### 3) Download a GGUF model (recommended: `huggingface-cli download`)
```bash
huggingface-cli download janhq/Jan-v1-4B-GGUF-Q4_K_M \
  --include "*Q4_K_M.gguf" \
  --local-dir models/jan-v1-4b
```

- **TinyLlama (lightweight / quick check)**
```bash
# Example community GGUF repo
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --include "*Q4_K_M.gguf" \
  --local-dir models/tinyllama-1.1b
```

**Verify**
```bash
ls -lh models/<your_model_dir>
shasum -a 256 models/<your_model_dir>/*.gguf   # optional integrity check
```

> **Why the CLI over `git clone`?**  
> Large LFS repos often include many artifacts you don’t need. `huggingface-cli download --include` pulls only what you ask for and avoids common failures/timeouts.

### 4) Build a RAGpack (chunks + embeddings)
Use the notebook under `notebooks/` to turn your documents into a self‑contained **RAGpack**. Output files:
- `chunks.json` — split text using improved token-based chunking
- `embeddings.npy` — NumPy embeddings (fast to load)
- `embeddings.csv` — CSV embeddings (easy to load from Swift/iOS, etc.)
- `metadata.json` — enhanced with chunking parameters

The chunker now uses **token-based splitting** with configurable overlap instead of simple character-based splitting:
- **Chunk size**: Configure in tokens (default 512) for better LLM compatibility
- **Overlap**: Configurable token overlap (default 50) for context preservation
- **Smart boundaries**: Attempts to break at sentence boundaries when possible
- **Unicode support**: Proper handling of non-ASCII text, emojis, and multiple languages

For more details, see `chunker/README.md`.

> RAGpack is model‑agnostic and independent of the GGUF download step.

---

## Optional: run on Google Colab
You can do the same on Colab using the helper notebook. Choose a `repo_id` and download `.gguf` files directly to a mounted Google Drive folder or local Colab storage.

**Notebook**: `gguf_downloader_colab.ipynb`  
Usage:
1. Upload the notebook to Colab and run the first cell to install dependencies.
2. (Optional) Mount Google Drive if you want to persist models.
3. Log in with your HF token (fine‑grained, Read; enable *Gated repos: Read* if necessary).
4. Enter the `repo_id` of the model you want.
5. The notebook lists `.gguf` files → choose one → **Download**.

---

## Troubleshooting
- **403 Forbidden (gated)**: Accept the license on the model page and ensure your token allows **Gated repos: Read**.
- **Nothing downloads / 404**: Double‑check `repo_id` and make sure the repo actually contains `.gguf` files.
- **Slow/unstable**: Install `hf_transfer` and set `HF_HUB_ENABLE_HF_TRANSFER=1`. Use `--resume-download` to continue interrupted downloads.
- **Colab disk limits**: Mount Google Drive and set `--local-dir` to a Drive folder.

---

## Minimal repo layout
```
noesisnoema-pipeline/
├── notebooks/            # RAGpack notebook(s), Colab‑friendly
├── exported/             # Artifacts (kept empty; has a `.gitkeep`)
├── README.md
└── .gitignore
```

`.gitignore` (excerpt):
```
__pycache__/
.ipynb_checkpoints/
*.pyc
*.pyo
*.pyd
.env
.venv
.DS_Store
*.log
*.csv
*.npy
*.jsonl
*.gguf
exported/
models/
dist/
build/
```

---

## Legal Disclaimer

This project provides tools (pipelines, utilities, and examples) for creating RAGpacks and experimenting with Retrieval‑Augmented Generation (RAG). **No copyrighted texts, PDFs, or derivative datasets are included in this repository.**

Demonstration videos (YouTube) are included in the README for educational purposes; they do not distribute copyrighted materials, only show the workflow.

Users are responsible for ensuring that their use of this project complies with applicable copyright and data‑protection laws in their jurisdiction. For example, creating embeddings from copyrighted works may be permissible for private research or experimentation (e.g., under "text and data mining" exceptions), but redistribution of the original texts or derived chunks is typically prohibited.

This repository and its maintainers do not provide legal advice. Use at your own risk.

---

## License
MIT License (see `LICENSE`). Each model retains its own license; always follow the model’s Hugging Face page.

## Acknowledgements
- Hugging Face and the OSS community.
- All contributors to NoesisNoema / RAGfish.

---

## [1.1] - 2025-08
### Added
- **Precise Citations**: Paragraph boundaries, character offsets, and optional span‑level source mapping for highlighting.
- **Rich Metadata**: Embedder version, chunker parameters, indexing timestamps, and source diversity metrics.
- **Preview Support**: Snippet extraction with context for DeepSearch UI and API.
- **Validation**: Built‑in CLI validation with `nn-pack validate` including schema checks.
- **Backward Compatible**: Automatically handles v1.0 RAGpacks with clear deprecation warnings.
