# BUILDCheck India

BUILDCheck India is a pre-submission advisory tool. It reads a 2D site plan supplied as a PDF, a DXF, or both, extracts the measurements that building bye-laws depend on, and checks them against a machine-readable ruleset. The current ruleset is the BBMP Building Bye-laws 2003 for Bengaluru.

Its central design rule: **uncertain evidence never becomes a PASS or a FAIL.** Every extracted value carries a confidence level. If a value is missing, low-confidence or contradicted by other evidence, the affected rule returns *insufficient data*, *requires review* or *conflicting evidence* instead of a verdict.

> **Advisory only.** BUILDCheck India does not approve or reject plans and does not replace review by the competent authority. Treat every result, including PASS, as a prompt for human checking.

This repository accompanies a research manuscript. To find which script and result file produced each number in the paper, see [`REPRODUCE_PAPER.md`](REPRODUCE_PAPER.md). Per-plan extraction results for each evidence source are in [`EXTRACTION_RESULTS.md`](EXTRACTION_RESULTS.md).

---

## Contents

1. [How it works](#how-it-works)
2. [Confidence and rule outcomes](#confidence-and-rule-outcomes)
3. [Rulesets](#rulesets)
4. [Repository layout](#repository-layout)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the web app and API](#running-the-web-app-and-api)
8. [Command-line tools](#command-line-tools)
9. [Optional components](#optional-components)
10. [Evaluation corpus and reproducing the paper](#evaluation-corpus-and-reproducing-the-paper)
11. [Tests](#tests)
12. [Data and privacy](#data-and-privacy)
13. [Known limitations](#known-limitations)

---

## How it works

```
 PDF ─┬─ native text + vector geometry (PyMuPDF)
      ├─ OCR (Tesseract) ............ only when the page has no text layer
      ├─ OpenCV raster geometry ..... always runs
      └─ optional vision-language model (off by default)
                │
 DXF ── entities, text, dimensions, wall/outline reconstruction (ezdxf, shapely)
                │
                ▼
     spatial reasoning: scale, plot / building / road resolution,
     setbacks, floors, building use, derived coverage and FAR
                │
                ▼
     evidence reconciliation ─ final fusion ─ PDF↔DXF reconciliation
                │
                ▼
     NormalizedPlan (every value = number + unit + confidence + provenance)
                │
                ▼
     deterministic rule engine  ◄── data/runtime_rules/<MUNICIPALITY>/rules.json
                │
                ▼
     per-rule results + overall status + templated fix suggestions + PDF report
```

**1. Extraction (`backend/cv_extraction/`)**

- **PDF:** native text and vector paths are read with PyMuPDF. Tesseract OCR runs only when a page lacks a text layer. An OpenCV pass over the rendered page always runs and contributes independent line and contour evidence.
- **DXF:** entities, `TEXT`/`MTEXT`, `DIMENSION` blocks and header units are read with ezdxf. Building outlines are reconstructed from wall networks with shapely. DXF unit inference is weak, so DXF-derived lengths start at low confidence.

**2. Spatial reasoning (`backend/spatial_reasoning/`)**

This stage turns raw evidence into plan facts: drawing scale, plot and building extents, road width, the four setbacks, floor count, building use, and derived coverage and FAR.

**3. Reconciliation.** Three layers decide what value is shipped, and at what confidence:

| Layer | File | Behaviour |
|---|---|---|
| Evidence reconciliation | `evidence_reconciliation.py` | Groups candidates for one field. Tolerance is max(0.05 m, 5 % of the median); outliers are found by MAD z-score > 3.5. A single uncorroborated candidate is capped at MEDIUM. |
| Final fusion | `final_fusion.py` | Merges CV and (optional) vision values. A disagreement becomes CONFLICTING, with no value shipped. A vision-only value with no grounding is capped at LOW. |
| PDF↔DXF reconciliation | `pdf_dxf_reconciliation.py` | When both files are supplied, the PDF value is kept. Agreement within max(0.15 m, 5 %) promotes it to HIGH; disagreement only adds a flag. |

**4. Compliance (`backend/compliance/engine.py`)**

A deterministic engine evaluates every rule in the municipality's `rules.json` against the `NormalizedPlan`. No language model is involved in setting any rule status.

---

## Confidence and rule outcomes

Every field in the `NormalizedPlan` is a `ValueField` with one of five confidence levels:

| Level | Meaning |
|---|---|
| `HIGH` | score ≥ 0.85, e.g. independent sources agree |
| `MEDIUM` | score > 0.75, e.g. one good source |
| `LOW` | weak or single unreliable source |
| `CONFLICTING` | sources disagree; no value shipped |
| `MISSING` | no evidence |

For each rule, the engine applies these checks in order and stops at the first one that matches:

1. Not applicable to this plan → `NOT_APPLICABLE`
2. Applicability cannot be decided → `INSUFFICIENT_DATA`
3. Required value missing → `INSUFFICIENT_DATA`
4. Required value conflicting → `CONFLICTING_EVIDENCE`
5. Required value LOW confidence → `REQUIRES_REVIEW`
6. Threshold itself missing → `INSUFFICIENT_DATA`
7. Otherwise → `PASS` or `FAIL`

The overall status is conservative. Any `FAIL` makes it `FAIL`. Otherwise any uncertain outcome (`CONFLICTING_EVIDENCE`, `INSUFFICIENT_DATA`, `REQUIRES_REVIEW`) outranks `PASS`. `NOT_APPLICABLE` rules are ignored.

Rules may reference only these 18 allow-listed field paths (`backend/rase/schema.py`):

`plot.width`, `plot.depth`, `plot.area`, `building.width`, `building.depth`, `building.footprint_area`, `building.floor_count`, `road.width`, `setbacks.front`, `setbacks.rear`, `setbacks.left`, `setbacks.right`, `coverage`, `far`, `building_height_estimated`, `building_height_excluding_stilt`, `development_area`, `building_use`.

`development_area` and `building_use` can be supplied as metadata (see `--development-area` and `--building-use` below). If they are not supplied and cannot be read from the drawing, they are never guessed.

---

## Rulesets

| Folder | Content |
|---|---|
| `data/runtime_rules/BBMP/rules.json` | 288 rules transcribed from the BBMP Building Bye-laws 2003: 277 `ACTIVE`, 11 `DRAFT`. They cover setbacks, coverage and FAR (Tables 4–6), special-use buildings and high-rise provisions. See `RULESET_AUDIT.md` in the same folder. |
| `data/runtime_rules/PMC_DEMO/rules.json` | 13 illustrative rules. **Not** a verified transcription of any real Pune bye-law; it only shows that a second municipality can be added without code changes. |

The engine evaluates only `ACTIVE` rules. Adding a municipality means adding a folder with a `rules.json`. Optionally, you can also add its regulation PDFs under `data/regulations/<MUNICIPALITY>/` for retrieval.

---

## Repository layout

```
backend/
  app/                 FastAPI app (main.py) and routes (analyze.py, regulations.py)
  cv_extraction/       PDF and DXF extractors, OCR, OpenCV, scale notes, site-plan resolver
  spatial_reasoning/   plan resolution, setbacks, floors, reconciliation and fusion layers
  compliance/          rule engine (engine.py) and optional LLM explainer (explainer.py)
  schemas/             shared Pydantic models: NormalizedPlan, ValueField, enums, results
  runtime_rules/       rule contracts
  rase/                RASE rule schema and LLM-assisted rule drafting (extractor.py)
  rag/                 regulation ingestion (chunking, MiniLM embeddings, FAISS) and hybrid retrieval (FAISS + BM25)
  vision_extraction/   optional VLM backends: smolvlm, qwen, api
  gnn_extraction/      experimental graph model (not used at runtime; see below)
  corpus/              evaluation corpus store, scoring and correction records
  tools/               command-line entry points
  config.py            all settings (read from .env)
frontend/              single-page web UI (index.html, app.js, style.css)
data/
  regulations/         source regulation PDFs per municipality
  runtime_rules/       machine-readable rulesets per municipality
  corpus/              evaluation manifest and truth files (drawings not included)
  uploads/             runtime uploads (git-ignored)
experiments/           paper evaluation scripts, saved results and logs
scripts/               development and diagnostic scripts
tests/                 pytest suite (fixtures build synthetic PDFs/DXFs)
```

---

## Installation

1. Clone the repository and set up the environment:

   ```bash
   git clone <this repository>
   cd BUILD_CHECK_INDIA_
   python3 -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\activate
   pip install -r requirements.txt      # or requirements-lock.txt for the exact evaluated versions
   ```

2. **Install the Tesseract OCR binary** and make sure it is on `PATH`. On Linux: `apt install tesseract-ocr`. On Windows, install it and set `TESSERACT_CMD` in `.env` if it is not on `PATH`. It is needed only for scanned pages, but some tests expect it.

3. `requirements.txt` includes `sentence-transformers` and `faiss-cpu`, which are used for regulation retrieval. Plan checking itself does not need them.

**Optional extras:**

| File | For |
|---|---|
| `requirements-vision.txt` | local vision models (`smolvlm`, `qwen`): torch, transformers. A GPU is recommended. |
| `requirements-vision-qwen.txt` | extra packages for the Qwen-VL backend |


The hosted `api` vision backend needs no extra packages.

---

## Configuration

Copy the template and edit it locally:

```bash
cp .env.example .env
```

**Never commit `.env`.** It is git-ignored. The main settings are:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | empty | Needed only for RASE rule drafting, the LLM explainer, and the `api` vision backend on Groq |
| `VISION_ENABLED` | `false` | Turn the vision-language model on |
| `VISION_BACKEND` | `smolvlm` | `smolvlm`, `qwen` (local) or `api` (any OpenAI-compatible endpoint) |
| `VISION_API_BASE_URL`, `VISION_API_KEY`, `VISION_API_MODEL` | Groq endpoint; falls back to `GROQ_API_KEY` | `api` backend settings |
| `VISION_RENDER_DPI` | `200` | Page render resolution for the VLM |
| `EXTRACTION_TIMEOUT_SECONDS` | `180` | Per-document extraction budget; on expiry an explicit failure is returned |
| `MAX_UPLOAD_SIZE_MB` | `100` | Upload limit (traced DXFs can be large) |
| `USE_EVIDENCE_DECISION_ENGINE`, `USE_DXF_EVIDENCE_DECISION_ENGINE` | `false` | Experimental alternative resolvers, off by default |

See `.env.example` and `backend/config.py` for the full list, including the RAG and RASE settings.

---

## Running the web app and API

```bash
uvicorn backend.app.main:app --reload
```

Open **http://localhost:8000/**. The page lets you:

- upload a PDF, a DXF, or a matching PDF + DXF pair;
- choose the municipality and whether to use vision;
- see every extracted value with its confidence and each rule's outcome;
- correct values and re-check the plan;
- download a PDF report.

`GET /health` returns service status.

| Method and path | Purpose |
|---|---|
| `GET /api/municipalities` | Municipalities with rules, regulations or an index |
| `GET /api/sample-plans` | Plans found in `data/test_plans/` (empty in this public copy) |
| `POST /api/analyze/upload` | Start a job. Form fields: `file`, optional `second_file` (the other format of the same plan), `municipality` (default `BBMP`), `vision` |
| `POST /api/analyze/sample` | Start a job on a file from `data/test_plans/` |
| `GET /api/jobs/{job_id}` | Poll job status and get the result (plan, rule results, overall status, suggestions) |
| `POST /api/jobs/{job_id}/edit` | Override plan values and re-run the rule engine (recorded as a correction, see below) |
| `GET /api/jobs/{job_id}/report.pdf` | Download a PDF report |
| `POST /api/regulations/upload` | Upload regulation PDFs for a municipality and index them (background job) |
| `GET /api/regulations/jobs/{job_id}` | Poll an indexing job |
| `GET /api/regulations`, `GET /api/regulations/{municipality}` | Inspect what is indexed |
| `POST /api/regulations/{municipality}/search` | Retrieve regulation passages (FAISS + BM25) |

**Suggestions** in the result are fixed text templates, one per field of a failed rule, for example "increase the front setback…". They are not generated by a language model.

**Jobs** are held in memory and are lost when the server restarts.

---

## Command-line tools

All tools are run from the repository root with `python -m backend.tools.<name>`.

| Tool | Example | What it does |
|---|---|---|
| `run_full_compliance` | `python -m backend.tools.run_full_compliance plan.pdf --municipality BBMP --output out.json` | Extraction, then plan, then rule check for a PDF. Options: `--vision`, `--backend`, `--building-use`, `--development-area`, `--height-excluding-stilt` |
| `run_pipeline` | `python -m backend.tools.run_pipeline plan.pdf --output plan.json` | Extraction and plan resolution only |
| `run_compliance` | `python -m backend.tools.run_compliance plan.json BBMP --output result.json` | Rule check on a saved plan |
| `run_cv` / `run_vision` / `run_validation` | `python -m backend.tools.run_cv plan.pdf --output cv.json` | Run the CV layer, the vision layer, or the independent CV-vs-vision validation on their own |
| `run_explain_compliance` | `python -m backend.tools.run_explain_compliance result.json` | Optional plain-language explanation of a saved result (needs `GROQ_API_KEY`) |
| `run_ingest_regulations` | `python -m backend.tools.run_ingest_regulations BBMP [--force]` | Index `data/regulations/BBMP/` for retrieval |
| `run_rase_draft` | `python -m backend.tools.run_rase_draft draft BBMP "minimum front setback"` (also `list`, `promote`) | LLM-assisted rule drafting (see below) |
| `run_corpus` | `python -m backend.tools.run_corpus eval --split dev --no-cache --json out.json` (also `validate`, `list`, `add`) | Evaluation corpus management and scoring |

DXF files, and PDF + DXF pairs, are handled through the web API (`/api/analyze/upload`) and through `run_corpus`.

---

## Optional components

All of these are **off by default** or separate from the verdict path.

- **Vision-language model** (`backend/vision_extraction/`, `VISION_ENABLED=true`). This renders PDF pages and asks a VLM for dimensions, areas and regions. Its output only adds evidence: final fusion caps ungrounded vision-only values at LOW, and any disagreement with CV becomes CONFLICTING. The paper's VLM experiments used the `api` backend with `qwen/qwen3.8-27b` on Groq.
- **Regulation retrieval** (`backend/rag/`). This chunks regulation PDFs, embeds them with MiniLM, and searches with FAISS + BM25 using reciprocal rank fusion. It is used to look up clauses and to ground rule drafting. It does not feed the rule engine.
- **RASE rule drafting** (`backend/rase/extractor.py`). An LLM drafts a rule from retrieved clauses. Drafts are tagged `DRAFT`, must cite the retrieved chunks, and are written to a separate `drafts.json`. They enter the live `rules.json` only when a person runs `promote` after reviewing the citation. If too few chunks are retrieved or no API key is set, drafting is refused.
- **LLM explainer** (`backend/compliance/explainer.py`). This writes a plain-language narrative of an already-computed result. It cannot change any status. Without an API key it falls back to the engine's own explanations.
- **Graph model** (`backend/gnn_extraction/`). This holds a GATConv model with training and prediction code. No trained weights are shipped, and the runtime pipeline does not call it. It is kept for future work only.
- **Corrections** (`backend/corpus/corrections.py`). When a user edits values through `/api/jobs/{id}/edit`, a correction record is written to `data/corpus/corrections/`, together with a **copy of the uploaded drawing**. That folder is git-ignored. Keep it that way if the drawings are private.

---

## Evaluation corpus and reproducing the paper

The evaluation corpus is described by:

- `data/corpus/manifest.json`: 8 site plans, each with a PDF and a DXF; there is no PLAN3.
- `data/corpus/truth/PLAN*.json`: truth values. Each value has a verification tier (`printed`, `derived`, `inspected`, `must_abstain` or `unverified`) and records where it came from.

A value counts as correct if it is within max(0.15 m, 5 %) of the truth. Width and depth are scored as an unordered pair.

Everything used in the paper is in `experiments/`, and [`REPRODUCE_PAPER.md`](REPRODUCE_PAPER.md) maps each table and claim to its script and result file. Because the drawings are not distributed (see below), the extraction runs cannot be repeated from this repository alone. The saved outputs in `experiments/results/` can be re-scored and re-analysed.

```bash
python experiments/analyze_extraction.py
python experiments/compliance_eval.py
python experiments/stats_tests.py
```

---

## Tests

```bash
pytest tests/
```

The fixtures in `tests/fixtures/` build synthetic PDFs and DXFs, so most tests need no real drawings. Tests that use the original permit drawings skip automatically when those files are absent.

---

## Data and privacy

- **The original permit drawings are not included.** They contain applicant and engineer names and site addresses. They are available from the authors on request, subject to the owners' permission.
- The same applies to the legacy truth spreadsheet that the truth files name as a source: it lists the plots' addresses.
- In code comments and tests, a property ID, a permit number and a planning-district name that were copied from the drawings have been replaced with made-up values.
- `.gitignore` excludes drawings, uploads, correction copies, caches, local model files and `.env`. Do not override these exclusions with `git add -f`.

---

## Known limitations

- **Small corpus.** The evaluation corpus is small (8 plans, all from one city). The results are not evidence of general accuracy.
- **DXF is weak and not fully deterministic.** DXF extraction is the weakest source. Unit inference is unreliable, and the plot outline reconstructed for one plan changed between runs. DXF values therefore stay at LOW confidence and cannot produce a verdict on their own.
- **The VLM is not deterministic.** It can differ between runs even at temperature 0.
- **Not every bye-law is encoded.** Only rules expressible over the 18 field paths above are checked. Anything else, such as structural, fire or parking detail, is outside scope.
- **Ruleset accuracy.** The BBMP ruleset is a transcription and may contain errors. See `data/runtime_rules/BBMP/RULESET_AUDIT.md`. The PMC ruleset is a demo only.
- **Production readiness.** Jobs are in memory and there is no authentication. The web app is a research prototype, not a production service.
