"""
Centralized configuration.

Nothing in this codebase should read os.environ directly, and nothing
should hard-code API keys, paths, ports, model names, upload limits or
geometry tolerances. Import `get_settings()` everywhere instead.

Values are loaded from a `.env` file (see .env.example for the full list
of supported keys) with environment variables taking precedence over the
file, and the defaults below taking precedence only when neither is set.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "buildcheck-india"
    environment: str = Field(default="development")
    api_port: int = Field(default=8000)
    api_host: str = Field(default="0.0.0.0")

    # --- Paths (all relative to repo root unless absolute) ---
    data_dir: Path = Field(default=Path("data"))
    regulations_dir: Path = Field(default=Path("data/regulations"))
    runtime_rules_dir: Path = Field(default=Path("data/runtime_rules"))
    test_plans_dir: Path = Field(default=Path("data/test_plans"))
    vector_store_dir: Path = Field(default=Path("data/vector_store"))
    upload_dir: Path = Field(default=Path("data/uploads"))

    # --- Database ---
    database_url: str = Field(default="sqlite:///./buildcheck.db")

    # --- OCR ---
    tesseract_cmd: str | None = Field(
        default=None,
        description="Explicit path to the tesseract executable (e.g. "
        r"'C:\Program Files\Tesseract-OCR\tesseract.exe' on Windows). `pip install "
        "pytesseract` only installs the Python wrapper -- Tesseract itself is a separate "
        "binary that must be installed via the OS package manager / installer and is not "
        "always on PATH after that (very common on Windows). Leave unset if `tesseract` is "
        "already resolvable on PATH; set this to point pytesseract at it explicitly "
        "otherwise. Has no effect on the 'pytesseract and Pillow are required' import-time "
        "error, which means the Python packages themselves are not installed.",
    )

    # --- LLM / RAG (Groq + embeddings) ---
    groq_api_key: str = Field(default="")
    groq_model_name: str = Field(default="llama3-70b-8192")
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)
    rag_top_k: int = Field(default=8)

    # --- RAG ingestion (chunking) ---
    rag_max_chunk_tokens: int = Field(
        default=512, description="Max approx tokens per regulation chunk before sub-splitting."
    )
    rag_chunk_overlap_tokens: int = Field(
        default=50, description="Approx token overlap kept between sub-split chunks."
    )

    # --- RAG retrieval (dense / BM25 / RRF fusion) ---
    rag_top_k_dense: int = Field(default=20, description="Candidates fetched from FAISS before fusion.")
    rag_top_k_bm25: int = Field(default=20, description="Candidates fetched from BM25 before fusion.")
    rag_top_k_final: int = Field(default=8, description="Final chunk count returned after RRF fusion.")
    rag_rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant.")

    # --- RAG document classification (filename-based inference) ---
    # Extended with the AMENDMENT/PARKING/FIRE categories from the previous
    # (compliance.zip) regulation_registry.classify_regulation vocabulary --
    # this project's two previously-UNKNOWN regulation PDFs ("...Parking
    # Rules, 2026..pdf" and "Revised setback gazette copy.pdf") are exactly
    # a parking regulation and a gazette amendment, so they were being
    # classified as UNKNOWN and shown that way in the compliance citation UI.
    # This purely improves retrieval/citation METADATA (what document a
    # clause is attributed to); it does not change chunking, indexing,
    # retrieval ranking, or any compliance verdict.
    rag_doc_type_keywords: dict[str, str] = Field(
        default_factory=lambda: {
            "nbc": "NBC",
            "national_building": "NBC",
            # Amendments/notifications/gazettes are checked before the
            # generic "dcr"/"zoning" keywords so a filename like "Revised
            # setback gazette copy.pdf" is classified by what it actually
            # is (an amendment to the base DCR) rather than falling through
            # to a broader byelaw/zoning match.
            "amendment": "AMENDMENT",
            "amendments": "AMENDMENT",
            "notification": "AMENDMENT",
            "gazette": "AMENDMENT",
            "revised setback": "AMENDMENT",
            "parking": "PARKING",
            "fire": "FIRE",
            "dcr": "DCR",
            "byelaw": "DCR",
            "bye-law": "DCR",
            "bylaw": "DCR",
            "building_bylaw": "DCR",
            "zoning": "DCR",
            "master_plan": "DCR",
        },
        description="filename substring -> doc_type, checked in insertion order.",
    )
    rag_city_keywords: dict[str, str] = Field(
        default_factory=lambda: {
            "bbmp": "Bengaluru",
            "bangalore": "Bengaluru",
            "bengaluru": "Bengaluru",
            "mcgm": "Mumbai",
            "mumbai": "Mumbai",
            "dda": "Delhi",
            "delhi": "Delhi",
            "pmc": "Pune",
            "pune": "Pune",
            "gvmc": "Vizag",
        },
        description="filename substring -> city, checked in insertion order.",
    )

    # --- RASE (regulation -> RuntimeRuleDefinition drafting) ---
    rase_draft_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0,
        description="Groq sampling temperature for rule drafting. 0.0 for deterministic, "
        "least-invented threshold extraction.",
    )
    rase_min_retrieval_chunks: int = Field(
        default=3, description="Minimum number of retrieved regulation chunks required before "
        "attempting to draft a rule; below this, drafting is refused rather than hallucinated.",
    )

    # --- Architecture V2, Phase 8: evidence decision engine feature flag ---
    use_evidence_decision_engine: bool = Field(
        default=False,
        description="When True, backend.spatial_reasoning.pipeline.build_normalized_plan ships field "
        "values/confidence from backend.spatial_reasoning.evidence_decision.decide_field instead of "
        "final_fusion.apply_final_agreement_to_plan's CV+Vision-only averaging, for the fields both "
        "cover. The old path is ALWAYS still computed and logged for comparison regardless of this "
        "flag (see ARCHITECTURE_V2.md's Phase 8/9) -- this only controls which one ships. Defaults to "
        "False (unchanged existing behavior) until the Phase 9 old/new comparison across all real "
        "plans is complete.",
    )

    # --- Architecture V2, Phase DXF-3: DXF structural-hypothesis decision engine feature flag ---
    use_dxf_evidence_decision_engine: bool = Field(
        default=False,
        description="When True, backend.cv_extraction.dxf_extractor.DXFHybridExtractor's own region "
        "resolution (_resolve_via_regions) ships the plot/building/road entries from whichever region "
        "backend.spatial_reasoning.structural_hypothesis_decision.decide_structural_hypothesis accepts "
        "for PLOT_BOUNDARY, instead of the max-score `best` selection. The old max-score selection is "
        "ALWAYS still computed and logged for comparison regardless of this flag (see "
        "ARCHITECTURE_V2.md's DXF redesign plan) -- this only controls which one ships. Defaults to "
        "False (unchanged existing behavior) until a DXF-specific old/new comparison across all real "
        "DXF plans is complete.",
    )

    # --- Optional multimodal architectural-plan extraction ---
    vision_enabled: bool = Field(
        default=False, description="Enable local vision-language semantic extraction in the PDF pipeline."
    )
    vision_backend: str = Field(
        default="smolvlm",
        description="Which vision-language backend to use: 'smolvlm' (default -- local "
        "SmolVLM2-2.2B-Instruct, ~4.4GB/~5GB RAM), 'qwen' (local Qwen2.5-VL-7B-Instruct, "
        "~16GB, heavier but stronger), or 'api' (no local weights at all -- calls a hosted, "
        "OpenAI-Chat-Completions-compatible vision endpoint such as Groq, OpenAI, or "
        "OpenRouter; use this on machines without enough RAM/VRAM for a local VLM).",
    )
    vision_model_name: str = Field(
        default="",
        description="Optional explicit Hugging Face model id override. Empty string (default) "
        "uses the selected backend's own default model.",
    )
    vision_render_dpi: float = Field(
        default=200.0, ge=72.0, le=300.0,
        description="PDF rasterization DPI used for vision-model input."
    )
    vision_max_new_tokens: int = Field(
        default=5500, ge=64, le=6000,
        description="Max tokens generated per page by the vision backend. IMPORTANT: too low "
        "a value silently breaks extraction, not just truncates it -- a real multi-view "
        "architectural sheet (site plan + several floor plans + elevation/section/details, all "
        "on one page) can easily need 2500+ tokens for its regions+dimensions+areas JSON alone, "
        "and a response cut off before its closing brace fails to parse at all "
        "(BaseArchitecturalPlanExtractor._extract_json raises), which drops that page's vision "
        "evidence entirely rather than partially. `areas` is the LAST field in the schema, so "
        "it's the first thing lost to truncation -- confirmed directly on a real 9-region plan "
        "(PLAN5: site plan + 3 floor plans + section + elevation + title block + area "
        "statement), whose raw response cut off exactly at the start of `\"areas\"` at the old "
        "default of 3000. Lower this only for a quick CPU smoke test on a simple single-view "
        "page; raise it further (up to 6000) for an even busier sheet.",
    )

    vision_focus_max_new_tokens: int = Field(
        default=1200, ge=128, le=4000,
        description="Max tokens for the independent Vision-only site-plan focus pass. "
        "The focused crop needs far fewer tokens than the full architectural sheet, "
        "so keeping this lower reduces provider rate-limit pressure while leaving enough "
        "room for dimensions, setbacks, and area JSON."
    )
    vision_site_focus_render_dpi: float = Field(
        default=400.0, ge=72.0, le=600.0,
        description="Render DPI used ONLY for the region-focused Vision passes (site plan, "
        "area statement, elevation/section height). Deliberately higher than "
        "vision_render_dpi: these crops are re-rendered directly from the PDF's own vector "
        "page at this resolution (see vision_extraction.page_renderer.render_pdf_region), not "
        "upscaled from the full-page raster -- small printed text (setback decimals, table "
        "cells) needs the extra real pixel detail.",
    )
    vision_area_focus_max_new_tokens: int = Field(
        default=800, ge=128, le=4000,
        description="Max tokens for the AREA_STATEMENT focused Vision pass (reads the sheet's "
        "printed area-statement table row by row). Kept modest -- a table crop's JSON is small "
        "-- both because it's genuinely enough here and because several hosted-API free/low "
        "tiers enforce a hard per-request output-token-per-minute ceiling well under 1000 "
        "combined across this pass plus the full-page and other focus passes on the same page.",
    )
    vision_area_focus_render_dpi: float = Field(
        default=450.0, ge=72.0, le=600.0,
        description="Render DPI for the AREA_STATEMENT focus crop specifically -- table text is "
        "often the smallest print on the whole sheet.",
    )
    vision_height_focus_max_new_tokens: int = Field(
        default=800, ge=128, le=4000,
        description="Max tokens for the ELEVATION/SECTION focused Vision pass (reads building/"
        "floor/plinth/parapet height dimension lines). Kept modest for the same rate-limit "
        "reason as vision_area_focus_max_new_tokens.",
    )
    vision_height_focus_render_dpi: float = Field(
        default=400.0, ge=72.0, le=600.0,
        description="Render DPI for the ELEVATION/SECTION height focus crop.",
    )

    # --- 'api' vision backend only (no local weights -- hosted VLM over HTTP) ---
    vision_api_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL of an OpenAI-Chat-Completions-compatible endpoint. The default "
        "points at Groq (already used for text via GROQ_API_KEY/groq_model_name). Swap this "
        "to e.g. https://api.openai.com/v1 or https://openrouter.ai/api/v1 for other providers.",
    )
    vision_api_key: str = Field(
        default="",
        description="API key (bearer token) for the 'api' vision backend. Falls back to "
        "groq_api_key when empty and vision_api_base_url is left at the Groq default, so a "
        "project already configured for Groq text/RAG needs no extra key.",
    )
    vision_api_model: str = Field(
        default="",
        description="Model id passed to the hosted API, e.g. 'qwen/qwen3.6-27b' on Groq or "
        "'gpt-4o' on OpenAI. Empty uses vision_model_name, then the backend's own default. "
        "Hosted-vision-model line-ups change often -- check the provider's current docs "
        "(e.g. https://console.groq.com/docs/vision for Groq) rather than assuming the "
        "built-in default is still current/production-ready.",
    )
    vision_api_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0,
        description="Sampling temperature for the 'api' backend. 0.0 for the most deterministic, "
        "least-invented structured-extraction output.",
    )
    vision_api_json_mode: bool = Field(
        default=True,
        description="Ask the API for response_format={'type': 'json_object'} (supported by "
        "Groq and OpenAI). Disable for providers that reject the parameter.",
    )
    vision_api_reasoning_format: str = Field(
        default="hidden",
        description="For reasoning/'thinking' models (e.g. Groq's qwen/qwen3.6-27b): controls "
        "how chain-of-thought is returned. 'hidden' (default) -- no reasoning in the response, "
        "content is the direct answer; 'parsed' -- reasoning returned in a separate field, "
        "content is still just the answer; 'raw' -- reasoning is inlined in content wrapped in "
        "<think>...</think> tags (base.py strips these defensively, but reasoning still eats "
        "into the vision_max_new_tokens budget before the actual JSON is generated, so 'raw' is "
        "the most likely to truncate on a low token budget). Set to '' to omit the parameter "
        "entirely for providers that don't support it. Per Groq's docs, 'parsed' or 'hidden' is "
        "REQUIRED when combining a reasoning model with JSON mode (vision_api_json_mode=true).",
    )
    vision_api_reasoning_effort: str = Field(
        default="none",
        description="For hybrid thinking/non-thinking models that support it (e.g. Groq's "
        "qwen/qwen3.6-27b): 'default' spends a large, variable amount of the "
        "vision_max_new_tokens budget on chain-of-thought before answering -- for a "
        "structured-extraction task like this, that reasoning can consume the entire budget "
        "and leave zero tokens for the actual JSON (finish_reason='length', empty content). "
        "'none' (default here) skips deep thinking and answers directly, which is what this "
        "extraction prompt needs. Set to '' to omit the parameter for models/providers that "
        "don't support it (sending an unrecognized value is more likely to error than an "
        "unrecognized param name, so this is opt-in per model).",
    )
    vision_api_timeout_seconds: float = Field(
        default=120.0, gt=0, description="Per-request HTTP timeout for the 'api' vision backend."
    )
    vision_api_max_retries: int = Field(
        default=3, ge=0, le=10,
        description="Retries (with backoff -- honors a 429's Retry-After header when the "
        "provider sends one, otherwise backs off toward a ~60s window, since a 429 on this "
        "project's hosted-API testing has consistently been a per-MINUTE token quota, not a "
        "generic transient failure) for transient 429/5xx/network errors on the 'api' vision "
        "backend. A single page can make several sequential region-focused Vision calls sharing "
        "one quota (see vision_focus_max_new_tokens etc.), so the last call in that sequence "
        "needs real retry headroom, not just 1-2 quick attempts.",
    )
    vision_only_dimension_min_confidence: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Minimum vision-model confidence required before a dimension the "
        "deterministic CV/OCR pipeline never found at all (no matching native/OCR "
        "Dimension candidate on that page) is admitted as its own evidence source. Higher "
        "than dimension_association_min_confidence deliberately -- this reading has no "
        "corroborating dimension-line geometry at all, so it needs to be a genuinely "
        "confident model read, not just plausible.",
    )
    vision_api_max_image_pixels: int = Field(
        default=33_177_600,
        description="Hard pixel-count budget (width*height) enforced before sending an image "
        "to the 'api' vision backend. Large-format architectural sheets rendered at typical "
        "DPI easily exceed a provider's cap (e.g. Groq rejects images over 33,177,600px with "
        "HTTP 400) -- images over this are downscaled (preserving aspect ratio) rather than "
        "failing the page. Set to 0 to disable downscaling.",
    )

    # --- Upload limits ---
    # A vectorized/traced DXF (walls/boundaries retraced as many thousands
    # of tiny line/polygon fragments -- see dxf_extractor.py's own notes on
    # the real 6,346-fragment sample) is legitimately much larger than a
    # PDF sheet; 25MB rejected a genuine real-world upload (45.7MB) with a
    # bare 413, so this is sized with real headroom above that rather than
    # just above the one observed file.
    max_upload_size_mb: int = Field(default=100)
    allowed_upload_extensions: tuple[str, ...] = (".pdf",)

    # --- Geometry ---
    geometry_tolerance_m: float = Field(
        default=0.05, description="Snap/merge tolerance for geometric reasoning, in metres"
    )
    default_points_per_metre: float = Field(
        default=72.0, description="Fallback page->metric scale when no scale bar is detected"
    )

    # --- Phase 3.1: page-frame / candidate-validity thresholds ---
    page_frame_touch_tolerance_pts: float = Field(
        default=4.0,
        description="A candidate edge within this many page-points of the page boundary is "
        "considered to be 'touching' it, for page-frame detection.",
    )
    page_frame_area_fraction_threshold: float = Field(
        default=0.92,
        description="A candidate whose polygon area covers more than this fraction of the page "
        "area, AND touches the page boundary, is rejected as a drawing-sheet frame rather than a "
        "plot boundary.",
    )
    min_valid_polygon_area_pts2: float = Field(
        default=4.0, description="Polygons smaller than this area (in page-points^2) are degenerate."
    )
    max_valid_perimeter_area_ratio: float = Field(
        default=25.0,
        description="Polygons whose perimeter^2/area exceeds this are rejected as slivers/degenerate "
        "(a square has perimeter^2/area == 16; this allows fairly elongated but still-plausible shapes).",
    )
    candidate_dedupe_tolerance_pts: float = Field(
        default=2.0, description="Bounding boxes within this tolerance are treated as duplicate candidates."
    )

    # --- Phase 3.1: dimension-to-geometry association ---
    dimension_association_search_window_pts: float = Field(
        default=40.0,
        description="Spatial search-window radius (page-points) used to find candidate geometry "
        "lines near a dimension-text annotation, before association scoring.",
    )
    dimension_association_min_confidence: float = Field(
        default=0.35,
        description="Minimum association score (0-1) required before a dimension candidate is "
        "linked to a geometry line. Below this, geometry association is left unresolved (None) "
        "rather than guessed.",
    )

    # --- Phase 3.1: local scale estimation ---
    local_scale_region_padding_factor: float = Field(
        default=1.5,
        description="The selected plot's bounding box is padded by this factor (relative to its "
        "own diagonal) to define the 'local' drawing region used for scale-sample search before "
        "falling back to whole-document scale samples.",
    )
    scale_sample_disagreement_tolerance: float = Field(
        default=0.20,
        description="Relative spread above which local scale samples are treated as CONFLICTING "
        "rather than resolved to a single confident scale.",
    )

    # --- Phase 3.1: performance / bounded processing ---
    # NOTE: this budget wraps BOTH the native/OpenCV pass AND (in Vision/Fusion
    # mode) the Vision backend call within it -- see
    # PDFHybridExtractor._fold()/extract_with_debug(). Benchmarking against the
    # project's own sample plans showed real, successful CV-only extractions
    # taking 30-70+s depending on drawing complexity/dimension count (e.g. a
    # 140-dimension plan), which was intermittently exceeding the previous 45s
    # default under normal, non-pathological load -- causing a legitimate but
    # slow extraction to be aborted and returned as an all-MISSING result
    # (indistinguishable from a real extraction failure), and making Fusion
    # mode (CV + a VLM call, sharing this same budget) time out routinely.
    # 180s keeps the original safety intent (still bounded, so a truly
    # pathological/degenerate PDF can't hang forever) while giving normal
    # documents -- and a Vision backend call layered on top in Fusion mode --
    # realistic headroom to finish.
    extraction_timeout_seconds: float = Field(
        default=180.0, description="Hard wall-clock budget for a single document's extraction "
        "(native/OpenCV pass, plus the Vision backend call when Vision/Fusion mode is active). "
        "On expiry, extraction returns an explicit failure/warning instead of hanging."
    )
    max_raster_dpi: float = Field(
        default=300.0, description="Upper bound on rasterization DPI for OCR/OpenCV passes."
    )
    max_opencv_contours: int = Field(
        default=400, description="Upper bound on OpenCV contours kept per page (largest-first)."
    )
    max_opencv_lines: int = Field(
        default=1000, description="Upper bound on OpenCV Hough line segments kept per page (longest-first)."
    )
    max_dimension_association_lines: int = Field(
        default=500,
        description="Upper bound on candidate geometry lines considered per dimension-text "
        "association search, to avoid unrestricted every-text-x-every-line comparison.",
    )
    max_dxf_role_inference_polygons: int = Field(
        default=300,
        description="Upper bound on closed polygons fed into DXF plot/building/road role "
        "inference (backend.spatial_reasoning.site_graph.build_site_graph), largest-area-first. "
        "That graph build is O(n^2) in polygon count by design (see its own docstring) -- fine "
        "for a normal architect-authored DXF's few hundred closed shapes, but a DXF produced by "
        "vectorizing a scanned/rasterized sheet can contain many thousands of tiny closed "
        "fragments (e.g. each dash of a dash-dot property line rendered as its own tiny filled "
        "quad), which made this pipeline effectively hang -- 6000+ fragments squared is tens of "
        "millions of pairwise geometry comparisons, run three times over (once each for plot/"
        "building/road) before this cap and the single-shared-graph-build fix below. The real "
        "plot/building/road are always among the largest closed shapes on the sheet, so capping "
        "to the largest N loses nothing relevant while making role inference bounded.",
    )
    max_dxf_reconstruction_segments: int = Field(
        default=8000,
        description="Upper bound on open-line-work segments fed into "
        "backend.cv_extraction.dxf_reconstruction.reconstruct_building_polygon's graph-building "
        "stages (_snap_endpoints/_merge_collinear), longest-segment-first. _merge_collinear "
        "iteratively contracts degree-2 nodes and is worst-case O(nodes^2) -- the same unbounded-"
        "graph-stage bug class already found and capped for DXF role inference above "
        "(max_dxf_role_inference_polygons), recurring on this sibling code path. A real wall or "
        "boundary run is always among the longer segments on the sheet; short segments are "
        "disproportionately tracing noise/duplicate fragments, so capping to the longest N loses "
        "essentially nothing relevant while making reconstruction bounded. Set comfortably above "
        "the 5,000-6,000-segment range observed on real bundled DXF regions so normal files never "
        "hit the cap.",
    )
    max_dxf_region_merge_candidates: int = Field(
        default=6,
        description="Upper bound on how many of a sheet's structurally-highest-scoring DXF regions "
        "(backend.cv_extraction.dxf_extractor._resolve_via_regions) are tried pairwise as a possible "
        "single merged drawing, before evidence recovery runs. Density-based region clustering can "
        "fragment one real, large, sparse boundary (e.g. a dash-dot property line) into several "
        "disjoint regions along its own straight sides -- each fragment then scores as its own "
        "incomplete candidate. A merge is only ever ACCEPTED when re-resolving the pooled pair's "
        "polygons/segments scores strictly better (via the same "
        "backend.cv_extraction.dxf_extractor._score_region_resolution used for every individual "
        "region) than either region alone, so this can only ever add a candidate, never replace a "
        "working single-region result with a worse one. Bounded to pairs (not larger combinations) "
        "among the top N by score to keep worst-case cost to N*(N-1)/2 extra resolution attempts, "
        "each comparable in cost to a single region's own resolution.",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to this file's repo root."""
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parent.parent
        return repo_root / path

    # --- Municipality-scoped RAG/RASE paths ---
    # Every municipality (BBMP, MCGM, ...) gets its own regulation corpus,
    # vector index, and runtime-rule directory so byelaws never mix across
    # cities and swapping municipalities requires zero code changes.

    def regulations_dir_for(self, municipality: str) -> Path:
        return self.resolve(self.regulations_dir / municipality.upper())

    def vector_store_dir_for(self, municipality: str) -> Path:
        return self.resolve(self.vector_store_dir / municipality.upper())

    def faiss_index_path(self, municipality: str) -> Path:
        return self.vector_store_dir_for(municipality) / "faiss.index"

    def metadata_jsonl_path(self, municipality: str) -> Path:
        return self.vector_store_dir_for(municipality) / "metadata.jsonl"

    def file_hashes_path(self, municipality: str) -> Path:
        return self.vector_store_dir_for(municipality) / "file_hashes.json"

    def runtime_rules_dir_for(self, municipality: str) -> Path:
        return self.resolve(self.runtime_rules_dir / municipality.upper())

    def runtime_rules_path(self, municipality: str) -> Path:
        return self.runtime_rules_dir_for(municipality) / "rules.json"

    def runtime_rule_drafts_path(self, municipality: str) -> Path:
        return self.runtime_rules_dir_for(municipality) / "drafts.json"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Tests can bypass the cache via Settings(...) directly."""
    return Settings()