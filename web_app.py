from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, url_for
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, CLIPProcessor

from src.query_pipeline import process_query_image
from src.similarity_search import load_feature_db, search_cosine, search_euclidean

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_IMAGE_DIR = (PROJECT_ROOT / "data" / "myntradataset" / "images").resolve()
UPLOAD_DIR = PROJECT_ROOT / "web" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CLIP_CACHE_DIR = PROJECT_ROOT / "features" / "clip_vlm_cache"
CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CLIP_INDEX_FILE = CLIP_CACHE_DIR / "clip_index.faiss"
CLIP_META_FILE = CLIP_CACHE_DIR / "clip_index_meta.json"
CAPTION_CACHE_FILE = CLIP_CACHE_DIR / "moondream_caption_cache.json"
CLIP_BUILD_LOCK_FILE = CLIP_CACHE_DIR / "clip_index.build.lock"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "moondream"
OLLAMA_TIMEOUT_SEC = int(os.environ.get("OLLAMA_TIMEOUT_SEC", "120"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")
OLLAMA_OPTIONS = {"temperature": 0, "num_predict": 50}
VLM_LOG_MAX_CHARS = int(os.environ.get("VLM_LOG_MAX_CHARS", "800"))
VLM_MIN_CHARS = int(os.environ.get("VLM_MIN_CHARS", "20"))
VLM_MIN_WORDS = int(os.environ.get("VLM_MIN_WORDS", "4"))
VLM_RETRY_NUM_PREDICT = int(os.environ.get("VLM_RETRY_NUM_PREDICT", "120"))
CLIP_VLM_MAX_IMAGES = 3000
STRICT_USE_CACHED_INDEX = os.environ.get("STRICT_USE_CACHED_INDEX", "0") == "1"
CLIP_FORCE_REBUILD = os.environ.get("CLIP_FORCE_REBUILD", "0") == "1"
CLIP_LOCK_STALE_SEC = int(os.environ.get("CLIP_LOCK_STALE_SEC", "3600"))
VLM_GALLERY_SHORTLIST = max(1, min(int(os.environ.get("VLM_GALLERY_SHORTLIST", "3")), 10))
VLM_RERANK_CANDIDATES = max(
    1,
    min(int(os.environ.get("VLM_RERANK_CANDIDATES", str(VLM_GALLERY_SHORTLIST))), 20),
)
DEFAULT_VLM_PROMPT = os.environ.get(
    "VLM_PROMPT",
    "Identify this object. Is it a real organic item or a manufactured prop/ornament? "
    "Look for artificial textures, seams, or unnatural gloss. Describe its material and nature.",
)

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "web" / "templates"),
    static_folder=str(PROJECT_ROOT / "web" / "static"),
)

FEATURE_DB = {}
DB_ERROR = None

try:
    FEATURE_DB = load_feature_db()
except Exception as exc:
    DB_ERROR = f"Could not load feature database: {exc}"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device: {device}")

print("[INIT] Loading CLIP...")
clip_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_id).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_id)

print("[INIT] Loading SBERT...")
text_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"[INIT] Dataset image directory: {DATASET_IMAGE_DIR}")
print(f"[INIT] CLIP gallery cap: first {CLIP_VLM_MAX_IMAGES} images")
print(f"[INIT] VLM rerank pool: top {VLM_RERANK_CANDIDATES} CLIP candidates")

CLIP_INDEX_DIM = 512
clip_index = faiss.IndexFlatL2(CLIP_INDEX_DIM)
clip_gallery: list[dict] = []
caption_cache: dict[str, str] = {}
_clip_gallery_ready = False
_clip_gallery_lock = threading.Lock()
_caption_cache_lock = threading.Lock()
_caption_inflight_lock = threading.Lock()
_caption_inflight: dict[str, threading.Event] = {}


def _load_caption_cache() -> dict[str, str]:
    if not CAPTION_CACHE_FILE.exists():
        return {}
    try:
        with open(CAPTION_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:
        print(f"[WARN] Could not load caption cache: {exc}")
    return {}


def _save_caption_cache() -> None:
    try:
        tmp_file = CAPTION_CACHE_FILE.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(caption_cache, f, ensure_ascii=True)
        os.replace(tmp_file, CAPTION_CACHE_FILE)
    except Exception as exc:
        print(f"[WARN] Could not save caption cache: {exc}")


caption_cache = _load_caption_cache()


def _is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _validate_path(base_dir: Path, filename: str) -> bool:
    """Prevent path traversal attacks"""
    try:
        resolved = (base_dir / filename).resolve()
        return str(resolved).startswith(str(base_dir.resolve()))
    except (ValueError, RuntimeError):
        return False


def _build_result_rows(results: list[tuple[str, float]], method: str) -> list[dict]:
    metric_label = "Similarity" if method == "cosine" else "Distance"
    rows = []

    for rank, (filename, score) in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "filename": filename,
                "score": float(score),
                "metric_label": metric_label,
                "image_url": url_for("serve_dataset_image", filename=filename),
            }
        )

    return rows


def _print_terminal_report(report: dict):
    print("\n[REQUEST REPORT]")
    print(json.dumps(report, indent=2, ensure_ascii=True))


def _iter_dataset_files(limit: int | None = None) -> list[str]:
    files = []
    for path in DATASET_IMAGE_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        files.append(path.name)
    files.sort()
    if limit is not None:
        return files[:limit]
    return files


def _read_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _clip_features_from_image(image: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    if isinstance(features, torch.Tensor):
        tensor = features
    elif hasattr(features, "pooler_output") and isinstance(features.pooler_output, torch.Tensor):
        tensor = features.pooler_output
    elif hasattr(features, "last_hidden_state") and isinstance(features.last_hidden_state, torch.Tensor):
        # Fallback for output objects that expose only token embeddings.
        tensor = features.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"Unexpected CLIP feature output type: {type(features).__name__}")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    vector = tensor.detach().to(torch.float32).cpu().numpy()
    return vector


def _is_caption_usable(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False

    words = [w for w in normalized.split() if w]
    if len(normalized) < VLM_MIN_CHARS:
        return False
    if len(words) < VLM_MIN_WORDS:
        return False
    if normalized.lower().startswith("unable to infer"):
        return False
    return True


def _log_vlm_text(context: str, attempt: str, text: str) -> None:
    normalized = (text or "").replace("\n", " ").strip()
    if len(normalized) > VLM_LOG_MAX_CHARS:
        preview = normalized[:VLM_LOG_MAX_CHARS] + "..."
    else:
        preview = normalized
    print(
        f"[OLLAMA] response context={context} attempt={attempt} "
        f"text={json.dumps(preview, ensure_ascii=True)}"
    )


def _ollama_generate(
    image: Image.Image,
    prompt: str,
    options: dict | None = None,
    context: str = "image",
) -> str:
    request_options = options or OLLAMA_OPTIONS

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    def _single_call(call_prompt: str, call_options: dict, attempt: str) -> str:
        prompt_hash = hashlib.sha1(call_prompt.encode("utf-8")).hexdigest()[:12]
        start = time.perf_counter()
        print(
            f"[OLLAMA] request model={OLLAMA_MODEL} context={context} "
            f"attempt={attempt} prompt_hash={prompt_hash}"
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": call_prompt,
            "images": [image_b64],
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": call_options,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT_SEC) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "ignore")
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(
                f"[OLLAMA] error context={context} attempt={attempt} status={exc.code} "
                f"elapsed_ms={elapsed_ms:.1f}"
            )
            raise ValueError(f"Ollama request failed: {exc.code} {error_body}") from exc
        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(
                f"[OLLAMA] error context={context} attempt={attempt} status=unreachable "
                f"elapsed_ms={elapsed_ms:.1f}"
            )
            raise ValueError(
                f"Ollama not reachable at {OLLAMA_BASE_URL}. Is it running?"
            ) from exc

        parsed = json.loads(body)
        if "response" not in parsed:
            raise ValueError(f"Unexpected Ollama response: {parsed}")

        response_text = parsed["response"].strip()
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(
            f"[OLLAMA] ok context={context} attempt={attempt} "
            f"elapsed_ms={elapsed_ms:.1f} chars={len(response_text)}"
        )
        _log_vlm_text(context, attempt, response_text)
        return response_text

    primary_text = _single_call(prompt, request_options, attempt="primary")
    if _is_caption_usable(primary_text):
        return primary_text

    print(
        f"[OLLAMA] warn context={context} low_quality_primary chars={len(primary_text)}; "
        "retrying with detail prompt"
    )
    retry_prompt = (
        "Describe this image in 2-3 concise sentences for retrieval. Mention the main object, "
        "material, colors, shape, and whether it appears real-world or decorative/manufactured."
    )
    retry_options = dict(request_options)
    retry_options["num_predict"] = max(
        int(retry_options.get("num_predict", 0)),
        VLM_RETRY_NUM_PREDICT,
    )
    retry_text = _single_call(retry_prompt, retry_options, attempt="retry")
    if _is_caption_usable(retry_text):
        return retry_text

    fallback_text = "Unable to infer a reliable identity description from this image."
    print(
        f"[OLLAMA] warn context={context} low_quality_retry chars={len(retry_text)}; "
        "using fallback text"
    )
    return fallback_text


def _get_caption_for_file(filename: str, prompt: str) -> str:
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    cache_key = f"{OLLAMA_MODEL}:{prompt_hash}:{filename}"

    with _caption_cache_lock:
        if cache_key in caption_cache:
            cached = caption_cache[cache_key]
            if _is_caption_usable(cached):
                print(f"[OLLAMA] cache_hit context=gallery:{filename}")
                return cached
            print(f"[OLLAMA] cache_stale context=gallery:{filename} reason=low_quality")

    owner = False
    event: threading.Event | None = None
    while not owner:
        with _caption_inflight_lock:
            existing_event = _caption_inflight.get(cache_key)
            if existing_event is None:
                event = threading.Event()
                _caption_inflight[cache_key] = event
                owner = True
            else:
                event = existing_event

        if owner:
            break

        print(f"[OLLAMA] wait context=gallery:{filename} reason=inflight_request")
        event.wait(timeout=OLLAMA_TIMEOUT_SEC + 30)
        with _caption_cache_lock:
            if cache_key in caption_cache:
                cached = caption_cache[cache_key]
                if _is_caption_usable(cached):
                    print(f"[OLLAMA] cache_hit_after_wait context=gallery:{filename}")
                    return cached

    try:
        print(f"[OLLAMA] cache_miss context=gallery:{filename}")
        image = _read_image(DATASET_IMAGE_DIR / filename)
        caption = _ollama_generate(image, prompt, context=f"gallery:{filename}")

        with _caption_cache_lock:
            caption_cache[cache_key] = caption
            _save_caption_cache()

        return caption
    finally:
        with _caption_inflight_lock:
            pending = _caption_inflight.pop(cache_key, None)
        if pending is not None:
            pending.set()


def _build_gallery_signature(dataset_files: list[str]) -> str:
    records = []
    for filename in dataset_files:
        path = DATASET_IMAGE_DIR / filename
        stat = path.stat()
        records.append((filename, stat.st_size, int(stat.st_mtime)))

    payload = {
        "dataset_dir": str(DATASET_IMAGE_DIR),
        "model": clip_id,
        "index_dim": CLIP_INDEX_DIM,
        "max_images": CLIP_VLM_MAX_IMAGES,
        "files": records,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _try_load_cached_clip_index(expected_signature: str) -> bool:
    global clip_index

    if not CLIP_INDEX_FILE.exists() or not CLIP_META_FILE.exists():
        return False

    try:
        with open(CLIP_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("signature") != expected_signature:
            return False

        cached_files = meta.get("files", [])
        if not isinstance(cached_files, list) or not cached_files:
            return False

        loaded_index = faiss.read_index(str(CLIP_INDEX_FILE))
        if loaded_index.ntotal != len(cached_files):
            return False

        clip_index = loaded_index
        clip_gallery.clear()
        clip_gallery.extend(
            {"filename": fn, "id": fn} for fn in cached_files
        )
        print(
            f"[INIT] Loaded cached CLIP index with {loaded_index.ntotal} items "
            f"from {CLIP_INDEX_FILE}"
        )
        return True
    except Exception as exc:
        print(f"[WARN] Failed to load cached CLIP index: {exc}")
        return False


def _try_load_cached_clip_index_strict_count(expected_count: int) -> bool:
    """Load cached index without signature check, but require expected image count."""
    global clip_index

    if not CLIP_INDEX_FILE.exists() or not CLIP_META_FILE.exists():
        return False

    try:
        with open(CLIP_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cached_files = meta.get("files", [])
        if not isinstance(cached_files, list) or len(cached_files) != expected_count:
            return False

        loaded_index = faiss.read_index(str(CLIP_INDEX_FILE))
        if loaded_index.ntotal != expected_count:
            return False

        clip_index = loaded_index
        clip_gallery.clear()
        clip_gallery.extend(
            {"filename": fn, "id": fn} for fn in cached_files
        )
        print(
            f"[INIT] Loaded cached CLIP index with {loaded_index.ntotal} items "
            f"from {CLIP_INDEX_FILE}"
        )
        return True
    except Exception as exc:
        print(f"[WARN] Failed to load cached CLIP index: {exc}")
        return False


def _acquire_build_lock_or_wait(expected_count: int, timeout_sec: int = 900) -> bool:
    """Return True if this process acquired build lock; False if another finished and cache loaded."""
    start = time.perf_counter()
    while True:
        try:
            fd = os.open(str(CLIP_BUILD_LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii", "ignore"))
            os.close(fd)
            print(f"[INIT] Acquired CLIP build lock: {CLIP_BUILD_LOCK_FILE}")
            return True
        except FileExistsError:
            # Recover from stale lock files left after interrupted runs.
            try:
                age_sec = time.time() - CLIP_BUILD_LOCK_FILE.stat().st_mtime
                if age_sec > CLIP_LOCK_STALE_SEC:
                    print(
                        f"[WARN] Removing stale CLIP build lock (age={age_sec:.0f}s): "
                        f"{CLIP_BUILD_LOCK_FILE}"
                    )
                    CLIP_BUILD_LOCK_FILE.unlink(missing_ok=True)
                    continue
            except Exception:
                pass

            # Another process is building; wait for cache to appear.
            if _try_load_cached_clip_index_strict_count(expected_count):
                print("[INIT] CLIP build lock held by another process; loaded completed cache")
                return False

            if (time.perf_counter() - start) > timeout_sec:
                raise TimeoutError(
                    "Timed out waiting for CLIP index build lock to release. "
                    "If no build is running, delete features/clip_vlm_cache/clip_index.build.lock"
                )
            time.sleep(0.5)


def _release_build_lock() -> None:
    try:
        if CLIP_BUILD_LOCK_FILE.exists():
            CLIP_BUILD_LOCK_FILE.unlink()
            print(f"[INIT] Released CLIP build lock: {CLIP_BUILD_LOCK_FILE}")
    except Exception as exc:
        print(f"[WARN] Failed to release CLIP build lock: {exc}")


def _save_cached_clip_index(dataset_files: list[str], signature: str) -> None:
    try:
        index_tmp = CLIP_INDEX_FILE.with_suffix(".tmp")
        faiss.write_index(clip_index, str(index_tmp))
        os.replace(index_tmp, CLIP_INDEX_FILE)

        meta = {
            "signature": signature,
            "files": dataset_files,
            "index_dim": CLIP_INDEX_DIM,
            "model": clip_id,
            "max_images": CLIP_VLM_MAX_IMAGES,
        }
        meta_tmp = CLIP_META_FILE.with_suffix(".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True)
        os.replace(meta_tmp, CLIP_META_FILE)
        print(f"[INIT] Saved CLIP index cache to {CLIP_INDEX_FILE}")
    except Exception as exc:
        print(f"[WARN] Failed to save CLIP index cache: {exc}")


def _ensure_clip_gallery() -> None:
    global _clip_gallery_ready
    if _clip_gallery_ready:
        return

    with _clip_gallery_lock:
        if _clip_gallery_ready:
            return

        dataset_files = _iter_dataset_files(limit=CLIP_VLM_MAX_IMAGES)
        if not dataset_files:
            raise ValueError(f"No dataset images found in {DATASET_IMAGE_DIR}")

        expected_count = len(dataset_files)

        # Strict fast path: load already-built cache and never rebuild unless forced.
        if not CLIP_FORCE_REBUILD and _try_load_cached_clip_index_strict_count(expected_count):
            _clip_gallery_ready = True
            return

        if STRICT_USE_CACHED_INDEX and not CLIP_FORCE_REBUILD:
            raise ValueError(
                "Strict cached-index mode is enabled and no matching cached CLIP index was found. "
                f"Expected {expected_count} items at {CLIP_INDEX_FILE}. "
                "Build once intentionally by setting CLIP_FORCE_REBUILD=1 for one run."
            )

        signature = _build_gallery_signature(dataset_files)
        if _try_load_cached_clip_index(signature):
            _clip_gallery_ready = True
            return

        acquired_lock = _acquire_build_lock_or_wait(expected_count)
        if not acquired_lock:
            _clip_gallery_ready = True
            return

        try:
            print(
                f"[INIT] Building CLIP gallery index from {len(dataset_files)} images "
                f"(limit={CLIP_VLM_MAX_IMAGES})"
            )

            vectors = []
            clip_gallery.clear()
            for idx, filename in enumerate(dataset_files, start=1):
                image_path = DATASET_IMAGE_DIR / filename
                try:
                    image = _read_image(image_path)
                    emb = _clip_features_from_image(image)
                    faiss.normalize_L2(emb)
                    vectors.append(emb)
                    clip_gallery.append({
                        "filename": filename,
                        "id": filename,
                    })
                    if idx % 50 == 0 or idx == len(dataset_files):
                        print(f"[INIT] CLIP index progress: {idx}/{len(dataset_files)}")
                except Exception as exc:
                    print(f"[WARN] Skipping {filename}: {exc}")

            if not vectors:
                raise ValueError("Failed to build CLIP gallery; no valid embeddings were produced")

            matrix = np.vstack(vectors).astype(np.float32)
            clip_index.reset()
            clip_index.add(matrix)
            _save_cached_clip_index(dataset_files, signature)
            _clip_gallery_ready = True
            print(f"[INIT] CLIP gallery ready with {len(clip_gallery)} items")
        finally:
            _release_build_lock()


def _search_clip_vlm(
    query_image: Image.Image,
    top_k: int,
    alpha: float,
    prompt: str,
) -> tuple[list[dict], dict]:
    _ensure_clip_gallery()

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    q_emb = _clip_features_from_image(query_image)
    faiss.normalize_L2(q_emb)
    timings["query_clip"] = time.perf_counter() - t0

    # Keep VLM mandatory for rerank set, but control latency by bounding candidate pool.
    search_limit = min(max(top_k, VLM_RERANK_CANDIDATES), len(clip_gallery))
    t1 = time.perf_counter()
    distances, indices = clip_index.search(q_emb, search_limit)
    timings["clip_search"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    query_caption = _ollama_generate(query_image, prompt, context="query")
    q_text_emb = text_model.encode(query_caption, convert_to_tensor=True)
    timings["query_caption"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    semantic_candidates = []
    semantic_limit = len(indices[0])
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx < 0 or idx >= len(clip_gallery):
            continue
        gallery = clip_gallery[int(idx)]
        filename = gallery["filename"]
        visual_score = float(np.clip(1.0 - float(dist) / 2.0, 0.0, 1.0))

        candidate_caption = _get_caption_for_file(filename, prompt)
        m_text_emb = text_model.encode(candidate_caption, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(q_text_emb, m_text_emb).item())
        final_score = (alpha * visual_score) + ((1.0 - alpha) * semantic_score)
        analysis = (
            f"[FUSION SCORE: {final_score:.4f}] "
            f"(CLIP: {visual_score:.4f} | SBERT: {semantic_score:.4f})"
        )
        semantic_candidates.append(
            {
                "rank": rank,
                "id": gallery["id"],
                "filename": filename,
                "image_url": url_for("serve_dataset_image", filename=filename),
                "caption": candidate_caption,
                "visual_score": visual_score,
                "semantic_score": semantic_score,
                "final_score": float(final_score),
                "analysis": analysis,
            }
        )

    timings["semantic_rerank"] = time.perf_counter() - t3

    semantic_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return semantic_candidates[:top_k], {
        "query_caption": query_caption,
        "semantic_shortlist": semantic_limit,
        "semantic_processed": len(semantic_candidates),
        "timings": timings,
    }


def _render_error(
    *,
    message: str,
    method: str,
    top_k: int,
    pipeline: str,
    alpha: float,
):
    return render_template(
        "index.html",
        db_size=len(FEATURE_DB),
        clip_db_size=len(clip_gallery),
        db_error=DB_ERROR,
        error=message,
        method=method,
        top_k=top_k,
        pipeline=pipeline,
        alpha=alpha,
        mode_label=(
            f"{method.capitalize()} similarity"
            if pipeline == "cnn"
            else "CLIP + Moondream fusion"
        ),
    )


@app.route("/")
def index():
    return render_template(
        "index.html",
        db_size=len(FEATURE_DB),
        clip_db_size=len(clip_gallery),
        db_error=DB_ERROR,
        method="cosine",
        top_k=8,
        pipeline="cnn",
        alpha=0.4,
        mode_label="Cosine similarity",
    )


@app.route("/search", methods=["POST"])
def search():
    pipeline = request.form.get("pipeline", "cnn").lower()
    if pipeline not in {"cnn", "clip_vlm"}:
        pipeline = "cnn"

    mode_label = "CLIP + Moondream fusion" if pipeline == "clip_vlm" else "Similarity search"

    method = request.form.get("method", "cosine").lower()
    if method not in {"cosine", "euclidean"}:
        method = "cosine"

    try:
        alpha = float(request.form.get("alpha", "0.4"))
    except ValueError:
        alpha = 0.4
    alpha = float(np.clip(alpha, 0.0, 1.0))
    vlm_prompt = DEFAULT_VLM_PROMPT

    try:
        top_k = int(request.form.get("top_k", "8"))
    except ValueError:
        top_k = 8
    top_k = max(1, min(top_k, 20))

    if DB_ERROR and pipeline == "cnn":
        return _render_error(
            message="Search is unavailable until feature DB is loaded.",
            method=method,
            top_k=top_k,
            pipeline=pipeline,
            alpha=alpha,
        )

    uploaded = request.files.get("query_image")
    if uploaded is None or uploaded.filename == "":
        return _render_error(
            message="Please choose an image to run search.",
            method=method,
            top_k=top_k,
            pipeline=pipeline,
            alpha=alpha,
        )

    if not _is_allowed_file(uploaded.filename):
        return _render_error(
            message="Unsupported file type. Use JPG, JPEG, PNG, BMP, or WEBP.",
            method=method,
            top_k=top_k,
            pipeline=pipeline,
            alpha=alpha,
        )

    # Check file size
    if uploaded.content_length and uploaded.content_length > MAX_FILE_SIZE:
        return _render_error(
            message=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.",
            method=method,
            top_k=top_k,
            pipeline=pipeline,
            alpha=alpha,
        )

    # Validate MIME type
    mime_type = mimetypes.guess_type(uploaded.filename)[0]
    if mime_type not in ALLOWED_MIMETYPES:
        return _render_error(
            message="Invalid file type. Please upload a valid image.",
            method=method,
            top_k=top_k,
            pipeline=pipeline,
            alpha=alpha,
        )

    file_suffix = Path(uploaded.filename).suffix.lower()
    saved_name = f"{uuid.uuid4().hex}{file_suffix}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        uploaded.save(str(saved_path))

        if pipeline == "clip_vlm":
            total_start = time.perf_counter()
            query_image = _read_image(saved_path)
            clip_results, clip_meta = _search_clip_vlm(
                query_image=query_image,
                top_k=top_k,
                alpha=alpha,
                prompt=vlm_prompt,
            )
            total_time = time.perf_counter() - total_start

            report = {
                "pipeline": "clip_vlm",
                "query_file": uploaded.filename,
                "saved_query_file": saved_name,
                "top_k": top_k,
                "alpha": alpha,
                "semantic_shortlist": clip_meta["semantic_shortlist"],
                "semantic_processed": clip_meta["semantic_processed"],
                "prompt": vlm_prompt,
                "timings_ms": {
                    "query_clip": round(clip_meta["timings"]["query_clip"] * 1000, 2),
                    "clip_search": round(clip_meta["timings"]["clip_search"] * 1000, 2),
                    "query_caption": round(clip_meta["timings"]["query_caption"] * 1000, 2),
                    "semantic_rerank": round(clip_meta["timings"]["semantic_rerank"] * 1000, 2),
                    "total": round(total_time * 1000, 2),
                },
                "query_caption": clip_meta["query_caption"],
                "results": [
                    {
                        "rank": idx + 1,
                        "filename": item["filename"],
                        "final_score": round(item["final_score"], 4),
                    }
                    for idx, item in enumerate(clip_results)
                ],
            }
            _print_terminal_report(report)

            return render_template(
                "index.html",
                db_size=len(FEATURE_DB),
                clip_db_size=len(clip_gallery),
                db_error=DB_ERROR,
                method=method,
                top_k=top_k,
                pipeline=pipeline,
                alpha=alpha,
                mode_label=mode_label,
                query_image_url=url_for("serve_query_image", filename=saved_name),
                clip_vlm={
                    "query_caption": clip_meta["query_caption"],
                    "results": clip_results,
                    "alpha": alpha,
                },
                timings={
                    "feature": clip_meta["timings"]["query_clip"],
                    "search": clip_meta["timings"]["clip_search"] + clip_meta["timings"]["semantic_rerank"],
                    "total": total_time,
                },
            )

        total_start = time.perf_counter()

        feature_start = time.perf_counter()
        query_vector = process_query_image(saved_path)
        feature_time = time.perf_counter() - feature_start

        if method == "cosine":
            raw_results, search_time = search_cosine(query_vector, FEATURE_DB, top_k=top_k)
        else:
            raw_results, search_time = search_euclidean(query_vector, FEATURE_DB, top_k=top_k)

        total_time = time.perf_counter() - total_start

        report = {
            "pipeline": "cnn",
            "query_file": uploaded.filename,
            "saved_query_file": saved_name,
            "method": method,
            "top_k": top_k,
            "timings_ms": {
                "feature": round(feature_time * 1000, 2),
                "search": round(search_time * 1000, 2),
                "total": round(total_time * 1000, 2),
            },
            "results": [
                {
                    "rank": idx + 1,
                    "filename": filename,
                    "score": round(float(score), 4),
                }
                for idx, (filename, score) in enumerate(raw_results)
            ],
        }
        _print_terminal_report(report)

        return render_template(
            "index.html",
            db_size=len(FEATURE_DB),
            clip_db_size=len(clip_gallery),
            db_error=DB_ERROR,
            method=method,
            top_k=top_k,
            pipeline=pipeline,
            alpha=alpha,
            mode_label=f"{method.capitalize()} similarity",
            query_image_url=url_for("serve_query_image", filename=saved_name),
            results=_build_result_rows(raw_results, method),
            timings={
                "feature": feature_time,
                "search": search_time,
                "total": total_time,
            },
        )

    except Exception as exc:
        report = {
            "pipeline": pipeline,
            "query_file": uploaded.filename,
            "status": "error",
            "error": str(exc),
        }
        _print_terminal_report(report)
        return _render_error(
            message=f"Search failed: {exc}",
            method=method,
            top_k=top_k,
            pipeline=pipeline,
            alpha=alpha,
        )


@app.route("/dataset-image/<path:filename>")
def serve_dataset_image(filename: str):
    if not _validate_path(DATASET_IMAGE_DIR, filename):
        return "Invalid file path", 400
    return send_from_directory(DATASET_IMAGE_DIR, filename)


@app.route("/query-image/<path:filename>")
def serve_query_image(filename: str):
    if not _validate_path(UPLOAD_DIR, filename):
        return "Invalid file path", 400
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
