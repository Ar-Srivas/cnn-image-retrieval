from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
import os
import mimetypes
import urllib.error
import urllib.request

from flask import Flask, render_template, request, send_from_directory, url_for

from src.query_pipeline import process_query_image
from src.similarity_search import load_feature_db, search_cosine, search_euclidean

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_IMAGE_DIR = PROJECT_ROOT / "data" / "myntradataset" / "images"
UPLOAD_DIR = PROJECT_ROOT / "web" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MOONDREAM_BASE_URL = os.environ.get("MOONDREAM_BASE_URL", "http://localhost:8000")
MOONDREAM_CAPTION_ENDPOINT = os.environ.get("MOONDREAM_CAPTION_ENDPOINT", "/caption")
MOONDREAM_CAPTION_URL = f"{MOONDREAM_BASE_URL}{MOONDREAM_CAPTION_ENDPOINT}"

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


def _build_multipart_body(file_path: Path) -> tuple[bytes, str]:
    boundary = f"----MoonBoundary{uuid.uuid4().hex}"
    filename = file_path.name
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    return bytes(body), boundary


def _call_moondream_caption(file_path: Path) -> dict:
    body, boundary = _build_multipart_body(file_path)
    req = urllib.request.Request(
        MOONDREAM_CAPTION_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise ValueError(f"Moondream request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(
            f"Moondream server not reachable at {MOONDREAM_CAPTION_URL}. Is server.py running?"
        ) from exc

    data = json.loads(payload)
    if data.get("status") != "success":
        raise ValueError(data.get("message", "Moondream request failed"))

    return data


@app.route("/")
def index():
    return render_template(
        "index.html",
        db_size=len(FEATURE_DB),
        db_error=DB_ERROR,
        method="cosine",
        top_k=8,
        use_moondream=False,
        mode_label="Similarity search",
    )


@app.route("/search", methods=["POST"])
def search():
    use_moondream = request.form.get("use_moondream") == "on"
    mode_label = "Moondream caption" if use_moondream else "Similarity search"
    method = request.form.get("method", "cosine").lower()
    if method not in {"cosine", "euclidean"}:
        method = "cosine"

    try:
        top_k = int(request.form.get("top_k", "8"))
    except ValueError:
        top_k = 8
    top_k = max(1, min(top_k, 20))

    if DB_ERROR and not use_moondream:
        return render_template(
            "index.html",
            db_size=0,
            db_error=DB_ERROR,
            error="Search is unavailable until feature DB is loaded.",
            method=method,
            top_k=top_k,
            use_moondream=use_moondream,
            mode_label=mode_label,
        )

    uploaded = request.files.get("query_image")
    if uploaded is None or uploaded.filename == "":
        return render_template(
            "index.html",
            db_size=len(FEATURE_DB),
            db_error=DB_ERROR,
            error="Please choose an image to run search.",
            method=method,
            top_k=top_k,
            use_moondream=use_moondream,
            mode_label=mode_label,
        )

    if not _is_allowed_file(uploaded.filename):
        return render_template(
            "index.html",
            db_size=len(FEATURE_DB),
            db_error=DB_ERROR,
            error="Unsupported file type. Use JPG, JPEG, PNG, BMP, or WEBP.",
            method=method,
            top_k=top_k,
            use_moondream=use_moondream,
            mode_label=mode_label,
        )

    # Check file size
    if uploaded.content_length and uploaded.content_length > MAX_FILE_SIZE:
        return render_template(
            "index.html",
            db_size=len(FEATURE_DB),
            db_error=DB_ERROR,
            error=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.",
            method=method,
            top_k=top_k,
            use_moondream=use_moondream,
            mode_label=mode_label,
        )

    # Validate MIME type
    mime_type = mimetypes.guess_type(uploaded.filename)[0]
    if mime_type not in ALLOWED_MIMETYPES:
        return render_template(
            "index.html",
            db_size=len(FEATURE_DB),
            db_error=DB_ERROR,
            error="Invalid file type. Please upload a valid image.",
            method=method,
            top_k=top_k,
            use_moondream=use_moondream,
            mode_label=mode_label,
        )

    file_suffix = Path(uploaded.filename).suffix.lower()
    saved_name = f"{uuid.uuid4().hex}{file_suffix}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        uploaded.save(str(saved_path))

        if use_moondream:
            moondream_start = time.perf_counter()
            moondream_response = _call_moondream_caption(saved_path)
            moondream_time = time.perf_counter() - moondream_start

            return render_template(
                "index.html",
                db_size=len(FEATURE_DB),
                db_error=DB_ERROR,
                method=method,
                top_k=top_k,
                use_moondream=use_moondream,
                mode_label=mode_label,
                query_image_url=url_for("serve_query_image", filename=saved_name),
                moondream={
                    "caption": moondream_response.get("caption", ""),
                    "filename": moondream_response.get("filename", saved_name),
                    "time": moondream_time,
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

        return render_template(
            "index.html",
            db_size=len(FEATURE_DB),
            db_error=DB_ERROR,
            method=method,
            top_k=top_k,
            use_moondream=use_moondream,
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
        error_msg = "Search failed. Please check your image and try again."
        return render_template(
            "index.html",
            db_size=len(FEATURE_DB),
            db_error=DB_ERROR,
            error=error_msg,
            method=method,
            top_k=top_k,
            use_moondream=use_moondream,
            mode_label=mode_label,
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
    app.run(debug=True)
