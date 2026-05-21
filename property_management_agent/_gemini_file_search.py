"""
Gemini File Search backend — managed RAG built into the google-genai SDK.

Compared to the local sqlite-vec engine: Google does the chunking, embedding,
retrieval, and grounding. Compared to Vertex AI Search: no separate API to
enable, no IAM role to grant — uses the SAME GOOGLE_API_KEY (Tier 1 Postpay
so inputs aren't training data) we already use for embeddings and multimodal
extraction.

USAGE
- ensure_store()                — idempotent create of a FileSearchStore
- index_email_archive()         — push every row of email_archive
- index_attachment_extractions()— push effective_content per attachment
- index_everything()            — both
- search(query, limit)          — grounded query; returns top chunks
- get_status()                  — UI/CLI probe

IDEMPOTENCE
File Search has no native "upsert" — each upload creates a new Document.
To stay idempotent we set a `source_id` in `custom_metadata` per document
("email_<thread_id>" or "attachment_<drive_file_id>") and, before each
upload, list-and-delete any existing docs with the same source_id.

CONFIG (.env)
- GEMINI_FILE_SEARCH_STORE_NAME   default: "property-archive"
- GEMINI_EMBEDDING_MODEL          default: "gemini-embedding-001" (auto if unset)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)

STORE_DISPLAY_NAME = (
    os.getenv("GEMINI_FILE_SEARCH_STORE_NAME", "property-archive").strip()
    or "property-archive"
)
# Default model is auto-selected by Google if empty.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip() or None

_state: dict[str, Any] = {}


def _setup_help() -> str:
    return (
        "Gemini File Search needs GOOGLE_API_KEY set in "
        "property_management_agent/.env. Same key the rest of the agent "
        "uses for embeddings and multimodal extraction. No other setup."
    )


def _get_state() -> Optional[dict]:
    if _state:
        return _state
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError as e:
        logger.warning("google-genai not installed: %s", e)
        return None
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    _state["client"] = genai.Client(api_key=api_key)
    _state["types"] = genai_types
    return _state


def _find_existing_store(client, types_mod) -> Optional[Any]:
    """Return the FileSearchStore with our display name, or None."""
    try:
        for store in client.file_search_stores.list():
            if (store.display_name or "") == STORE_DISPLAY_NAME:
                return store
    except Exception as e:
        logger.warning("Could not list file search stores: %s", e)
    return None


def ensure_store() -> dict:
    """Idempotently create / get the file search store. Returns its name."""
    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}
    client = s["client"]
    types_mod = s["types"]

    existing = _find_existing_store(client, types_mod)
    if existing is not None:
        s["store_name"] = existing.name
        return {"status": "exists", "store_name": existing.name,
                "display_name": existing.display_name}

    try:
        cfg_kwargs = {"display_name": STORE_DISPLAY_NAME}
        if EMBEDDING_MODEL:
            cfg_kwargs["embedding_model"] = EMBEDDING_MODEL
        created = client.file_search_stores.create(
            config=types_mod.CreateFileSearchStoreConfig(**cfg_kwargs)
        )
        s["store_name"] = created.name
        return {"status": "created", "store_name": created.name,
                "display_name": created.display_name}
    except Exception as e:
        return {"isError": True, "message": f"Failed to create File Search store: {e}"}


def _get_store_name() -> tuple[Optional[str], Optional[dict]]:
    """Return (store_name, error_dict). On error, error_dict is set."""
    s = _get_state()
    if s is None:
        return None, {"isError": True, "needsSetup": True, "message": _setup_help()}
    if "store_name" in s:
        return s["store_name"], None
    res = ensure_store()
    if res.get("isError"):
        return None, res
    return s["store_name"], None


# ── Indexing ──────────────────────────────────────────────────────────────────

def _meta_list_to_dict(meta_list) -> dict:
    """Flatten a list[CustomMetadata] back into a plain dict for callers."""
    out: dict = {}
    for m in (meta_list or []):
        key = getattr(m, "key", None) or (m.get("key") if isinstance(m, dict) else None)
        if not key:
            continue
        v = (
            getattr(m, "string_value", None)
            or getattr(m, "numeric_value", None)
            or getattr(m, "string_list_value", None)
        )
        if v is None and isinstance(m, dict):
            v = m.get("string_value") or m.get("numeric_value") or m.get("string_list_value")
        out[key] = v
    return out


def _dict_to_meta_list(types_mod, d: dict) -> list:
    """Build the SDK's list[CustomMetadata] from a flat dict."""
    out = []
    for k, v in d.items():
        if v is None:
            continue
        kwargs = {"key": str(k)}
        if isinstance(v, bool):
            # Booleans serialize as "true"/"false" strings (no native bool type)
            kwargs["string_value"] = "true" if v else "false"
        elif isinstance(v, (int, float)):
            kwargs["numeric_value"] = float(v)
        elif isinstance(v, (list, tuple)):
            kwargs["string_list_value"] = [str(x) for x in v]
        else:
            sv = str(v)
            if sv:
                kwargs["string_value"] = sv
            else:
                continue
        out.append(types_mod.CustomMetadata(**kwargs))
    return out


def _delete_docs_with_source_id(client, store_name: str, source_id: str) -> int:
    """Delete every existing Document in the store whose custom_metadata
    source_id matches. Used for idempotent re-uploads."""
    n = 0
    try:
        for doc in client.file_search_stores.documents.list(parent=store_name):
            meta = _meta_list_to_dict(doc.custom_metadata)
            if meta.get("source_id") == source_id:
                try:
                    client.file_search_stores.documents.delete(name=doc.name)
                    n += 1
                except Exception as e:
                    logger.warning("Could not delete existing doc %s: %s", doc.name, e)
    except Exception as e:
        logger.warning("Could not list docs in %s: %s", store_name, e)
    return n


def upsert_document(
    source_id: str,
    text: str,
    display_name: str = "",
    extra_metadata: Optional[dict] = None,
) -> dict:
    """Upload a document, replacing any previous version with the same source_id.
    Idempotent — re-uploads delete the old doc first."""
    if not text or not text.strip():
        return {"isError": True, "message": "empty text"}

    store_name, err = _get_store_name()
    if err:
        return err
    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}
    client = s["client"]
    types_mod = s["types"]

    _delete_docs_with_source_id(client, store_name, source_id)

    flat_meta: dict = {"source_id": source_id}
    if extra_metadata:
        flat_meta.update(extra_metadata)

    # Upload via a temporary file (File Search upload API takes a path)
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        op = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=tmp_path,
            config=types_mod.UploadToFileSearchStoreConfig(
                display_name=display_name or source_id,
                mime_type="text/plain",
                custom_metadata=_dict_to_meta_list(types_mod, flat_meta),
            ),
        )
        # Upload returns a long-running operation; wait briefly so the doc is
        # indexed by the time the caller asserts. Most uploads complete in
        # under 10s; we cap at 120s to avoid hanging the CLI on a giant file.
        try:
            op.result(timeout=120)
        except Exception:
            # Even if we time out waiting, the upload usually succeeds —
            # log and continue. The caller can check via get_status() later.
            logger.info("Upload op timeout (will finish async): %s", source_id)
        return {"status": "uploaded", "source_id": source_id}
    except Exception as e:
        return {"isError": True, "message": f"Upload failed: {e}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def index_email_archive(limit: int = 0) -> dict:
    from .database_agent.db import get_conn

    store_name, err = _get_store_name()
    if err:
        return err

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT thread_id, subject, snippet, body_text, participants, "
            "web_view_link FROM email_archive"
        ).fetchall()
    if limit and limit > 0:
        rows = rows[:limit]

    indexed, errors = 0, []
    for r in rows:
        source_id = f"email_{r['thread_id']}"
        text = (
            f"Subject: {r['subject'] or ''}\n"
            f"Participants: {r['participants'] or ''}\n\n"
            f"{r['body_text'] or ''}"
        )
        res = upsert_document(
            source_id=source_id,
            text=text,
            display_name=(r["subject"] or r["thread_id"])[:120],
            extra_metadata={
                "source":        "email_archive",
                "thread_id":     r["thread_id"] or "",
                "subject":       r["subject"] or "",
                "web_view_link": r["web_view_link"] or "",
            },
        )
        if res.get("isError"):
            errors.append({"source_id": source_id, "error": res.get("message", "")[:200]})
        else:
            indexed += 1

    return {"source": "email_archive", "indexed": indexed,
            "errors": len(errors), "errors_detail": errors[:10]}


def index_attachment_extractions(limit: int = 0) -> dict:
    from .database_agent.db import get_conn

    store_name, err = _get_store_name()
    if err:
        return err

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT drive_file_id, file_name, mime_type, web_view_link, "
            "content_type, extracted_content, corrected_content "
            "FROM attachment_extractions"
        ).fetchall()
    if limit and limit > 0:
        rows = rows[:limit]

    indexed, errors = 0, []
    for r in rows:
        effective = (r["corrected_content"] or r["extracted_content"] or "").strip()
        if not effective:
            continue
        source_id = f"attachment_{r['drive_file_id']}"
        text = (
            f"Filename: {r['file_name'] or ''}\n"
            f"Type: {r['mime_type'] or ''}\n\n"
            f"{effective}"
        )
        res = upsert_document(
            source_id=source_id,
            text=text,
            display_name=(r["file_name"] or r["drive_file_id"])[:120],
            extra_metadata={
                "source":         "attachment_extractions",
                "drive_file_id":  r["drive_file_id"] or "",
                "file_name":      r["file_name"] or "",
                "mime_type":      r["mime_type"] or "",
                "web_view_link":  r["web_view_link"] or "",
                "has_correction": bool(r["corrected_content"]),
            },
        )
        if res.get("isError"):
            errors.append({"source_id": source_id, "error": res.get("message", "")[:200]})
        else:
            indexed += 1

    return {"source": "attachment_extractions", "indexed": indexed,
            "errors": len(errors), "errors_detail": errors[:10]}


def index_everything(limit: int = 0) -> dict:
    r1 = index_email_archive(limit=limit)
    if r1.get("needsSetup"):
        return r1
    r2 = index_attachment_extractions(limit=limit)
    return {
        "email_archive": r1,
        "attachment_extractions": r2,
        "note": "File Search indexing is typically fast (seconds-to-minutes).",
    }


# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, limit: int = 10) -> dict:
    """Run a grounded Gemini query against the File Search store. Returns
    matching chunks parsed from grounding_metadata so the UI can render them
    in the same shape as the local + Vertex backends, plus the model's
    synthesized answer for bonus context."""
    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}
    if not query.strip():
        return {"isError": True, "message": "query is empty"}

    store_name, err = _get_store_name()
    if err:
        return err

    client = s["client"]
    types_mod = s["types"]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                f"Find documents most relevant to this question. Return a "
                f"concise answer grounded in the retrieved documents, citing "
                f"every source. Question: {query}"
            ),
            config=types_mod.GenerateContentConfig(
                tools=[types_mod.Tool(
                    file_search=types_mod.FileSearch(
                        file_search_store_names=[store_name],
                        top_k=max(1, min(int(limit) if str(limit).isdigit() else 10, 50)),
                    )
                )],
            ),
        )
    except Exception as e:
        return {"isError": True, "message": f"Gemini File Search query failed: {e}"}

    # Parse grounding chunks into a uniform results list
    answer = (response.text or "").strip() if hasattr(response, "text") else ""
    results: list[dict] = []
    try:
        for cand in (response.candidates or []):
            gm = getattr(cand, "grounding_metadata", None)
            if not gm:
                continue
            for chunk in (gm.grounding_chunks or []):
                rc = getattr(chunk, "retrieved_context", None)
                if not rc:
                    continue
                meta = _meta_list_to_dict(rc.custom_metadata)
                results.append({
                    "doc_id":        rc.document_name or meta.get("source_id", ""),
                    "title":         rc.title or meta.get("subject") or meta.get("file_name") or "",
                    "snippet":       rc.text or "",
                    "source":        meta.get("source", ""),
                    "web_view_link": meta.get("web_view_link", "") or rc.uri or "",
                    "page_number":   rc.page_number,
                    "struct":        meta,
                })
    except Exception as e:
        logger.warning("Could not parse grounding metadata: %s", e)

    return {"count": len(results), "answer": answer, "results": results}


def get_status() -> dict:
    """Probe for the UI. Returns availability + store stats."""
    s = _get_state()
    if s is None:
        return {"available": False, "reason": _setup_help()}
    client = s["client"]

    try:
        existing = _find_existing_store(client, s["types"])
        if existing is None:
            return {
                "available": True,
                "store_exists": False,
                "display_name": STORE_DISPLAY_NAME,
                "message": "API reachable; store will be created on first index/search.",
            }
        return {
            "available":         True,
            "store_exists":      True,
            "store_name":        existing.name,
            "display_name":      existing.display_name,
            "active_documents":  getattr(existing, "active_documents_count", 0),
            "pending_documents": getattr(existing, "pending_documents_count", 0),
            "failed_documents":  getattr(existing, "failed_documents_count", 0),
            "size_bytes":        getattr(existing, "size_bytes", 0),
            "embedding_model":   getattr(existing, "embedding_model", ""),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)[:300]}
