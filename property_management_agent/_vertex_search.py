"""
Vertex AI Search (Agent Builder) integration — managed RAG over the same
content that lives in email_archive + attachment_extractions. Offered as
an ALTERNATIVE search backend so the user can compare it side-by-side
with the local sqlite-vec implementation.

WHY THIS EXISTS
Gemini's own Drive search is more accurate than our local RAG because
Google does smarter chunking, uses a larger embedding model, and reranks
with a learned model. This wrapper exposes the same managed stack to us
without writing any of that ourselves — we just push documents in and
query through the SDK.

GCP SETUP — REQUIRED ONE-TIME, BY THE USER
1. Enable the Discovery Engine API in the project that owns your service
   account:
       gcloud services enable discoveryengine.googleapis.com \\
           --project=<your-project>
   Or via console:
       https://console.cloud.google.com/apis/library/discoveryengine.googleapis.com

2. Grant the service account the Discovery Engine Editor role:
       gcloud projects add-iam-policy-binding <your-project> \\
           --member="serviceAccount:<sa-email>" \\
           --role="roles/discoveryengine.editor"

Until both are done, vertex_* tools return a clear "not configured"
message and the existing local RAG continues to work untouched.

CONFIG (.env, all optional)
    VERTEX_DATA_STORE_ID   default: property-archive
    VERTEX_LOCATION        default: global
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from ._auth import SERVICE_ACCOUNT_JSON_PATH

logger = logging.getLogger(__name__)

VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global").strip() or "global"
VERTEX_DATA_STORE_ID = (
    os.getenv("VERTEX_DATA_STORE_ID", "property-archive").strip() or "property-archive"
)

# Lazy state
_state: dict[str, Any] = {}


def _setup_help() -> str:
    return (
        "Vertex AI Search is not configured. One-time setup in the GCP project "
        "that owns your service account:\n"
        "  1. Enable the Discovery Engine API:\n"
        "       gcloud services enable discoveryengine.googleapis.com --project=<project>\n"
        "  2. Grant the SA the Discovery Engine Editor role:\n"
        "       gcloud projects add-iam-policy-binding <project> \\\n"
        "         --member=\"serviceAccount:<sa-email>\" \\\n"
        "         --role=\"roles/discoveryengine.editor\"\n"
        "Until then, the local sqlite-vec RAG is still available."
    )


def _get_state() -> Optional[dict]:
    """Build the clients + project ID on first call. Returns None if anything
    needed is missing (SDK not installed, no SA file). The caller surfaces a
    user-facing message instead of crashing."""
    if _state:
        return _state

    try:
        from google.cloud import discoveryengine_v1 as discoveryengine
        from google.api_core.client_options import ClientOptions
        from google.oauth2 import service_account
    except ImportError as e:
        logger.warning("google-cloud-discoveryengine not installed: %s", e)
        return None

    if not os.path.exists(SERVICE_ACCOUNT_JSON_PATH):
        logger.warning("Service-account JSON not at %s", SERVICE_ACCOUNT_JSON_PATH)
        return None

    with open(SERVICE_ACCOUNT_JSON_PATH) as f:
        sa = json.load(f)
    project_id = sa.get("project_id", "")
    if not project_id:
        return None

    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_JSON_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    # Regional endpoints exist for some locations; for "global" the default works.
    opts: Optional[ClientOptions] = None
    if VERTEX_LOCATION and VERTEX_LOCATION != "global":
        opts = ClientOptions(api_endpoint=f"{VERTEX_LOCATION}-discoveryengine.googleapis.com")

    _state.update({
        "module":     discoveryengine,
        "project_id": project_id,
        "creds":      creds,
        "client_opts": opts,
        "ds_client": discoveryengine.DataStoreServiceClient(credentials=creds, client_options=opts),
        "doc_client": discoveryengine.DocumentServiceClient(credentials=creds, client_options=opts),
        "search_client": discoveryengine.SearchServiceClient(credentials=creds, client_options=opts),
        "engine_client": discoveryengine.EngineServiceClient(credentials=creds, client_options=opts)
            if hasattr(discoveryengine, "EngineServiceClient") else None,
    })
    return _state


def _parent_collection(s: dict) -> str:
    return (
        f"projects/{s['project_id']}/locations/{VERTEX_LOCATION}"
        f"/collections/default_collection"
    )


def _data_store_path(s: dict) -> str:
    return f"{_parent_collection(s)}/dataStores/{VERTEX_DATA_STORE_ID}"


def _branch_path(s: dict) -> str:
    return f"{_data_store_path(s)}/branches/default_branch"


def _serving_config_path(s: dict) -> str:
    return f"{_data_store_path(s)}/servingConfigs/default_search"


# ── Setup ─────────────────────────────────────────────────────────────────────

def ensure_data_store() -> dict:
    """Make sure the data store exists. Idempotent — returns immediately if
    it does. Creates a generic content-required data store if not."""
    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}

    de = s["module"]
    ds_client = s["ds_client"]

    # Try to GET the store first; if it doesn't exist, CREATE it.
    try:
        ds_client.get_data_store(name=_data_store_path(s))
        return {"status": "exists", "data_store": VERTEX_DATA_STORE_ID}
    except Exception as e:
        msg = str(e)
        if "has not been used" in msg or "is disabled" in msg:
            return {"isError": True, "needsSetup": True,
                    "message": "Discovery Engine API is not enabled. " + _setup_help()}
        if "PERMISSION_DENIED" in msg.upper():
            return {"isError": True, "needsSetup": True,
                    "message": "Service account lacks roles/discoveryengine.editor. "
                                + _setup_help()}
        # NOT_FOUND → fall through to create
        if "NOT_FOUND" not in msg.upper() and "404" not in msg:
            return {"isError": True, "message": f"Unexpected error: {msg[:300]}"}

    # Create
    try:
        data_store = de.DataStore(
            display_name="Property archive",
            industry_vertical=de.IndustryVertical.GENERIC,
            solution_types=[de.SolutionType.SOLUTION_TYPE_SEARCH],
            content_config=de.DataStore.ContentConfig.CONTENT_REQUIRED,
        )
        op = ds_client.create_data_store(
            parent=_parent_collection(s),
            data_store=data_store,
            data_store_id=VERTEX_DATA_STORE_ID,
        )
        # Wait for the LRO — usually < 30s for a generic store.
        op.result(timeout=180)
        return {"status": "created", "data_store": VERTEX_DATA_STORE_ID}
    except Exception as e:
        return {"isError": True, "message": f"Failed to create data store: {e}"}


# ── Indexing ──────────────────────────────────────────────────────────────────

def _build_doc(s: dict, doc_id: str, text: str, struct: dict) -> Any:
    """Construct a Document protobuf with raw text content."""
    de = s["module"]
    return de.Document(
        id=doc_id,
        content=de.Document.Content(
            mime_type="text/plain",
            raw_bytes=text.encode("utf-8"),
        ),
        struct_data=struct,
    )


def upsert_document(doc_id: str, text: str, struct: Optional[dict] = None) -> dict:
    """Create or replace one document. Idempotent on doc_id."""
    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}

    ensure = ensure_data_store()
    if ensure.get("isError"):
        return ensure

    de = s["module"]
    doc_client = s["doc_client"]
    parent = _branch_path(s)
    doc = _build_doc(s, doc_id, text, struct or {})

    name = f"{parent}/documents/{doc_id}"
    try:
        # Upsert via delete-then-create. Discovery Engine doesn't have a
        # native idempotent "put" for documents; this pattern is the standard
        # workaround and is safe for a single-writer UI flow.
        try:
            doc_client.delete_document(name=name)
        except Exception:
            pass  # didn't exist — that's fine, we'll create below

        created = doc_client.create_document(
            parent=parent,
            document=doc,
            document_id=doc_id,
        )
        return {"status": "indexed", "doc_id": doc_id, "name": created.name}
    except Exception as e:
        msg = str(e)
        if "has not been used" in msg or "is disabled" in msg:
            return {"isError": True, "needsSetup": True,
                    "message": "Discovery Engine API not enabled. " + _setup_help()}
        return {"isError": True, "message": f"Indexing failed: {msg[:400]}"}


def index_email_archive(limit: int = 0) -> dict:
    """Push every row of email_archive (or up to `limit` rows) into Vertex
    AI Search. Idempotent — same doc_id replaces."""
    from .database_agent.db import get_conn

    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}
    ensure = ensure_data_store()
    if ensure.get("isError"):
        return ensure

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT thread_id, subject, snippet, body_text, participants, "
            "web_view_link FROM email_archive"
        ).fetchall()
    if limit and limit > 0:
        rows = rows[:limit]

    indexed = 0
    errors: list[dict] = []
    for r in rows:
        doc_id = f"email_{r['thread_id']}"
        text = (
            f"Subject: {r['subject'] or ''}\n"
            f"Participants: {r['participants'] or ''}\n\n"
            f"{r['body_text'] or ''}"
        )
        struct = {
            "source":         "email_archive",
            "thread_id":      r["thread_id"] or "",
            "subject":        r["subject"] or "",
            "participants":   r["participants"] or "",
            "web_view_link":  r["web_view_link"] or "",
        }
        res = upsert_document(doc_id, text, struct)
        if res.get("isError"):
            errors.append({"doc_id": doc_id, "error": res.get("message", "")[:200]})
            if res.get("needsSetup"):
                # No point continuing — every call will fail with the same setup error
                return {
                    "isError": True, "needsSetup": True,
                    "indexed": indexed, "errors": len(errors) + 1,
                    "message": res["message"],
                }
        else:
            indexed += 1
    return {"source": "email_archive", "indexed": indexed,
            "errors": len(errors), "errors_detail": errors[:10]}


def index_attachment_extractions(limit: int = 0) -> dict:
    """Push every row of attachment_extractions (effective_content) into
    Vertex AI Search. Uses corrected_content where the user has edited it."""
    from .database_agent.db import get_conn

    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}
    ensure = ensure_data_store()
    if ensure.get("isError"):
        return ensure

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT drive_file_id, file_name, mime_type, web_view_link, "
            "content_type, extracted_content, corrected_content "
            "FROM attachment_extractions"
        ).fetchall()
    if limit and limit > 0:
        rows = rows[:limit]

    indexed = 0
    errors: list[dict] = []
    for r in rows:
        doc_id = f"attachment_{r['drive_file_id']}"
        effective = (r["corrected_content"] or r["extracted_content"] or "").strip()
        if not effective:
            continue
        text = (
            f"Filename: {r['file_name'] or ''}\n"
            f"Type: {r['mime_type'] or ''}\n\n"
            f"{effective}"
        )
        struct = {
            "source":         "attachment_extractions",
            "drive_file_id":  r["drive_file_id"] or "",
            "file_name":      r["file_name"] or "",
            "mime_type":      r["mime_type"] or "",
            "web_view_link":  r["web_view_link"] or "",
            "has_correction": bool(r["corrected_content"]),
        }
        res = upsert_document(doc_id, text, struct)
        if res.get("isError"):
            errors.append({"doc_id": doc_id, "error": res.get("message", "")[:200]})
            if res.get("needsSetup"):
                return {
                    "isError": True, "needsSetup": True,
                    "indexed": indexed, "errors": len(errors) + 1,
                    "message": res["message"],
                }
        else:
            indexed += 1
    return {"source": "attachment_extractions", "indexed": indexed,
            "errors": len(errors), "errors_detail": errors[:10]}


def index_everything(limit: int = 0) -> dict:
    """Convenience: index both email_archive and attachment_extractions."""
    r1 = index_email_archive(limit=limit)
    if r1.get("needsSetup"):
        return r1
    r2 = index_attachment_extractions(limit=limit)
    return {
        "email_archive":           r1,
        "attachment_extractions":  r2,
        "note": (
            "Vertex AI Search indexes asynchronously. Newly-pushed documents "
            "typically become searchable within 5–30 minutes."
        ),
    }


# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, limit: int = 10) -> dict:
    """Run a semantic search against Vertex AI Search and return the top
    results in a shape that mirrors the local search response so the UI can
    render them side-by-side.

    Returns:
        {"count": N, "results": [{title, snippet, source, web_view_link,
                                   doc_id, struct, score}, ...]} —
        OR {"isError": ..., "message": ...} if anything goes wrong.
    """
    s = _get_state()
    if s is None:
        return {"isError": True, "needsSetup": True, "message": _setup_help()}
    if not query.strip():
        return {"isError": True, "message": "query is empty"}

    de = s["module"]
    sc = s["search_client"]
    request = de.SearchRequest(
        serving_config=_serving_config_path(s),
        query=query,
        page_size=max(1, min(int(limit) if str(limit).isdigit() else 10, 50)),
        content_search_spec=de.SearchRequest.ContentSearchSpec(
            snippet_spec=de.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
            ),
        ),
    )
    try:
        response = sc.search(request=request)
    except Exception as e:
        msg = str(e)
        if "has not been used" in msg or "is disabled" in msg:
            return {"isError": True, "needsSetup": True,
                    "message": "Discovery Engine API not enabled. " + _setup_help()}
        if "PERMISSION_DENIED" in msg.upper():
            return {"isError": True, "needsSetup": True,
                    "message": "Service account lacks roles/discoveryengine.editor. "
                                + _setup_help()}
        return {"isError": True, "message": f"Vertex search failed: {msg[:400]}"}

    results = []
    for item in response.results:
        doc = item.document
        # Try to extract a snippet
        snippet = ""
        try:
            for d in doc.derived_struct_data.get("snippets", []) or []:
                if isinstance(d, dict) and d.get("snippet"):
                    snippet = d["snippet"]
                    break
        except Exception:
            pass
        struct = {}
        try:
            struct = dict(doc.struct_data) if doc.struct_data else {}
        except Exception:
            pass
        results.append({
            "doc_id":        doc.id,
            "title":         struct.get("subject") or struct.get("file_name") or doc.id,
            "snippet":       snippet,
            "source":        struct.get("source", ""),
            "web_view_link": struct.get("web_view_link", ""),
            "struct":        struct,
        })
    return {"count": len(results), "results": results}


def get_status() -> dict:
    """Returns whether Vertex AI Search is reachable and configured. Used
    by the UI to show a setup banner vs the compare panel."""
    s = _get_state()
    if s is None:
        return {"available": False, "reason": _setup_help()}
    ensure = ensure_data_store()
    if ensure.get("isError"):
        return {"available": False, "reason": ensure.get("message", "")}
    # Try a 1-row probe to test query path too
    try:
        sc = s["search_client"]
        de = s["module"]
        sc.search(request=de.SearchRequest(
            serving_config=_serving_config_path(s),
            query="probe",
            page_size=1,
        ))
        return {"available": True, "data_store": VERTEX_DATA_STORE_ID,
                "project_id": s["project_id"], "location": VERTEX_LOCATION}
    except Exception as e:
        return {"available": False, "reason": str(e)[:300]}
