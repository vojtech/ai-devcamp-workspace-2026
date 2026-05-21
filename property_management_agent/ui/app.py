"""
Streamlit UI for browsing the property management SQLite database AND
running semantic search against the embedded email archive.

Run from the project root:
    streamlit run property_management_agent/ui/app.py

Reads the same property_data.db the ADK agents write to. Read-only on the
table browser side. The Email-archive search tab also calls embed_for_query
(Gemini API) and the sqlite-vec KNN search, but never writes new rows on
its own — ingestion happens via the agent or the cron CLI.
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Bootstrap ─────────────────────────────────────────────────────────────────
# Make the property_management_agent package importable when streamlit is
# invoked from anywhere (e.g. from /Users/.../DevCamp via `streamlit run ...`).
_AGENT_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _AGENT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env so GOOGLE_API_KEY (and DRIVE_FOLDERS, DB_PATH, etc.) reach the
# embedding helper. Streamlit does not auto-load .env.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(_AGENT_DIR / ".env")
except ImportError:
    pass

# ── DB location ─────────────────────────────────────────────────────────────────
# Use the exact same resolution logic the agent's db.py applies, so the UI
# always points at the file the agent is actually writing to — even when the
# user sets DB_PATH (relative or absolute) in .env.
_RAW = os.getenv("DB_PATH") or "property_data.db"
DB_PATH = _RAW if os.path.isabs(_RAW) else os.path.normpath(str(_AGENT_DIR / _RAW))


@st.cache_data(ttl=5)  # refresh every 5s; auto-busts when user clicks Refresh
def load_table(table: str, _cache_buster: int = 0) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except pd.errors.DatabaseError:
            return pd.DataFrame()


def filter_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply a dict of {column: value} filters, skipping empty values."""
    out = df
    for col, val in filters.items():
        if val in ("", None, "All", []):
            continue
        if col not in out.columns:
            continue
        if isinstance(val, list):
            out = out[out[col].isin(val)]
        elif isinstance(val, bool):
            out = out[out[col] == int(val)]
        else:
            out = out[out[col].astype(str).str.contains(str(val), case=False, na=False)]
    return out


def render_classification_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty-print the JSON tags column as a comma-separated string."""
    if "tags" not in df.columns:
        return df
    df = df.copy()
    def _fmt(v):
        if not v:
            return ""
        try:
            arr = json.loads(v)
            return ", ".join(str(t) for t in arr) if isinstance(arr, list) else str(v)
        except (json.JSONDecodeError, ValueError, TypeError):
            return str(v)
    df["tags"] = df["tags"].apply(_fmt)
    return df


@st.cache_data(ttl=30)
def fetch_archive_row(thread_id: str, _cache_buster: int = 0) -> dict | None:
    """Return the full email_archive row (incl. body_text) for a thread_id."""
    if not thread_id or not os.path.exists(DB_PATH):
        return None
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        try:
            r = conn.execute(
                "SELECT * FROM email_archive WHERE thread_id=?", (thread_id,)
            ).fetchone()
            return dict(r) if r else None
        except sqlite3.OperationalError:
            return None


def render_thread_detail(row: dict, key_prefix: str = "") -> None:
    """Render the full message body + metadata for one archive row inside a
    bordered container. Used by both the search results and the
    'All indexed threads' expander."""
    if not row:
        return
    with st.container(border=True):
        # Header block: subject + open-in-Drive button
        head_a, head_b = st.columns([4, 1])
        with head_a:
            st.markdown(f"### {row.get('subject') or '(no subject)'}")
            meta_bits = []
            if row.get("participants"):
                meta_bits.append(f"**Participants:** {row['participants']}")
            if row.get("message_count"):
                meta_bits.append(f"**Messages:** {row['message_count']}")
            if row.get("drive_modified_time"):
                meta_bits.append(f"**Drive modified:** {row['drive_modified_time']}")
            if row.get("ingested_at"):
                meta_bits.append(f"**Ingested:** {row['ingested_at']}")
            if meta_bits:
                st.caption(" · ".join(meta_bits))
        with head_b:
            if row.get("web_view_link"):
                st.link_button("🔗 Open in Drive", row["web_view_link"])

        # Full body. body_text was assembled in _parse_thread_json with one
        # "[date | From: ...]" header per message followed by the body, so
        # we render it with hard line-breaks preserved via st.text.
        body = row.get("body_text", "") or ""
        if body:
            view_mode = st.radio(
                "View",
                options=["📜 Preview (first 500 chars)", "📄 Full message"],
                horizontal=True,
                key=f"{key_prefix}view_mode",
                label_visibility="collapsed",
            )
            if view_mode.startswith("📜"):
                st.text(body[:500] + ("…" if len(body) > 500 else ""))
            else:
                st.text(body)
        else:
            st.info("(no body text stored for this thread)")


# ── Page setup ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Management Browser",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Property Management Data Browser")
st.caption(f"DB: `{DB_PATH}`")

if not os.path.exists(DB_PATH):
    st.error(
        f"Database not found at `{DB_PATH}`.\n\n"
        "Run the ADK agent and ask it to analyse some emails first — "
        "the DB will be created automatically."
    )
    st.stop()

# Refresh button bumps cache key
if "cache_buster" not in st.session_state:
    st.session_state.cache_buster = 0
if st.sidebar.button("🔄 Refresh data"):
    st.session_state.cache_buster += 1
    st.cache_data.clear()

cb = st.session_state.cache_buster

# ── Sidebar: archive sync ──────────────────────────────────────────────────────
# Lets the user trigger an incremental sync from Drive without opening adk web.
# Lazy-imports the heavy agent module so the UI loads instantly otherwise.

with st.sidebar:
    st.divider()
    st.subheader("Email archive")
    if st.button("⬇️  Sync from Drive"):
        with st.spinner("Syncing archive from Drive..."):
            try:
                from property_management_agent.agent import ingest_email_archive_from_drive
                report = json.loads(ingest_email_archive_from_drive())
                st.success(
                    f"Synced — new={report.get('new', 0)}, "
                    f"updated={report.get('updated', 0)}, "
                    f"skipped={report.get('skipped_unchanged', 0)}, "
                    f"errors={report.get('errors', 0)}"
                )
                st.cache_data.clear()
                st.session_state.cache_buster += 1
            except Exception as e:
                st.error(f"Sync failed: {type(e).__name__}: {e}")
    st.caption(
        "Pulls new / updated email-thread JSONs from the Drive folder and "
        "embeds them. Files unchanged since the last sync are skipped."
    )


# ── Summary header ──────────────────────────────────────────────────────────────
def summary_metrics():
    cols = st.columns(8)
    counts = {
        "Employees":        len(load_table("employees", cb)),
        "Managers":         len(load_table("property_managers", cb)),
        "Contacts":         len(load_table("key_contacts", cb)),
        "Tasks":            len(load_table("tasks", cb)),
        "Meetings":         len(load_table("meetings", cb)),
        "Classifications":  len(load_table("email_classifications", cb)),
        "Archive (RAG)":    len(load_table("email_archive", cb)),
        "Attachments":      len(load_table("attachment_extractions", cb)),
    }
    for col, (label, n) in zip(cols, counts.items()):
        col.metric(label, n)


summary_metrics()
st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_search, tab_attach, tab_class, tab_emp, tab_mgr, tab_con, tab_task, tab_meet = st.tabs([
    "🔎 Email archive search",
    "📎 Attachments",
    "📨 Email classifications",
    "👥 Employees",
    "🧑‍💼 Property managers",
    "📞 Key contacts",
    "✅ Tasks",
    "📅 Meetings",
])


# ── Email Archive Search (semantic / RAG) ──────────────────────────────────────
with tab_search:
    st.subheader("Semantic search across the email archive")
    st.caption(
        "Embeds your query with `gemini-embedding-001` and runs KNN against "
        "the local `email_archive_vec` table. Combine with the classification "
        "filters for sharper results (e.g. 'urgent maintenance about heating')."
    )

    archive_df = load_table("email_archive", cb)
    if archive_df.empty:
        st.info(
            "No emails ingested yet. Click **⬇ Sync from Drive** in the "
            "sidebar to backfill the archive, or run "
            "`python3.11 -m property_management_agent.sync_archive`."
        )
    else:
        # Query + filter row
        q = st.text_input(
            "Search",
            placeholder="e.g. boiler problem in winter, lease renewal Mitchell Street, urgent maintenance",
            key="archive_query",
        )
        c1, c2, c3, c4 = st.columns(4)
        class_df = load_table("email_classifications", cb)
        cat_options = ["All"] + (
            sorted(class_df["category"].dropna().unique().tolist())
            if "category" in class_df.columns else []
        )
        category = c1.selectbox("Category", options=cat_options, key="srch_cat")
        urgency = c2.selectbox("Urgency", options=["All", "urgent", "high", "normal", "low"], key="srch_urg")
        action = c3.selectbox("Requires action", options=["All", "Yes", "No"], key="srch_act")
        limit = c4.number_input("Max results", min_value=1, max_value=50, value=10, key="srch_lim")

        cmp_a, cmp_b = st.columns(2)
        with cmp_a:
            compare_vertex = st.checkbox(
                "🆚 Vertex AI Search",
                value=False,
                key="srch_compare_vertex",
                help=(
                    "Runs the same query through Google's managed Vertex AI "
                    "Search data store. Needs one-time GCP setup — "
                    "instructions appear below if not configured yet."
                ),
            )
        with cmp_b:
            compare_gemini = st.checkbox(
                "🆚 Gemini File Search",
                value=False,
                key="srch_compare_gemini",
                help=(
                    "Runs the same query via Gemini's built-in File Search RAG. "
                    "Uses the same GOOGLE_API_KEY — no extra setup."
                ),
            )

        if q.strip():
            try:
                from property_management_agent._embeddings import embed_for_query
                from property_management_agent.database_agent.db import (
                    search_email_archive as _db_search_email_archive,
                )
            except Exception as e:
                st.error(f"Could not load search backend: {e}")
                st.stop()

            with st.spinner("Embedding query and searching..."):
                try:
                    qv = embed_for_query(q)
                except Exception as e:
                    st.error(
                        f"Failed to embed query: {e}\n\n"
                        "Check that GOOGLE_API_KEY is set in "
                        "`property_management_agent/.env`."
                    )
                    st.stop()
                result_raw = _db_search_email_archive(
                    query_embedding=qv,
                    limit=int(limit),
                    category="" if category == "All" else category,
                    urgency="" if urgency == "All" else urgency,
                    requires_action=(
                        "true" if action == "Yes"
                        else "false" if action == "No"
                        else ""
                    ),
                )
            result = json.loads(result_raw)
            if result.get("isError"):
                st.error(result.get("message", "Search failed."))
            else:
                hits = result.get("results", [])
                st.caption(f"**{len(hits)}** match(es). Lower distance = closer match.")
                if hits:
                    res_df = pd.DataFrame(hits)
                    # Render similarity column (1 - distance, clamped 0..1)
                    if "distance" in res_df.columns:
                        res_df["similarity"] = (1.0 - res_df["distance"]).clip(0, 1)
                    # Boolean-ish requires_action → label
                    if "requires_action" in res_df.columns:
                        res_df["requires_action"] = res_df["requires_action"].map(
                            lambda v: "Yes" if v in (1, "1", True, "true") else "No" if v in (0, "0", False, "false") else ""
                        )
                    show_cols = [c for c in (
                        "subject", "similarity", "category", "urgency", "requires_action",
                        "participants", "snippet", "web_view_link", "thread_id",
                    ) if c in res_df.columns]
                    st.caption(
                        "👉 Click any row to read the full message below "
                        "(or click 🔗 to open the original JSON in Drive)."
                    )
                    event = st.dataframe(
                        res_df[show_cols],
                        use_container_width=True,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        key="search_results_table",
                        column_config={
                            "web_view_link": st.column_config.LinkColumn(
                                "Open in Drive", display_text="🔗 open"
                            ),
                            "similarity": st.column_config.ProgressColumn(
                                "similarity", min_value=0.0, max_value=1.0, format="%.2f"
                            ),
                            "snippet":      st.column_config.TextColumn("snippet", width="large"),
                            "participants": st.column_config.TextColumn("participants", width="medium"),
                        },
                    )
                    selected_rows = (
                        event.selection.rows
                        if event and getattr(event, "selection", None)
                        else []
                    )
                    if selected_rows:
                        sel_tid = res_df.iloc[selected_rows[0]].get("thread_id")
                        if sel_tid:
                            full_row = fetch_archive_row(sel_tid, cb)
                            render_thread_detail(full_row or {}, key_prefix="srch_")

            # ── Vertex AI Search side-by-side ──────────────────────────
            if compare_vertex:
                st.divider()
                st.markdown("### 🆚 Vertex AI Search results")
                st.caption(
                    "Same query, Google's managed RAG (chunked + reranked). "
                    "Useful for accuracy comparisons. Indexing is async — "
                    "freshly-pushed docs take 5–30 min to appear."
                )
                try:
                    from property_management_agent import _vertex_search as _vx
                except Exception as e:
                    st.error(f"Could not load Vertex backend: {e}")
                    _vx = None

                if _vx is not None:
                    with st.spinner("Querying Vertex AI Search…"):
                        v_result = _vx.search(q, limit=int(limit))
                    if v_result.get("needsSetup"):
                        st.warning("Vertex AI Search is not configured yet.")
                        with st.expander("📘 One-time setup steps", expanded=True):
                            st.code(v_result.get("message", ""), language="text")
                            st.markdown(
                                "After running both commands, click "
                                "**Index now** below to push your data."
                            )
                    elif v_result.get("isError"):
                        st.error(v_result.get("message", "Vertex query failed."))
                    else:
                        v_hits = v_result.get("results", [])
                        st.caption(f"**{len(v_hits)}** Vertex match(es).")
                        if v_hits:
                            v_df = pd.DataFrame(v_hits)
                            v_show = [c for c in (
                                "title", "snippet", "source", "web_view_link", "doc_id",
                            ) if c in v_df.columns]
                            st.dataframe(
                                v_df[v_show],
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "web_view_link": st.column_config.LinkColumn(
                                        "Open", display_text="🔗"
                                    ),
                                    "snippet": st.column_config.TextColumn("snippet", width="large"),
                                    "title":   st.column_config.TextColumn("title", width="medium"),
                                },
                            )
                        # Status + index button at the bottom of the Vertex block
                        with st.expander("⚙️ Vertex index status / push data", expanded=False):
                            status = _vx.get_status()
                            if status.get("available"):
                                st.success(
                                    f"Connected — project={status.get('project_id')}, "
                                    f"location={status.get('location')}, "
                                    f"data_store={status.get('data_store')}"
                                )
                            else:
                                st.warning(status.get("reason", "Not available."))
                            if st.button("⬆ Index now (push email_archive + attachments)",
                                            key="vx_index_now"):
                                with st.spinner("Pushing documents… (each takes a moment)"):
                                    rep = _vx.index_everything()
                                if rep.get("needsSetup"):
                                    st.error(rep.get("message", "Setup required."))
                                else:
                                    ea = rep.get("email_archive", {})
                                    at = rep.get("attachment_extractions", {})
                                    st.success(
                                        f"Pushed: emails={ea.get('indexed', 0)} "
                                        f"(errors={ea.get('errors', 0)}), "
                                        f"attachments={at.get('indexed', 0)} "
                                        f"(errors={at.get('errors', 0)})"
                                    )
                                    st.info(rep.get("note", ""))

            # ── Gemini File Search side-by-side ────────────────────────
            if compare_gemini:
                st.divider()
                st.markdown("### 🆚 Gemini File Search results")
                st.caption(
                    "Same query, Google's built-in File Search (managed RAG "
                    "inside the Gemini API). Uses the same GOOGLE_API_KEY — "
                    "no extra GCP setup."
                )
                try:
                    from property_management_agent import _gemini_file_search as _gx
                except Exception as e:
                    st.error(f"Could not load Gemini File Search backend: {e}")
                    _gx = None

                if _gx is not None:
                    with st.spinner("Querying Gemini File Search…"):
                        g_result = _gx.search(q, limit=int(limit))
                    if g_result.get("needsSetup"):
                        st.warning("Gemini File Search not configured.")
                        st.code(g_result.get("message", ""), language="text")
                    elif g_result.get("isError"):
                        st.error(g_result.get("message", "Gemini File Search failed."))
                    else:
                        g_answer = g_result.get("answer", "")
                        g_hits = g_result.get("results", [])
                        if g_answer:
                            with st.container(border=True):
                                st.markdown("**📝 Grounded answer**")
                                st.write(g_answer)
                        st.caption(f"**{len(g_hits)}** retrieved chunk(s).")
                        if g_hits:
                            g_df = pd.DataFrame(g_hits)
                            g_show = [c for c in (
                                "title", "snippet", "source", "web_view_link",
                                "page_number", "doc_id",
                            ) if c in g_df.columns]
                            st.dataframe(
                                g_df[g_show],
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "web_view_link": st.column_config.LinkColumn(
                                        "Open", display_text="🔗"
                                    ),
                                    "snippet": st.column_config.TextColumn(
                                        "snippet", width="large"
                                    ),
                                    "title": st.column_config.TextColumn(
                                        "title", width="medium"
                                    ),
                                },
                            )
                        with st.expander(
                            "⚙️ Gemini File Search status / push data",
                            expanded=False,
                        ):
                            gs = _gx.get_status()
                            if gs.get("available"):
                                if gs.get("store_exists"):
                                    st.success(
                                        f"Store: {gs.get('display_name')} · "
                                        f"active={gs.get('active_documents')}, "
                                        f"pending={gs.get('pending_documents')}, "
                                        f"failed={gs.get('failed_documents')}, "
                                        f"size={gs.get('size_bytes')} B"
                                    )
                                else:
                                    st.info(
                                        "Store will be created on first index."
                                    )
                            else:
                                st.warning(gs.get("reason", ""))
                            if st.button(
                                "⬆ Index now (push email_archive + attachments)",
                                key="gx_index_now",
                            ):
                                with st.spinner("Pushing documents…"):
                                    rep = _gx.index_everything()
                                if rep.get("needsSetup"):
                                    st.error(rep.get("message", "Setup required."))
                                else:
                                    ea = rep.get("email_archive", {})
                                    at = rep.get("attachment_extractions", {})
                                    st.success(
                                        f"Pushed: emails={ea.get('indexed', 0)} "
                                        f"(errors={ea.get('errors', 0)}), "
                                        f"attachments={at.get('indexed', 0)} "
                                        f"(errors={at.get('errors', 0)})"
                                    )
                                    st.info(rep.get("note", ""))
        else:
            st.caption(
                f"Type a query above. Currently **{len(archive_df)}** "
                f"email thread(s) are indexed."
            )

        with st.expander(f"📂 All {len(archive_df)} indexed threads", expanded=False):
            st.caption("Click any row to read its full message.")
            # Show a sortable browse view of every ingested thread
            list_cols = [c for c in (
                "subject", "participants", "message_count",
                "ingested_at", "drive_modified_time", "web_view_link", "thread_id",
            ) if c in archive_df.columns]
            browse_df = archive_df[list_cols].sort_values(
                by="ingested_at" if "ingested_at" in archive_df.columns else "subject",
                ascending=False,
            ).reset_index(drop=True)
            event = st.dataframe(
                browse_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="archive_browse_table",
                column_config={
                    "web_view_link": st.column_config.LinkColumn(
                        "Open", display_text="🔗"
                    ),
                },
            )
            sel_rows = (
                event.selection.rows
                if event and getattr(event, "selection", None)
                else []
            )
            if sel_rows:
                sel_tid = browse_df.iloc[sel_rows[0]].get("thread_id")
                if sel_tid:
                    full_row = fetch_archive_row(sel_tid, cb)
                    render_thread_detail(full_row or {}, key_prefix="browse_")


# ── Attachments (extracted content + user corrections) ────────────────────────
with tab_attach:
    st.subheader("Attachments — extracted content + manual corrections")
    st.caption(
        "Browse files extracted via Gemini multimodal. Click a row to view "
        "the full text and **correct anything the model got wrong**. "
        "Corrections are preserved across re-extracts."
    )

    # Action row: extract everything in attachments
    a1, a2, a3 = st.columns([1, 1, 3])
    with a1:
        if st.button("⬇ Extract all", help="Extract every file in the 'attachments' folder. Skips unchanged."):
            with st.spinner("Extracting attachments… (one Gemini call per file)"):
                try:
                    from property_management_agent.drive_agent.agent import (
                        extract_and_save_all_attachments,
                    )
                    rep = json.loads(extract_and_save_all_attachments())
                    st.success(
                        f"Done — extracted={rep.get('extracted', 0)}, "
                        f"skipped={rep.get('skipped_unchanged', 0)}, "
                        f"errors={rep.get('errors', 0)}"
                    )
                    st.cache_data.clear()
                    st.session_state.cache_buster += 1
                except Exception as e:
                    st.error(f"Bulk extract failed: {type(e).__name__}: {e}")
    with a2:
        if st.button("🔁 Force re-extract all", help="Re-extract even files unchanged in Drive (uses Gemini quota)."):
            with st.spinner("Force-re-extracting attachments…"):
                try:
                    from property_management_agent.drive_agent.agent import (
                        extract_and_save_all_attachments,
                    )
                    rep = json.loads(extract_and_save_all_attachments(force=True))
                    st.success(
                        f"Done — extracted={rep.get('extracted', 0)}, "
                        f"errors={rep.get('errors', 0)} (corrections preserved)"
                    )
                    st.cache_data.clear()
                    st.session_state.cache_buster += 1
                except Exception as e:
                    st.error(f"Bulk re-extract failed: {type(e).__name__}: {e}")

    attach_df = load_table("attachment_extractions", cb)
    if attach_df.empty:
        st.info(
            "No attachments extracted yet. Click **⬇ Extract all** above, or "
            "ask the agent: *\"extract and save all attachments\"*."
        )
    else:
        # Filter chips
        c1, c2, c3 = st.columns(3)
        mime_filter = c1.text_input(
            "MIME contains",
            placeholder="pdf, image/, json …",
            key="attach_mime_filter",
        )
        has_corr = c2.selectbox(
            "Correction status",
            options=["All", "Only corrected", "Only uncorrected"],
            key="attach_corr_filter",
        )
        rows_limit = c3.number_input("Max rows", 1, 500, 100, key="attach_limit")

        df = attach_df.copy()
        if mime_filter.strip():
            df = df[df["mime_type"].astype(str).str.contains(
                mime_filter.strip(), case=False, na=False
            )]
        if has_corr == "Only corrected":
            df = df[df["corrected_content"].astype(str).str.strip() != ""]
        elif has_corr == "Only uncorrected":
            df = df[(df["corrected_content"].isna()) | (df["corrected_content"].astype(str).str.strip() == "")]
        df = df.head(int(rows_limit))

        # Render-ready columns
        df["corrected"] = df["corrected_content"].astype(str).str.strip().ne("").map(
            {True: "✏️", False: ""}
        )
        df["preview"] = (
            df["corrected_content"].fillna(df["extracted_content"]).astype(str)
              .str.replace("\n", " ", regex=False).str.slice(0, 200)
        )

        st.caption(f"**{len(df)}** attachment(s) shown. Click a row to view / edit.")
        list_cols = [c for c in (
            "file_name", "mime_type", "content_type", "corrected",
            "size_bytes", "extracted_at", "corrected_at", "preview",
            "web_view_link", "drive_file_id",
        ) if c in df.columns]
        evt = st.dataframe(
            df[list_cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="attach_browse_table",
            column_config={
                "web_view_link": st.column_config.LinkColumn("Open", display_text="🔗"),
                "preview": st.column_config.TextColumn("preview", width="large"),
                "file_name": st.column_config.TextColumn("file_name", width="medium"),
            },
        )

        sel = evt.selection.rows if evt and getattr(evt, "selection", None) else []
        if sel:
            sel_id = df.reset_index(drop=True).iloc[sel[0]]["drive_file_id"]
            # Pull the full row (incl. extracted_content + corrected_content)
            full_row = attach_df[attach_df["drive_file_id"] == sel_id]
            if not full_row.empty:
                row = full_row.iloc[0].to_dict()
                with st.container(border=True):
                    head_a, head_b = st.columns([4, 1])
                    with head_a:
                        st.markdown(f"### {row.get('file_name') or '(no name)'}")
                        meta = []
                        if row.get("mime_type"):    meta.append(f"**Type:** {row['mime_type']}")
                        if row.get("content_type"): meta.append(f"**Path:** {row['content_type']}")
                        if row.get("size_bytes"):   meta.append(f"**Size:** {row['size_bytes']:,} B")
                        if row.get("extracted_at"): meta.append(f"**Extracted:** {row['extracted_at']}")
                        if row.get("corrected_at"): meta.append(f"**Corrected:** {row['corrected_at']}")
                        if meta:
                            st.caption(" · ".join(meta))
                    with head_b:
                        if row.get("web_view_link"):
                            st.link_button("🔗 Open in Drive", row["web_view_link"])

                    extracted = row.get("extracted_content") or ""
                    corrected = row.get("corrected_content") or ""
                    has_correction = bool(corrected.strip())

                    if has_correction:
                        st.success("✏️ This attachment has a user correction stored.")

                    # Two side-by-side panes: model output (RO) | your correction (editable)
                    col_orig, col_edit = st.columns(2)
                    with col_orig:
                        st.markdown("**📄 Model extraction (read-only)**")
                        st.text_area(
                            "Extracted",
                            value=extracted,
                            height=400,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"attach_extracted_{sel_id}",
                        )
                    with col_edit:
                        st.markdown("**✏️ Your correction**")
                        edit_default = corrected if has_correction else extracted
                        new_text = st.text_area(
                            "Correction",
                            value=edit_default,
                            height=400,
                            label_visibility="collapsed",
                            key=f"attach_correction_{sel_id}",
                            help=(
                                "Edit anything the model got wrong. "
                                "Saved text overrides the model output everywhere."
                            ),
                        )

                    btn_a, btn_b, btn_c = st.columns([1, 1, 3])
                    with btn_a:
                        if st.button("💾 Save correction", key=f"save_corr_{sel_id}", type="primary"):
                            try:
                                from property_management_agent.database_agent.db import (
                                    correct_attachment_extraction,
                                )
                                res = json.loads(correct_attachment_extraction(sel_id, new_text))
                                if res.get("isError"):
                                    st.error(res.get("message", "Save failed."))
                                else:
                                    st.success(f"Correction saved ({res.get('status')}).")
                                    st.cache_data.clear()
                                    st.session_state.cache_buster += 1
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Save failed: {type(e).__name__}: {e}")
                    with btn_b:
                        if st.button("↩ Revert to model", key=f"revert_corr_{sel_id}",
                                       disabled=not has_correction,
                                       help="Clears your correction and uses the model's extraction."):
                            try:
                                from property_management_agent.database_agent.db import (
                                    correct_attachment_extraction,
                                )
                                res = json.loads(correct_attachment_extraction(sel_id, ""))
                                if res.get("isError"):
                                    st.error(res.get("message", "Revert failed."))
                                else:
                                    st.success("Correction cleared.")
                                    st.cache_data.clear()
                                    st.session_state.cache_buster += 1
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Revert failed: {type(e).__name__}: {e}")


# ── Email Classifications ──────────────────────────────────────────────────────
with tab_class:
    st.subheader("Email classifications")
    df = load_table("email_classifications", cb)

    if df.empty:
        st.info("No classifications yet. Ask the agent to *analyse emails from <domain>*.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        category = c1.selectbox(
            "Category", options=["All"] + sorted(df["category"].dropna().unique().tolist())
        )
        urgency = c2.selectbox(
            "Urgency", options=["All"] + ["urgent", "high", "normal", "low"]
        )
        sentiment = c3.selectbox(
            "Sentiment", options=["All"] + ["positive", "neutral", "negative"]
        )
        action = c4.selectbox("Requires action", options=["All", "Yes", "No"])
        tag = c5.text_input("Tag contains…", "")

        filters = {
            "category":  category,
            "urgency":   urgency,
            "sentiment": sentiment,
            "requires_action": True if action == "Yes" else False if action == "No" else "All",
            "tags":      tag,
        }
        result = filter_df(df, filters)
        result = render_classification_tags(result)

        st.caption(f"Showing **{len(result)}** of {len(df)} classified emails")

        # Group counts for quick overview
        with st.expander("📊 Breakdown by category / urgency", expanded=False):
            if not result.empty:
                left, right = st.columns(2)
                left.write("**By category**")
                left.bar_chart(result["category"].value_counts())
                right.write("**By urgency**")
                right.bar_chart(result["urgency"].value_counts())

        display_cols = [
            "id", "subject", "sender", "date", "category", "subcategory",
            "tags", "urgency", "sentiment", "requires_action", "summary",
            "thread_id", "classified_at",
        ]
        display_cols = [c for c in display_cols if c in result.columns]
        st.dataframe(result[display_cols], use_container_width=True, hide_index=True)


# ── Employees ──────────────────────────────────────────────────────────────────
with tab_emp:
    st.subheader("Employees")
    df = load_table("employees", cb)
    if df.empty:
        st.info("No employees stored yet.")
    else:
        c1, c2 = st.columns(2)
        search = c1.text_input("Search name / email", "")
        role = c2.selectbox(
            "Role", options=["All"] + sorted(df["role"].dropna().unique().tolist())
        )
        result = filter_df(df, {"name": search, "email": search, "role": role})
        # name/email are OR'd manually
        if search:
            mask = (
                df["name"].astype(str).str.contains(search, case=False, na=False)
                | df["email"].astype(str).str.contains(search, case=False, na=False)
            )
            result = df[mask]
            if role and role != "All":
                result = result[result["role"] == role]
        st.caption(f"{len(result)} of {len(df)}")
        st.dataframe(result, use_container_width=True, hide_index=True)


# ── Property Managers ──────────────────────────────────────────────────────────
with tab_mgr:
    st.subheader("Property managers")
    df = load_table("property_managers", cb)
    if df.empty:
        st.info("No property managers stored yet.")
    else:
        status = st.radio("Status", ["All", "current", "previous"], horizontal=True)
        result = filter_df(df, {"status": status})
        st.dataframe(result, use_container_width=True, hide_index=True)


# ── Key Contacts ───────────────────────────────────────────────────────────────
with tab_con:
    st.subheader("Key contacts")
    df = load_table("key_contacts", cb)
    if df.empty:
        st.info("No key contacts stored yet.")
    else:
        c1, c2 = st.columns(2)
        role = c1.selectbox(
            "Role", options=["All"] + sorted(df["role"].dropna().unique().tolist())
        )
        search = c2.text_input("Search name / email / company", "")
        result = df
        if role and role != "All":
            result = result[result["role"] == role]
        if search:
            mask = False
            for col in ("name", "email", "company"):
                if col in result.columns:
                    mask = mask | result[col].astype(str).str.contains(search, case=False, na=False)
            result = result[mask]
        st.caption(f"{len(result)} of {len(df)}")
        st.dataframe(result, use_container_width=True, hide_index=True)


# ── Tasks ──────────────────────────────────────────────────────────────────────
with tab_task:
    st.subheader("Tasks")
    df = load_table("tasks", cb)
    if df.empty:
        st.info("No tasks stored yet.")
    else:
        c1, c2, c3 = st.columns(3)
        status = c1.selectbox("Status", ["All", "open", "in_progress", "completed"])
        priority = c2.selectbox("Priority", ["All", "high", "normal"])
        search = c3.text_input("Search title / description", "")
        result = filter_df(df, {"status": status, "priority": priority})
        if search:
            mask = (
                result["title"].astype(str).str.contains(search, case=False, na=False)
                | result.get("description", pd.Series([""] * len(result))).astype(str)
                    .str.contains(search, case=False, na=False)
            )
            result = result[mask]
        st.caption(f"{len(result)} of {len(df)}")
        st.dataframe(result, use_container_width=True, hide_index=True)


# ── Meetings ───────────────────────────────────────────────────────────────────
with tab_meet:
    st.subheader("Meetings")
    df = load_table("meetings", cb)
    if df.empty:
        st.info("No meetings stored yet.")
    else:
        search = st.text_input("Search title / location / attendees", "")
        result = df
        if search:
            mask = False
            for col in ("title", "location", "attendees", "agenda"):
                if col in result.columns:
                    mask = mask | result[col].astype(str).str.contains(search, case=False, na=False)
            result = result[mask]
        st.caption(f"{len(result)} of {len(df)}")
        st.dataframe(result, use_container_width=True, hide_index=True)


st.sidebar.divider()
st.sidebar.caption(
    "Read-only browser. The ADK agent (`adk web`) writes to this DB; refresh "
    "after the agent finishes analysing emails."
)
