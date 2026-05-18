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
    cols = st.columns(7)
    counts = {
        "Employees":        len(load_table("employees", cb)),
        "Managers":         len(load_table("property_managers", cb)),
        "Contacts":         len(load_table("key_contacts", cb)),
        "Tasks":            len(load_table("tasks", cb)),
        "Meetings":         len(load_table("meetings", cb)),
        "Classifications":  len(load_table("email_classifications", cb)),
        "Archive (RAG)":    len(load_table("email_archive", cb)),
    }
    for col, (label, n) in zip(cols, counts.items()):
        col.metric(label, n)


summary_metrics()
st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_search, tab_class, tab_emp, tab_mgr, tab_con, tab_task, tab_meet = st.tabs([
    "🔎 Email archive search",
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
