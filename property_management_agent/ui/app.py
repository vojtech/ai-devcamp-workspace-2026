"""
Streamlit UI for browsing the property management SQLite database.

Run from the project root:
    streamlit run property_management_agent/ui/app.py

Reads the same property_data.db the ADK agents write to. Read-only — no
edits or deletes — to keep the agent the single source of truth for writes.
"""
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

# ── DB location ─────────────────────────────────────────────────────────────────
# Same default the agents use: property_data.db one level above database_agent/
DEFAULT_DB = (
    Path(__file__).resolve().parent.parent / "property_data.db"
)
DB_PATH = os.getenv("DB_PATH", str(DEFAULT_DB))


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

# ── Summary header ──────────────────────────────────────────────────────────────
def summary_metrics():
    cols = st.columns(6)
    counts = {
        "Employees":        len(load_table("employees", cb)),
        "Managers":         len(load_table("property_managers", cb)),
        "Contacts":         len(load_table("key_contacts", cb)),
        "Tasks":            len(load_table("tasks", cb)),
        "Meetings":         len(load_table("meetings", cb)),
        "Classifications":  len(load_table("email_classifications", cb)),
    }
    for col, (label, n) in zip(cols, counts.items()):
        col.metric(label, n)


summary_metrics()
st.divider()

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab_class, tab_emp, tab_mgr, tab_con, tab_task, tab_meet = st.tabs([
    "📨 Email classifications",
    "👥 Employees",
    "🧑‍💼 Property managers",
    "📞 Key contacts",
    "✅ Tasks",
    "📅 Meetings",
])


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
