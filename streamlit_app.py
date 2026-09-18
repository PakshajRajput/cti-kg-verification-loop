"""
CTI Extraction Pipeline — Production Dashboard.

A professional, white-background Streamlit UI with side-by-side comparison
of Baseline vs. Verification Loop results, interactive graphs, and
detailed analytics.
"""

import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import time
import math

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="CTI Knowledge Graph — Extraction Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── Google Font + CSS ───────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* ── Reset & Base ─────────────────────────────────────── */
    .stApp {
        background-color: #FFFFFF;
        color: #1E293B;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    /* ── Header ───────────────────────────────────────────── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #64748B;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 4px;
        margin-bottom: 24px;
    }
    /* ── Cards ─────────────────────────────────────────────── */
    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        transition: box-shadow 0.2s ease;
    }
    .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .card-header {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    /* ── Metric Tiles ────────────────────────────────────── */
    .metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
    .metric-tile {
        flex: 1;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
    }
    .metric-tile.green  { border-left: 4px solid #10B981; }
    .metric-tile.amber  { border-left: 4px solid #F59E0B; }
    .metric-tile.red    { border-left: 4px solid #EF4444; }
    .metric-tile.blue   { border-left: 4px solid #3B82F6; }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-top: 4px;
    }
    /* ── Badges ────────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 9999px;
        color: #fff;
        margin: 0 2px;
        vertical-align: middle;
        letter-spacing: 0.02em;
    }
    .b-threatactor, .b-threat_actor { background: #DC2626; }
    .b-malware       { background: #9333EA; }
    .b-tool          { background: #2563EB; }
    .b-vulnerability  { background: #EA580C; }
    .b-technique, .b-attack_pattern, .b-attackpattern { background: #16A34A; }
    .b-tactic         { background: #0D9488; }
    .b-target         { background: #0891B2; }
    .b-other          { background: #64748B; }

    .verdict-badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.72rem;
        font-weight: 700;
        border-radius: 9999px;
        color: #fff;
        vertical-align: middle;
    }
    .v-valid       { background: #10B981; }
    .v-corrected   { background: #F59E0B; }
    .v-hallucinated { background: #EF4444; }
    /* ── Triplet Row ──────────────────────────────────────── */
    .triplet-row {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.88rem;
    }
    .triplet-row:nth-child(odd) { background: #F8FAFC; }
    .relation-arrow {
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.85rem;
    }
    /* ── Comparison Table ────────────────────────────────── */
    .comparison-header {
        display: flex;
        gap: 16px;
        margin-bottom: 8px;
    }
    .comparison-col {
        flex: 1;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
    }
    /* ── Status Dot ───────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .dot-green { background: #10B981; }
    .dot-red   { background: #EF4444; }
    /* ── Section Divider ──────────────────────────────────── */
    .section-divider {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 32px 0;
    }
    /* ── Override Streamlit metric widget ──────────────────── */
    [data-testid="stMetricValue"] { font-weight: 800; }
    /* ── Sidebar metric ──────────────────────────────────── */
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #E2E8F0;
        font-size: 0.88rem;
    }
    .sidebar-stat-label { color: #64748B; font-weight: 500; }
    .sidebar-stat-value { color: #0F172A; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── Helpers ─────────────────────────────────────────────

def _badge(entity_type: str, text: str) -> str:
    """Return an entity-type badge."""
    known = {"threatactor", "threat_actor", "malware", "tool",
             "vulnerability", "technique", "attack_pattern",
             "attackpattern", "target", "tactic"}
    cls = entity_type.lower() if entity_type.lower() in known else "other"
    return f'<span class="badge b-{cls}">{text}</span>'


def _verdict_badge(verdict: str) -> str:
    """Return a colour-coded verdict badge."""
    return f'<span class="verdict-badge v-{verdict}">{verdict.upper()}</span>'


def _triplet_html(t: dict, show_verdict: str | None = None) -> str:
    """Render one triplet as a styled row."""
    e1 = _badge(t["entity1_type"], t["entity1"])
    e2 = _badge(t["entity2_type"], t["entity2"])
    prefix = f'{_verdict_badge(show_verdict)} ' if show_verdict else ""
    return (
        f'<div class="triplet-row">'
        f'{prefix}{e1} <span class="relation-arrow">➜ {t["relation"]} ➜</span> {e2}'
        f'</div>'
    )


def _metric_tile(value, label, colour="blue"):
    return (
        f'<div class="metric-tile {colour}">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )


# ─────────────────────── Sidebar ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ CTI Pipeline")
    st.caption("Verification-in-the-Loop")
    st.markdown("---")

    # System health
    st.markdown("#### System Status")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        api_ok = h.get("status") == "ready"
        rag_ok = h.get("vector_store_populated", False)
        graph_ok = h.get("neo4j_connected", False)
    except Exception:
        api_ok = rag_ok = graph_ok = False

    for label, ok in [("API Server", api_ok), ("RAG Index", rag_ok), ("Knowledge Graph", graph_ok)]:
        dot = "dot-green" if ok else "dot-red"
        txt = "Online" if ok else "Offline"
        st.markdown(f'<span class="status-dot {dot}"></span> **{label}:** {txt}', unsafe_allow_html=True)

    st.markdown("---")

    # Graph stats
    st.markdown("#### Graph Metrics")
    try:
        s = requests.get(f"{API_URL}/graph/stats", timeout=3).json()
        nodes = s.get("total_nodes", 0)
        edges = s.get("total_edges", 0)
    except Exception:
        nodes = edges = 0

    st.markdown(
        f'<div class="sidebar-stat"><span class="sidebar-stat-label">Nodes</span>'
        f'<span class="sidebar-stat-value">{nodes}</span></div>'
        f'<div class="sidebar-stat"><span class="sidebar-stat-label">Edges</span>'
        f'<span class="sidebar-stat-value">{edges}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Legend
    st.markdown("#### Entity Legend")
    legend_items = [
        ("DC2626", "Threat Actor"), ("9333EA", "Malware"), ("2563EB", "Tool"),
        ("EA580C", "Vulnerability"), ("16A34A", "Technique"), ("0891B2", "Target"),
        ("64748B", "Other"),
    ]
    for hex_col, name in legend_items:
        st.markdown(
            f'<span class="badge" style="background:#{hex_col};">{name}</span>',
            unsafe_allow_html=True,
        )


# ─────────────────────── Header ──────────────────────────────────────────────

st.markdown('<div class="hero-title">🛡️ Verification-in-the-Loop CTI Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Extract structured knowledge from unstructured Cyber Threat Intelligence reports — '
            'then verify every claim against MITRE ATT&CK before it enters your knowledge graph.</div>', unsafe_allow_html=True)


# ─────────────────────── Input Section ───────────────────────────────────────

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">📄  Input CTI Report</div>', unsafe_allow_html=True)

text_input = st.text_area(
    "Paste unstructured CTI text:",
    height=160,
    placeholder=(
        "Example: In a recent advisory, CISA warned that a state-sponsored "
        "threat actor known as Volt Typhoon has been targeting US critical "
        "infrastructure organizations using living-off-the-land techniques…"
    ),
    label_visibility="collapsed",
)

col_a, col_b, col_c = st.columns([1, 1, 2])
run_compare = col_a.button("⚡ Compare Both", type="primary", use_container_width=True)
run_full = col_b.button("🔒 Verification Loop Only", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────── Pipeline Execution ──────────────────────────────────

def _call_pipeline(endpoint: str, text: str) -> dict | None:
    """POST to the API and return JSON, or None on failure."""
    try:
        res = requests.post(
            f"{API_URL}{endpoint}",
            json={"text": text, "source": "streamlit"},
            timeout=300,
        )
        if res.status_code == 200:
            data = res.json()
            if data.get("error"):
                st.error(f"Pipeline Error: {data['error']}")
                return None
            return data
        else:
            st.error(f"HTTP {res.status_code}: {res.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


if run_compare or run_full:
    if not text_input.strip():
        st.warning("Please paste a CTI report above before running extraction.")
    else:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Run pipelines ────────────────────────────────────
        baseline_data = None
        verified_data = None

        if run_compare:
            with st.spinner("Running **Baseline** extraction…"):
                t0 = time.time()
                baseline_data = _call_pipeline("/extract/baseline", text_input)
                baseline_time = time.time() - t0

            with st.spinner("Running **Verification Loop** extraction…"):
                t0 = time.time()
                verified_data = _call_pipeline("/extract", text_input)
                verified_time = time.time() - t0
        else:
            with st.spinner("Running **Verification Loop** extraction…"):
                t0 = time.time()
                verified_data = _call_pipeline("/extract", text_input)
                verified_time = time.time() - t0

        # ── Summary Metrics ──────────────────────────────────
        if verified_data:
            decisions = verified_data.get("verification_decisions", [])
            n_valid = len([d for d in decisions if d["verdict"] == "valid"])
            n_corrected = len([d for d in decisions if d["verdict"] == "corrected"])
            n_hallucinated = len([d for d in decisions if d["verdict"] == "hallucinated"])
            n_total = len(decisions)
            hall_pct = f"{(n_hallucinated / n_total * 100):.0f}%" if n_total else "0%"

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📊  Verification Summary</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="metric-row">'
                + _metric_tile(n_total, "Total Extracted", "blue")
                + _metric_tile(n_valid, "Valid ✓", "green")
                + _metric_tile(n_corrected, "Corrected ✎", "amber")
                + _metric_tile(n_hallucinated, "Hallucinated ✗", "red")
                + _metric_tile(hall_pct, "Hallucination %", "red")
                + '</div>',
                unsafe_allow_html=True,
            )
            if run_compare and baseline_data:
                base_count = len(baseline_data.get("final_triplets", []))
                ver_count = len(verified_data.get("final_triplets", []))
                st.markdown(
                    f'<div class="metric-row">'
                    f'{_metric_tile(base_count, "Baseline Triplets", "blue")}'
                    f'{_metric_tile(ver_count, "Verified Triplets", "green")}'
                    f'{_metric_tile(f"{baseline_time:.1f}s", "Baseline Time", "blue")}'
                    f'{_metric_tile(f"{verified_time:.1f}s", "Verified Time", "green")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Side-by-Side Comparison ──────────────────────────
        if run_compare and baseline_data and verified_data:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">🔀  Side-by-Side Comparison</div>', unsafe_allow_html=True)

            col_base, col_sep, col_ver = st.columns([5, 0.2, 5])

            with col_base:
                st.markdown("##### 🟦 Baseline (No Verification)")
                st.caption(f"All {len(baseline_data.get('final_triplets', []))} triplets accepted blindly — no fact-checking.")
                for t in baseline_data.get("final_triplets", []):
                    st.markdown(_triplet_html(t), unsafe_allow_html=True)

            with col_sep:
                st.markdown(
                    "<div style='width:1px; background:#E2E8F0; height:100%; min-height:300px; margin:0 auto;'></div>",
                    unsafe_allow_html=True,
                )

            with col_ver:
                st.markdown("##### 🟩 Verification Loop")
                st.caption(f"{len(verified_data.get('final_triplets', []))} triplets survived — "
                           f"{n_hallucinated} hallucination(s) caught.")
                for d in verified_data.get("verification_decisions", []):
                    orig = d["original"]
                    verdict = d["verdict"]
                    if verdict == "hallucinated":
                        # Show the rejected triplet struck-through
                        e1 = _badge(orig["entity1_type"], orig["entity1"])
                        e2 = _badge(orig["entity2_type"], orig["entity2"])
                        st.markdown(
                            f'<div class="triplet-row" style="opacity:0.5; text-decoration: line-through;">'
                            f'{_verdict_badge(verdict)} {e1} '
                            f'<span class="relation-arrow">➜ {orig["relation"]} ➜</span> {e2}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    elif verdict == "corrected" and d.get("corrected"):
                        c = d["corrected"]
                        st.markdown(_triplet_html(c, show_verdict=verdict), unsafe_allow_html=True)
                    else:
                        st.markdown(_triplet_html(orig, show_verdict=verdict), unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ── Verified Triplets (standalone mode) ──────────────
        elif verified_data:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">✅  Verified Triplets</div>', unsafe_allow_html=True)
            for d in verified_data.get("verification_decisions", []):
                orig = d["original"]
                verdict = d["verdict"]
                if verdict == "corrected" and d.get("corrected"):
                    st.markdown(_triplet_html(d["corrected"], show_verdict=verdict), unsafe_allow_html=True)
                elif verdict == "hallucinated":
                    e1 = _badge(orig["entity1_type"], orig["entity1"])
                    e2 = _badge(orig["entity2_type"], orig["entity2"])
                    st.markdown(
                        f'<div class="triplet-row" style="opacity:0.5; text-decoration: line-through;">'
                        f'{_verdict_badge(verdict)} {e1} '
                        f'<span class="relation-arrow">➜ {orig["relation"]} ➜</span> {e2}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(_triplet_html(orig, show_verdict=verdict), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Verification Logs (Expandable) ───────────────────
        if verified_data and verified_data.get("verification_decisions"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📋  Verification Audit Log</div>', unsafe_allow_html=True)
            with st.expander("Show detailed reasoning for each triplet", expanded=False):
                for i, d in enumerate(verified_data["verification_decisions"], 1):
                    orig = d["original"]
                    st.markdown(
                        f'**{i}.** {_verdict_badge(d["verdict"])} '
                        f'`{orig["entity1"]}` ➜ `{orig["relation"]}` ➜ `{orig["entity2"]}`',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"**Reasoning:** {d['reasoning']}")
                    if d["verdict"] == "corrected" and d.get("corrected"):
                        c = d["corrected"]
                        st.markdown(
                            f'&nbsp;&nbsp;&nbsp;↳ *Corrected to:* `{c["entity1"]}` ➜ `{c["relation"]}` ➜ `{c["entity2"]}`'
                        )
                    st.markdown("")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Hallucination Reduction Chart ────────────────────
        if verified_data and verified_data.get("verification_decisions"):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📈  Hallucination Breakdown</div>', unsafe_allow_html=True)

            fig_pie, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))
            fig_pie.patch.set_facecolor("white")

            # Donut chart
            labels_pie, sizes_pie, colours_pie = [], [], []
            for lbl, cnt, col in [("Valid", n_valid, "#10B981"), ("Corrected", n_corrected, "#F59E0B"), ("Hallucinated", n_hallucinated, "#EF4444")]:
                if cnt > 0:
                    labels_pie.append(lbl)
                    sizes_pie.append(cnt)
                    colours_pie.append(col)

            if sizes_pie:
                wedges, texts, autotexts = ax_pie.pie(
                    sizes_pie, labels=labels_pie, colors=colours_pie,
                    autopct="%1.0f%%", startangle=90, pctdistance=0.75,
                    textprops={"fontsize": 10, "fontweight": "bold"},
                )
                centre_circle = plt.Circle((0, 0), 0.55, fc="white")
                ax_pie.add_artist(centre_circle)
                ax_pie.set_title("Verdict Distribution", fontsize=12, fontweight="bold", color="#0F172A")

            # Bar chart — only when comparing
            if run_compare and baseline_data:
                base_count = len(baseline_data.get("final_triplets", []))
                ver_count = len(verified_data.get("final_triplets", []))
                bars = ax_bar.bar(
                    ["Baseline", "Verified"],
                    [base_count, ver_count],
                    color=["#94A3B8", "#10B981"],
                    width=0.5,
                    edgecolor="white",
                    linewidth=1.5,
                )
                for bar in bars:
                    h = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width() / 2., h + 0.2,
                                f'{int(h)}', ha='center', va='bottom',
                                fontweight='bold', fontsize=13, color="#0F172A")
                ax_bar.set_ylabel("Triplet Count", fontsize=10, color="#64748B")
                ax_bar.set_title("Baseline vs Verified", fontsize=12, fontweight="bold", color="#0F172A")
                ax_bar.spines["top"].set_visible(False)
                ax_bar.spines["right"].set_visible(False)
                ax_bar.spines["left"].set_color("#E2E8F0")
                ax_bar.spines["bottom"].set_color("#E2E8F0")
                ax_bar.tick_params(colors="#64748B")
            else:
                ax_bar.axis("off")

            plt.tight_layout()
            st.pyplot(fig_pie)
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────── Knowledge Graph Visualisation ───────────────────────

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">🕸️  Knowledge Graph</div>', unsafe_allow_html=True)

if st.button("Refresh Graph Visualization", type="primary"):
    try:
        # Try results/ first, then data/
        kg_path = None
        for p in ["results/kg.json", "data/kg.json"]:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    kg_data = json.load(f)
                kg_path = p
                break
            except FileNotFoundError:
                continue

        if kg_path is None:
            st.info("No graph file found. Run an extraction first to populate `results/kg.json`.")
        else:
            G = nx.node_link_graph(kg_data, edges="links")

            if G.number_of_nodes() == 0:
                st.info("The knowledge graph is empty. Extract a CTI report to add nodes.")
            else:
                # Colour map
                COLOR_MAP = {
                    "threatactor": "#DC2626", "threat_actor": "#DC2626",
                    "malware": "#9333EA", "tool": "#2563EB",
                    "vulnerability": "#EA580C",
                    "technique": "#16A34A", "attack_pattern": "#16A34A", "attackpattern": "#16A34A",
                    "tactic": "#0D9488", "target": "#0891B2",
                }
                DEFAULT_COLOR = "#64748B"

                node_colors, node_sizes = [], []
                for _, d in G.nodes(data=True):
                    lbl = d.get("label", "").lower()
                    node_colors.append(COLOR_MAP.get(lbl, DEFAULT_COLOR))
                    deg = max(G.degree(_), 1) if G.degree(_) else 1
                    node_sizes.append(400 + deg * 200)

                fig, ax = plt.subplots(figsize=(12, 8))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")

                if len(G.nodes) < 30:
                    pos = nx.kamada_kawai_layout(G)
                else:
                    pos = nx.spring_layout(G, k=1.5, iterations=120, seed=42)

                nx.draw_networkx_nodes(
                    G, pos, node_color=node_colors, node_size=node_sizes,
                    ax=ax, alpha=0.92, edgecolors="white", linewidths=2,
                )
                nx.draw_networkx_edges(
                    G, pos, ax=ax, alpha=0.5, edge_color="#CBD5E1",
                    arrows=True, arrowsize=18, arrowstyle="-|>",
                    connectionstyle="arc3,rad=0.08",
                    min_source_margin=18, min_target_margin=18,
                )
                nx.draw_networkx_labels(
                    G, pos, ax=ax, font_size=9, font_family="sans-serif",
                    font_weight="bold", font_color="#1E293B",
                )
                edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(
                    G, pos, edge_labels=edge_labels, ax=ax, font_size=7.5,
                    font_color="#64748B",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=1.5),
                )

                # Legend patches
                seen = set()
                patches = []
                for _, d in G.nodes(data=True):
                    lbl = d.get("label", "other").lower()
                    if lbl not in seen:
                        seen.add(lbl)
                        col = COLOR_MAP.get(lbl, DEFAULT_COLOR)
                        patches.append(mpatches.Patch(color=col, label=lbl.replace("_", " ").title()))
                if patches:
                    ax.legend(handles=patches, loc="upper left", fontsize=8,
                              frameon=True, fancybox=True, shadow=False,
                              edgecolor="#E2E8F0")

                ax.set_axis_off()
                plt.tight_layout()
                st.pyplot(fig)

                # Quick stats row below graph
                st.markdown(
                    '<div class="metric-row">'
                    + _metric_tile(G.number_of_nodes(), "Nodes", "blue")
                    + _metric_tile(G.number_of_edges(), "Edges", "blue")
                    + _metric_tile(len(seen), "Entity Types", "green")
                    + '</div>',
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Failed to render graph: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────── Footer ──────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; color:#94A3B8; font-size:0.78rem; padding:8px 0 24px;">'
    'Verification-in-the-Loop CTI Knowledge Graph &mdash; Built with Streamlit, FastAPI, NetworkX & MITRE ATT&CK'
    '</div>',
    unsafe_allow_html=True,
)
