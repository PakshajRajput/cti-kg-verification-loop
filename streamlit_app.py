import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt
import json
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="CTI Extraction Pipeline", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for SaaS Dashboard Look ---
st.markdown("""
    <style>
        /* Base Theme */
        .stApp {
            background-color: #F8FAFC;
            color: #1E293B;
            font-family: 'Inter', sans-serif;
        }
        /* Gradient Header */
        .gradient-header {
            background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        /* Cards */
        .saas-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin-bottom: 20px;
            border: 1px solid #E2E8F0;
        }
        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25em 0.6em;
            font-size: 0.75em;
            font-weight: 700;
            border-radius: 9999px;
            color: white;
            margin-right: 5px;
        }
        .badge-threat_actor, .badge-threatactor { background-color: #E53935; }
        .badge-malware { background-color: #8E24AA; }
        .badge-tool { background-color: #1E88E5; }
        .badge-vulnerability { background-color: #FB8C00; }
        .badge-technique, .badge-attack_pattern { background-color: #43A047; }
        .badge-target { background-color: #00897B; }
        .badge-other { background-color: #607D8B; }
        
        .badge-valid { background-color: #10B981; }
        .badge-revised { background-color: #FB8C00; }
        .badge-rejected { background-color: #E53935; }
        
        /* Typography */
        h1, h2, h3 { color: #0F172A; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-header">🛡️ Verification-in-the-Loop CTI Extraction</div>', unsafe_allow_html=True)
st.markdown("**Extract, Verify, and Construct Knowledge Graphs from Cyber Threat Intelligence using LLM Pipelines.**", unsafe_allow_html=True)
st.divider()

# --- Sidebar ---
st.sidebar.markdown("### ⚙️ System Status")
try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success(f"API: {health_data.get('status', 'offline').upper()}")
        st.sidebar.info(f"RAG Populated: {health_data.get('vector_store_populated', False)}")
        st.sidebar.info(f"Graph Connected: {health_data.get('neo4j_connected', False)}")
    else:
        st.sidebar.error("API Error")
except Exception:
    st.sidebar.error("API Offline")

st.sidebar.markdown("### 📊 Graph Stats")
try:
    stats_resp = requests.get(f"{API_URL}/graph/stats", timeout=3)
    if stats_resp.status_code == 200:
        stats = stats_resp.json()
        st.sidebar.metric("Nodes", stats.get("total_nodes", 0))
        st.sidebar.metric("Edges", stats.get("total_edges", 0))
except Exception:
    pass

def get_badge(entity_type, text):
    css_class = f"badge badge-{entity_type.lower()}"
    if entity_type.lower() not in ["threatactor", "threat_actor", "malware", "technique", "attack_pattern", "tool", "vulnerability", "target"]:
        css_class = "badge badge-other"
    return f'<span class="{css_class}">{text}</span>'

# --- Main Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.subheader("📄 1. Input CTI Report")
    text_input = st.text_area("Paste your unstructured CTI text here:", height=200, placeholder="Example: APT29 uses Cobalt Strike to exfiltrate data via DNS...")
    
    col_btn1, col_btn2 = st.columns(2)
    run_full = col_btn1.button("Extract (Verification Loop)", type="primary", use_container_width=True)
    run_base = col_btn2.button("Extract (Baseline)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if run_full or run_base:
    if not text_input.strip():
        st.warning("Please provide a CTI report text.")
    else:
        endpoint = "/extract" if run_full else "/extract/baseline"
        
        with st.spinner("Processing CTI Report..."):
            try:
                start_time = time.time()
                res = requests.post(
                    f"{API_URL}{endpoint}",
                    json={"text": text_input, "source": "streamlit"},
                    timeout=300
                )
                elapsed = time.time() - start_time
                if res.status_code == 200:
                    data = res.json()
                    if data.get("error"):
                        st.error(f"Pipeline Error: {data['error']}")
                    else:
                        st.success(f"Processing completed in {elapsed:.1f}s")
                        
                        with col_left:
                            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                            st.subheader("🔍 2. Raw Extracted Triplets")
                            for d in data.get("verification_decisions", []):
                                orig = d["original"]
                                e1 = get_badge(orig['entity1_type'], orig['entity1'])
                                e2 = get_badge(orig['entity2_type'], orig['entity2'])
                                st.markdown(f"• {e1} ➔ *{orig['relation']}* ➔ {e2}", unsafe_allow_html=True)
                            if not data.get("verification_decisions"):
                                st.info("No triplets extracted.")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_right:
                            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                            st.subheader("✅ 3. Verified Triplets")
                            for t in data.get('final_triplets', []):
                                e1 = get_badge(t['entity1_type'], t['entity1'])
                                e2 = get_badge(t['entity2_type'], t['entity2'])
                                st.markdown(f"• {e1} ➔ **{t['relation']}** ➔ {e2}", unsafe_allow_html=True)
                            if not data.get("final_triplets"):
                                st.info("No triplets survived verification.")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                            st.subheader("📈 4. Hallucination Reduction")
                            total = len(data.get("verification_decisions", []))
                            valid = len([d for d in data.get("verification_decisions", []) if d["verdict"] == "valid"])
                            revised = len([d for d in data.get("verification_decisions", []) if d["verdict"] == "corrected"])
                            rejected = len([d for d in data.get("verification_decisions", []) if d["verdict"] == "hallucinated"])
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Valid", valid)
                            m2.metric("Revised", revised)
                            m3.metric("Rejected (Hallucinations)", rejected)
                            
                            with st.expander("View Verification Logs"):
                                for d in data.get("verification_decisions", []):
                                    orig = d["original"]
                                    status = f'<span class="badge badge-{d["verdict"]}">{d["verdict"].upper()}</span>'
                                    st.markdown(f"{status} **{orig['entity1']} ➔ {orig['relation']} ➔ {orig['entity2']}**", unsafe_allow_html=True)
                                    st.caption(f"Reasoning: {d['reasoning']}")
                                    if d["corrected"]:
                                        c = d["corrected"]
                                        st.markdown(f"↳ *Corrected to:* {c['entity1']} ➔ {c['relation']} ➔ {c['entity2']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

st.divider()

col_graph, col_legend = st.columns([3, 1], gap="large")

with col_graph:
    st.subheader("🕸️ Knowledge Graph Visualization")
    if st.button("Refresh Visualization", type="primary"):
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        try:
            with open("results/kg.json", "r", encoding="utf-8") as f:
                kg_data = json.load(f)
            G = nx.node_link_graph(kg_data, edges="links")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('#F8FAFC')
            
            if len(G.nodes) < 30:
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G, k=1.2, iterations=100, seed=42)
            
            node_colors = []
            node_sizes = []
            for node, d in G.nodes(data=True):
                lbl = d.get("label", "").lower()
                if lbl in ["threatactor", "threat_actor"]: node_colors.append("#E53935")
                elif lbl == "malware": node_colors.append("#8E24AA")
                elif lbl == "tool": node_colors.append("#1E88E5")
                elif lbl == "vulnerability": node_colors.append("#FB8C00")
                elif lbl in ["technique", "attack_pattern", "attackpattern"]: node_colors.append("#43A047")
                elif lbl == "target": node_colors.append("#00897B")
                else: node_colors.append("#607D8B")
                
                degree = G.degree(node) if G.degree(node) is not None else 1
                node_sizes.append(500 + degree * 250)
                
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9, edgecolors="white", linewidths=1.5)
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, edge_color="#CBD5E1", arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1", min_source_margin=15, min_target_margin=15)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_family="sans-serif", font_weight="bold", font_color="#1E293B")
            
            edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
            
            ax.set_axis_off()
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            with col_legend:
                st.markdown('<div class="saas-card" style="margin-top: 50px;">', unsafe_allow_html=True)
                st.markdown("### Legend")
                st.markdown('🔴 <span class="badge" style="background-color: #E53935;">ThreatActor</span>', unsafe_allow_html=True)
                st.markdown('🟣 <span class="badge" style="background-color: #8E24AA;">Malware</span>', unsafe_allow_html=True)
                st.markdown('🔵 <span class="badge" style="background-color: #1E88E5;">Tool</span>', unsafe_allow_html=True)
                st.markdown('🟠 <span class="badge" style="background-color: #FB8C00;">Vulnerability</span>', unsafe_allow_html=True)
                st.markdown('🟢 <span class="badge" style="background-color: #43A047;">Technique</span>', unsafe_allow_html=True)
                st.markdown('🟦 <span class="badge" style="background-color: #00897B;">Target</span>', unsafe_allow_html=True)
                st.markdown('⚪ <span class="badge" style="background-color: #607D8B;">Unknown</span>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("### Metrics")
                st.metric("Total Nodes", G.number_of_nodes())
                st.metric("Total Edges", G.number_of_edges())
                st.markdown('</div>', unsafe_allow_html=True)
                
        except FileNotFoundError:
            st.info("Graph is empty. Run the extraction pipeline first to generate `results/kg.json`.")
        except Exception as e:
            st.error(f"Failed to render graph: {e}")
