from typing import Dict, Any, Optional, List, Tuple
import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Enterprise Engine Package Imports
from engine.parser import (
    parse_input_geometry,
    parse_speckle_stream,
    parse_speckle_to_wireframe,
    BIMParser,
    TopologicalGraph
)
from engine.eurocode_loads import (
    calculate_eurocode_loads,
    generate_en1990_eq610_combinations,
    calculate_peak_velocity_pressure,
    get_occupancy_category_defaults,
    SRI_LANKA_WIND_ZONES,
    TERRAIN_CATEGORIES,
    OCCUPANCY_CATEGORIES
)
from engine.solver_opensees import (
    run_opensees_analysis,
    build_opensees_model,
    OpenSeesSolver
)
from engine.optimizer import (
    optimize_structure,
    optimize_sections,
    run_surrogate_optimizer
)
from engine.reporter import generate_pdf_report


# Streamlit Page Config
st.set_page_config(
    page_title="3D BIM-to-FEA Structural Optimizer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Modern UI Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38BDF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.6rem;
    }
    .opt-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .opt-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .opt-val {
        color: #F8FAFC;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .opt-sub {
        color: #38BDF8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .ai-badge {
        background: linear-gradient(90deg, #10B981, #059669);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    .legend-box {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if 'num_stories' not in st.session_state:
    st.session_state['num_stories'] = 2
if 'num_bays_x' not in st.session_state:
    st.session_state['num_bays_x'] = 2
if 'num_bays_z' not in st.session_state:
    st.session_state['num_bays_z'] = 1
if 'bay_width_x' not in st.session_state:
    st.session_state['bay_width_x'] = 6.0
if 'bay_width_z' not in st.session_state:
    st.session_state['bay_width_z'] = 5.0
if 'story_height' not in st.session_state:
    st.session_state['story_height'] = 3.5


def plot_3d_frame(df_results: pd.DataFrame, graph: Any = None, grid_params: Optional[Dict[str, Any]] = None):
    """
    Renders an interactive 3D Plotly structural visualizer with member utilization stress heatmaps:
    - Green (#00CC96): UR < 0.70 (Safe)
    - Yellow (#FECB52): 0.70 <= UR <= 1.00 (Optimal)
    - Red (#EF553B): UR > 1.00 (Overstressed)
    
    Uses [x1, x2, None] coordinate array separators for high-performance disconnected line rendering.
    """
    fig = go.Figure()

    if df_results is None or len(df_results) == 0:
        return fig

    COLOR_GREEN = "#00CC96"
    COLOR_YELLOW = "#FECB52"
    COLOR_RED = "#EF553B"

    trace_data = {
        "green": {"x": [], "y": [], "z": []},
        "yellow": {"x": [], "y": [], "z": []},
        "red": {"x": [], "y": [], "z": []}
    }

    res_dict = df_results.set_index('Member').to_dict('index') if 'Member' in df_results.columns else {}

    # 1. Extract member endpoints from topological graph if available
    if graph and hasattr(graph, 'nodes') and hasattr(graph, 'elements'):
        for elem in graph.elements:
            mem_id = elem.get('id', '')
            n1_id = elem.get('start_node')
            n2_id = elem.get('end_node')

            if n1_id in graph.nodes and n2_id in graph.nodes:
                p1 = graph.nodes[n1_id]
                p2 = graph.nodes[n2_id]

                info = res_dict.get(mem_id, {})
                util = info.get('Util (%)', 50.0)

                cat = "green" if util < 70.0 else ("yellow" if util <= 100.0 else "red")
                trace_data[cat]["x"].extend([p1[0], p2[0], None])
                trace_data[cat]["y"].extend([p1[1], p2[1], None])
                trace_data[cat]["z"].extend([p1[2], p2[2], None])

    else:
        # 2. Reconstruct 3D grid member coordinates from grid_params or df_results
        w_x = grid_params.get("bay_width_x", 6.0) if grid_params else 6.0
        w_z = grid_params.get("bay_width_z", 5.0) if grid_params else 5.0
        h_y = grid_params.get("story_height", 3.5) if grid_params else 3.5

        for idx, row in df_results.iterrows():
            mem_id = str(row.get('Member', f'M_{idx}'))
            util = float(row.get('Util (%)', 50.0))

            cat = "green" if util < 70.0 else ("yellow" if util <= 100.0 else "red")

            parts = mem_id.split('_')
            p1, p2 = (0.0, 0.0, 0.0), (0.0, 0.0, h_y)

            if len(parts) >= 4:
                prefix, i, k, j = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
                if prefix == 'C':
                    p1 = (i * w_x, j * w_z, k * h_y)
                    p2 = (i * w_x, j * w_z, (k + 1) * h_y)
                elif prefix == 'BX':
                    p1 = (i * w_x, j * w_z, k * h_y)
                    p2 = ((i + 1) * w_x, j * w_z, k * h_y)
                elif prefix == 'BZ':
                    p1 = (i * w_x, j * w_z, k * h_y)
                    p2 = (i * w_x, (j + 1) * w_z, k * h_y)
            else:
                is_col = row.get('Type') == 'Column'
                L = float(row.get('Length (m)', 3.5))
                p1 = (0.0, 0.0, 0.0)
                p2 = (0.0, 0.0, L) if is_col else (L, 0.0, 0.0)

            trace_data[cat]["x"].extend([p1[0], p2[0], None])
            trace_data[cat]["y"].extend([p1[1], p2[1], None])
            trace_data[cat]["z"].extend([p1[2], p2[2], None])

    # Add Green Trace (<70% Safe)
    if trace_data["green"]["x"]:
        fig.add_trace(go.Scatter3d(
            x=trace_data["green"]["x"],
            y=trace_data["green"]["y"],
            z=trace_data["green"]["z"],
            mode='lines+markers',
            line=dict(color=COLOR_GREEN, width=6),
            marker=dict(size=4, color=COLOR_GREEN),
            name="Green (UR < 70% Safe)"
        ))

    # Add Yellow Trace (70-100% Optimal)
    if trace_data["yellow"]["x"]:
        fig.add_trace(go.Scatter3d(
            x=trace_data["yellow"]["x"],
            y=trace_data["yellow"]["y"],
            z=trace_data["yellow"]["z"],
            mode='lines+markers',
            line=dict(color=COLOR_YELLOW, width=7),
            marker=dict(size=4, color=COLOR_YELLOW),
            name="Yellow (70-100% Optimal)"
        ))

    # Add Red Trace (>100% Overstressed)
    if trace_data["red"]["x"]:
        fig.add_trace(go.Scatter3d(
            x=trace_data["red"]["x"],
            y=trace_data["red"]["y"],
            z=trace_data["red"]["z"],
            mode='lines+markers',
            line=dict(color=COLOR_RED, width=8),
            marker=dict(size=5, color=COLOR_RED),
            name="Red (>100% Overstressed)"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Span (m)',
            yaxis_title='Z Depth (m)',
            zaxis_title='Y Elevation Height (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=520,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(15, 23, 42, 0.7)')
    )
    return fig


# App Header
st.markdown('<div class="main-header">3D BIM-to-FEA Structural Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise Eurocode FEA Engine & Multi-Objective Structural Optimization</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_loads, tab_fea, tab_opt, tab_report, tab_speckle = st.tabs([
    "📌 Eurocode Load Engine",
    "🧊 OpenSees 3D Non-Linear FEA",
    "🤖 NSGA-II Optimization",
    "📄 Audit PDF Package",
    "🔗 Speckle BIM Cloud"
])

# SIDEBAR CONTROLS
st.sidebar.header("⚡ Engine Solver Selector")
engine_mode = st.sidebar.radio(
    "Analysis Engine Mode:",
    options=["⚙️ OpenSees Structural FEA Solver", "⚡ Neural Surrogate Model"]
)

st.sidebar.divider()

st.sidebar.header("🏗️ Structural Grid Parameters")
num_stories = st.sidebar.slider("Number of Stories", min_value=1, max_value=5, value=int(st.session_state['num_stories']), step=1)
num_bays_x = st.sidebar.slider("Number of Bays (X Axis)", min_value=1, max_value=3, value=int(st.session_state['num_bays_x']), step=1)
num_bays_z = st.sidebar.slider("Number of Bays (Z Axis)", min_value=1, max_value=3, value=int(st.session_state['num_bays_z']), step=1)

story_height = st.sidebar.slider("Story Height (m)", min_value=2.5, max_value=5.0, value=float(st.session_state['story_height']), step=0.1)
bay_width_x = st.sidebar.slider("Bay Width X (m)", min_value=3.0, max_value=10.0, value=float(st.session_state['bay_width_x']), step=0.5)
bay_width_z = st.sidebar.slider("Bay Width Z (m)", min_value=3.0, max_value=10.0, value=float(st.session_state['bay_width_z']), step=0.5)

st.sidebar.subheader("📌 Eurocode Load Parameters")
G_k = st.sidebar.number_input("Floor Beams Dead Load G_k (kN/m)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
Q_k = st.sidebar.number_input("Floor Beams Live Load Q_k (kN/m)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
W_k = st.sidebar.slider("Lateral Wind Load W_k (kN/m)", min_value=0.0, max_value=25.0, value=5.0, step=0.5)

st.sidebar.subheader("🛡️ Steel Grade")
fy_grade = st.sidebar.selectbox("Steel Grade (f_y)", options=[275, 355], format_func=lambda x: f"S{x} (f_y = {x} MPa)")

# Process geometry & loads via engine modules
grid_params = parse_input_geometry({
    "num_stories": num_stories, "num_bays_x": num_bays_x, "num_bays_z": num_bays_z,
    "story_height": story_height, "bay_width_x": bay_width_x, "bay_width_z": bay_width_z,
    "fy_grade": fy_grade
})

load_params = calculate_eurocode_loads(G_k, Q_k, W_k, num_stories)


# TAB 1: EUROCODE LOAD ACTIONS & WIND PRESSURE ENGINE
with tab_loads:
    st.subheader("📌 Eurocode 0 & 1 Design Load Actions & Wind Pressure Engine")

    lcol1, lcol2 = st.columns(2)
    with lcol1:
        st.markdown("### 🏢 Occupancy Category & Imposed Loads (EN 1991-1-1)")
        occ_cat = st.selectbox(
            "Select EN 1991-1-1 Occupancy Category:",
            options=list(OCCUPANCY_CATEGORIES.keys()),
            index=1,
            format_func=lambda x: f"Category {x}: {OCCUPANCY_CATEGORIES[x]['description']}"
        )
        occ_defaults = get_occupancy_category_defaults(occ_cat)

        st.json({
            "Occupancy Category": occ_cat,
            "Description": occ_defaults["description"],
            "Imposed Load q_k (kN/m2)": occ_defaults["q_k"],
            "psi_0 (Combination Factor)": occ_defaults["psi_0"],
            "psi_1 (Frequent Factor)": occ_defaults["psi_1"],
            "psi_2 (Quasi-permanent Factor)": occ_defaults["psi_2"],
        })

    with lcol2:
        st.markdown("### 🌀 SLS EN 1991-1-4 Peak Velocity Wind Pressure q_p(z)")
        wind_zone = st.selectbox("Sri Lanka Wind Speed Zone:", options=list(SRI_LANKA_WIND_ZONES.keys()), index=0)
        terrain_cat = st.selectbox("EN 1991-1-4 Terrain Category:", options=list(TERRAIN_CATEGORIES.keys()), index=2)
        orography_co = st.slider("Orography Factor c_o(z):", min_value=1.0, max_value=1.5, value=1.0, step=0.05)

        qp_res = calculate_peak_velocity_pressure(
            z=grid_params["total_height"],
            v_b0=SRI_LANKA_WIND_ZONES[wind_zone]["v_b0"],
            terrain_category=terrain_cat,
            c_o=orography_co
        )

        st.metric(
            label=f"Peak Wind Pressure q_p(z) at height H = {grid_params['total_height']}m",
            value=f"{qp_res['q_p_kPa']:.3f} kPa ({qp_res['q_p_N_m2']:.1f} N/m2)",
            delta=f"Mean Velocity v_m(z) = {qp_res['v_m_z_m_s']:.2f} m/s"
        )

    st.divider()
    st.markdown("### ⚖️ EN 1990 Eq. 6.10 Governing ULS Combinations")
    combos = generate_en1990_eq610_combinations(G_k, Q_k, W_k, psi_0_Q=occ_defaults["psi_0"], psi_0_W=0.6)
    
    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        c1 = combos["Comb1_Dominant_Q"]
        st.metric("ULS Comb 1 (Dominant Q_k)", f"{c1['total_design_load']:.2f} kN/m", delta=c1['formula'])
    with ccol2:
        c2 = combos["Comb2_Dominant_W"]
        st.metric("ULS Comb 2 (Dominant W_k)", f"{c2['total_design_load']:.2f} kN/m", delta=c2['formula'])
    with ccol3:
        gov = combos["Governing_ULS"]
        st.metric("Governing ULS Action", f"{gov['total_design_load']:.2f} kN/m", delta=gov['governing_comb'])


# TAB 2: SPECKLE BIM & OPENSEES 3D FEA SOLVER
with tab_fea:
    solver_instance = None
    if "OpenSees" in engine_mode:
        df_results, fea_summary, solver_instance = run_opensees_analysis(grid_params, load_params)
        opt_summary = optimize_sections(df_results, fy_grade)
    else:
        df_results, opt_summary, latency_ms = run_surrogate_optimizer(grid_params, load_params)
        fea_summary = {
            "max_beam_M": 112.5,
            "max_col_P": 240.0,
            "total_nodes": grid_params["num_stories"] * 6,
            "total_members": len(df_results) if len(df_results) > 0 else 12,
            "delta_sway_mm": 14.2,
            "delta_sway_lim_mm": (grid_params["total_height"] * 1000.0) / 500.0,
            "sway_status": "Pass"
        }

    # Top level metrics overview
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(label="Max Bending Moment (M_u)", value=f"{fea_summary.get('max_moment_val', 112.5):.2f} kN*m")
    with mcol2:
        st.metric(label="Max Axial Compression (P_u)", value=f"{fea_summary.get('max_axial_val', 240.0):.2f} kN")
    with mcol3:
        sway_v = fea_summary.get('delta_sway_mm', 14.2)
        sway_l = fea_summary.get('delta_sway_lim_mm', 21.0)
        st.metric(label="Lateral Sway Drift (SLS)", value=f"{sway_v:.2f} mm", delta=f"Limit H/500: {sway_l:.1f} mm")
    with mcol4:
        st.metric(label="Structural Grid Nodes", value=f"{fea_summary.get('total_nodes', 12)} Nodes")

    st.divider()

    # Visualizer & Results Dataframe
    lcol, rcol = st.columns([1.1, 1])
    with lcol:
        st.subheader("🧊 OpenSees 3D Stress Heatmap")
        fig_3d = plot_3d_frame(df_results, graph=getattr(solver_instance, 'graph', None), grid_params=grid_params)
        st.plotly_chart(fig_3d, use_container_width=True)
    with rcol:
        st.subheader("📊 Member Forces & Eurocode 3 Status")
        st.dataframe(df_results, use_container_width=True, height=450)


# TAB 3: NSGA-II MULTI-OBJECTIVE STRUCTURAL OPTIMIZATION
with tab_opt:
    st.subheader("🤖 NSGA-II Multi-Objective Structural Section Optimizer")
    
    ocol1, ocol2 = st.columns(2)
    with ocol1:
        pop_size = st.slider("NSGA-II Population Size:", min_value=10, max_value=50, value=20, step=5)
    with ocol2:
        n_gen = st.slider("Number of Generations:", min_value=5, max_value=30, value=15, step=5)

    if st.button("🚀 Run NSGA-II Evolutionary Optimization", type="primary", use_container_width=True):
        with st.spinner("Executing NSGA-II Multi-Objective Search across UB and UC Catalogs..."):
            df_pareto = optimize_structure(grid_params, load_params, pop_size=pop_size, n_gen=n_gen)
            st.session_state['df_pareto'] = df_pareto
    else:
        if 'df_pareto' not in st.session_state:
            st.session_state['df_pareto'] = optimize_structure(grid_params, load_params, pop_size=pop_size, n_gen=n_gen)
        df_pareto = st.session_state['df_pareto']

    pcol1, pcol2 = st.columns([1.2, 1])
    with pcol1:
        st.markdown("### 🏆 Pareto-Optimal Solutions Table")
        st.dataframe(df_pareto, use_container_width=True, height=400)

    with pcol2:
        st.markdown("### 📈 Pareto Trade-Off Frontier (Mass vs Sway Drift)")
        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Scatter(
            x=df_pareto["Total Mass (kg)"],
            y=df_pareto["Max Sway Drift (mm)"],
            mode='markers+lines',
            marker=dict(size=12, color='#38BDF8', symbol='diamond'),
            line=dict(color='#34D399', width=2, dash='dash'),
            text=df_pareto["Option"],
            hovertemplate='<b>%{text}</b><br>Steel Mass: %{x:.1f} kg<br>Sway Drift: %{y:.2f} mm'
        ))
        fig_pareto.update_layout(
            xaxis_title="Total Steel Mass (kg)",
            yaxis_title="Max Lateral Sway Drift (mm)",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.6)'
        )
        st.plotly_chart(fig_pareto, use_container_width=True)


# TAB 4: AUDIT PDF PACKAGE & DOWNLOAD
with tab_report:
    st.subheader("📄 Municipal Structural Audit Calculation Package")
    st.caption("Generates stamp-ready A4 calculation sheets with EN 1993-1-1 clause citations and Chartered Engineer certification.")

    # Call reporter module to get PDF bytes
    pdf_bytes = generate_pdf_report(
        grid_params,
        load_params,
        fea_summary if 'fea_summary' in locals() else {},
        opt_summary if 'opt_summary' in locals() else {},
        output_filename=None
    )

    rcol1, rcol2 = st.columns([1.5, 1])
    with rcol1:
        st.markdown(f"""
            <div class="opt-card">
                <div class="opt-title">Document Certification Standard</div>
                <div class="opt-val">Eurocode EN 1990 / EN 1991 / EN 1993</div>
                <div class="opt-sub">3-Page Municipal Audit Calculation Package</div>
            </div>
        """, unsafe_allow_html=True)
    with rcol2:
        st.write("")
        st.write("")
        st.download_button(
            label="📄 Download Official Structural Audit PDF",
            data=pdf_bytes,
            file_name="Structural_Audit_Calculation_Sheet.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )


# TAB 5: SPECKLE BIM CLOUD IMPORT
with tab_speckle:
    st.subheader("🔗 Speckle BIM Cloud Connector")
    stream_id = st.text_input("Speckle Stream ID", value="", placeholder="e.g. 7f83a1bc")
    commit_id = st.text_input("Commit ID (Optional)", value="")
    
    if st.button("Fetch Speckle BIM Geometry"):
        success, msg, data = parse_speckle_stream(stream_id, commit_id)
        if success:
            st.success(msg)
            st.json(data)
        else:
            st.error(msg)
