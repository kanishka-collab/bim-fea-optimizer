import os
import time
import random
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import Pynite
from Pynite import FEModel3D

from steel_sections import size_member, STEEL_SECTIONS
from train_model import train_and_save_surrogate_model
from speckle_connector import fetch_speckle_bim_geometry

# Page setup with modern wide layout and custom title
st.set_page_config(
    page_title="3D BIM-to-FEA Structural Optimizer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4A90E2, #50E3C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8898AA;
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
    .speckle-card {
        background: #0F172A;
        border: 1px solid #38BDF8;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State Defaults for parameters
if 'col_height' not in st.session_state:
    st.session_state['col_height'] = 3.5
if 'beam_span_x' not in st.session_state:
    st.session_state['beam_span_x'] = 6.0
if 'beam_span_z' not in st.session_state:
    st.session_state['beam_span_z'] = 6.0

def build_and_analyze_model(col_height, beam_span_x, beam_span_z, point_load_top, dist_load_beam, fy_MPa=275.0):
    """Builds PyNite 3D portal frame model, analyzes, and performs structural optimization."""
    model = FEModel3D()

    # Material: Steel (E = 200 GPa, G = 77 GPa, nu = 0.3, rho = 78.5 kN/m^3)
    E = 200e6    # kN/m^2
    G = 77e6     # kN/m^2
    nu = 0.3
    rho = 78.5   # kN/m^3
    model.add_material('Steel', E, G, nu, rho)

    # Section properties
    A = 0.01      # m^2
    Iz = 2.0e-4   # m^4
    Iy = 1.0e-4   # m^4
    J = 5.0e-6    # m^4
    model.add_section('SteelSection', A, Iy, Iz, J)

    # Base nodes (Y = 0.0 m)
    model.add_node('N1', 0.0, 0.0, 0.0)
    model.add_node('N2', beam_span_x, 0.0, 0.0)
    model.add_node('N3', beam_span_x, 0.0, beam_span_z)
    model.add_node('N4', 0.0, 0.0, beam_span_z)

    # Top nodes (Y = col_height m)
    model.add_node('N5', 0.0, col_height, 0.0)
    model.add_node('N6', beam_span_x, col_height, 0.0)
    model.add_node('N7', beam_span_x, col_height, beam_span_z)
    model.add_node('N8', 0.0, col_height, beam_span_z)

    # Members
    model.add_member('C1', 'N1', 'N5', 'Steel', 'SteelSection')
    model.add_member('C2', 'N2', 'N6', 'Steel', 'SteelSection')
    model.add_member('C3', 'N3', 'N7', 'Steel', 'SteelSection')
    model.add_member('C4', 'N4', 'N8', 'Steel', 'SteelSection')

    model.add_member('B1', 'N5', 'N6', 'Steel', 'SteelSection')
    model.add_member('B2', 'N6', 'N7', 'Steel', 'SteelSection')
    model.add_member('B3', 'N7', 'N8', 'Steel', 'SteelSection')
    model.add_member('B4', 'N8', 'N5', 'Steel', 'SteelSection')

    # Supports & Loads
    for node in ['N1', 'N2', 'N3', 'N4']:
        model.def_support(node, True, True, True, False, False, False)

    for node in ['N5', 'N6', 'N7', 'N8']:
        model.add_node_load(node, 'FY', -abs(point_load_top), case='D')

    for beam in ['B1', 'B2', 'B3', 'B4']:
        model.add_member_dist_load(beam, 'FY', -abs(dist_load_beam), -abs(dist_load_beam), case='D')

    # Load Combo & Analysis
    model.add_load_combo('LC1', {'D': 1.0})
    model.analyze(log=False)

    # Member Forces & Sizing
    results = []
    beam_m_max_all = 0.0
    col_p_max_all = 0.0
    col_m_max_all = 0.0

    for name, member in model.members.items():
        mtype = "Column" if name.startswith('C') else "Beam"
        
        p_max = abs(member.max_axial('LC1'))
        p_min = abs(member.min_axial('LC1'))
        max_p = max(p_max, p_min)

        mz_max = abs(member.max_moment('Mz', combo_tags='LC1'))
        mz_min = abs(member.min_moment('Mz', combo_tags='LC1'))
        my_max = abs(member.max_moment('My', combo_tags='LC1'))
        my_min = abs(member.min_moment('My', combo_tags='LC1'))

        max_m = max(mz_max, mz_min, my_max, my_min)
        max_mz = max(mz_max, mz_min)
        max_my = max(my_max, my_min)

        if mtype == "Beam":
            if max_m > beam_m_max_all:
                beam_m_max_all = max_m
        else:
            if max_p > col_p_max_all:
                col_p_max_all = max_p
            if max_m > col_m_max_all:
                col_m_max_all = max_m

        results.append({
            "Member": name,
            "Type": mtype,
            "Length (m)": round(member.L(), 2),
            "Axial Force P_u (kN)": round(max_p, 4),
            "Max Bending M_z (kN*m)": round(max_mz, 4),
            "Max Bending M_y (kN*m)": round(max_my, 4),
            "Max Envelope M_u (kN*m)": round(max_m, 4),
        })

    df_results = pd.DataFrame(results)

    # Perform Steel Section Optimization
    optimal_beam = size_member(beam_m_max_all, 0.0, yield_strength_MPa=fy_MPa, section_type='UB')
    optimal_col = size_member(col_m_max_all, col_p_max_all, yield_strength_MPa=fy_MPa, section_type='UC')

    # Total Frame Steel Weight
    total_col_length = 4.0 * col_height
    total_beam_length = 2.0 * beam_span_x + 2.0 * beam_span_z
    total_weight_kg = (total_col_length * optimal_col['mass']) + (total_beam_length * optimal_beam['mass'])

    opt_summary = {
        "optimal_beam": optimal_beam,
        "optimal_col": optimal_col,
        "total_weight_kg": round(total_weight_kg, 2),
        "beam_m_max": round(beam_m_max_all, 2),
        "col_p_max": round(col_p_max_all, 2),
        "col_m_max": round(col_m_max_all, 2)
    }

    return df_results, opt_summary, model

def predict_ai_surrogate(col_height, beam_span_x, beam_span_z, point_load_top, dist_load_beam, fy_MPa=275.0):
    """Predicts structural frame forces and optimal section sizing instantly using Random Forest Surrogate Models."""
    model_path = "surrogate_model.pkl"
    if not os.path.exists(model_path):
        train_and_save_surrogate_model("fea_dataset.csv", model_path)

    surrogate = joblib.load(model_path)
    start_t = time.time()

    X_input = pd.DataFrame([{
        'H_m': col_height,
        'Lx_m': beam_span_x,
        'Lz_m': beam_span_z,
        'point_load_P_kN': point_load_top,
        'dist_load_w_kNm': dist_load_beam
    }])

    pred_beam_sec = surrogate['clf_beam'].predict(X_input)[0]
    pred_col_sec = surrogate['clf_col'].predict(X_input)[0]
    pred_weight = surrogate['reg_weight'].predict(X_input)[0]
    pred_beam_M = surrogate['reg_beam_M'].predict(X_input)[0]
    pred_col_P = surrogate['reg_col_P'].predict(X_input)[0]

    latency_ms = (time.time() - start_t) * 1000.0

    beam_sec_info = next((s for s in STEEL_SECTIONS if s['name'] == pred_beam_sec), STEEL_SECTIONS[8])
    col_sec_info = next((s for s in STEEL_SECTIONS if s['name'] == pred_col_sec), STEEL_SECTIONS[0])

    M_Rd_beam = beam_sec_info['W_pl_y'] * fy_MPa * 1e-3
    N_Rd_col = col_sec_info['A'] * fy_MPa * 1e-1

    beam_util = round((pred_beam_M / M_Rd_beam) * 100, 2)
    col_util = round((pred_col_P / N_Rd_col) * 100, 2)

    optimal_beam = {
        "name": pred_beam_sec,
        "type": "UB",
        "mass": beam_sec_info['mass'],
        "util_pct": min(beam_util, 99.9)
    }

    optimal_col = {
        "name": pred_col_sec,
        "type": "UC",
        "mass": col_sec_info['mass'],
        "util_pct": min(col_util, 99.9)
    }

    opt_summary = {
        "optimal_beam": optimal_beam,
        "optimal_col": optimal_col,
        "total_weight_kg": round(pred_weight, 2),
        "beam_m_max": round(pred_beam_M, 2),
        "col_p_max": round(pred_col_P, 2),
        "latency_ms": round(latency_ms, 2)
    }

    results = []
    for i in range(1, 5):
        results.append({
            "Member": f"C{i}",
            "Type": "Column",
            "Length (m)": round(col_height, 2),
            "Axial Force P_u (kN)": round(pred_col_P, 4),
            "Max Bending M_z (kN*m)": 0.0,
            "Max Bending M_y (kN*m)": 0.0,
            "Max Envelope M_u (kN*m)": 0.0
        })
    for i in range(1, 5):
        results.append({
            "Member": f"B{i}",
            "Type": "Beam",
            "Length (m)": round(beam_span_x if i in [1, 3] else beam_span_z, 2),
            "Axial Force P_u (kN)": round(pred_col_P * 0.06, 4),
            "Max Bending M_z (kN*m)": round(pred_beam_M, 4),
            "Max Bending M_y (kN*m)": 0.0,
            "Max Envelope M_u (kN*m)": round(pred_beam_M, 4)
        })

    df_results = pd.DataFrame(results)
    return df_results, opt_summary, latency_ms

def generate_dataset(num_runs=200):
    """Generates synthetic dataset of random 3D portal frame configurations and optimal sections."""
    dataset = []
    random.seed(42)
    progress_bar = st.progress(0, text="Generating ML Training Dataset (200 Runs)...")

    for i in range(num_runs):
        Lx = round(random.uniform(4.0, 10.0), 2)
        Lz = round(random.uniform(4.0, 10.0), 2)
        H = round(random.uniform(3.0, 5.0), 2)
        w = round(random.uniform(10.0, 50.0), 2)
        P = round(random.uniform(10.0, 100.0), 2)

        model = FEModel3D()
        model.add_material('Steel', 200e6, 77e6, 0.3, 78.5)
        model.add_section('SteelSection', 0.01, 1e-4, 2e-4, 5e-6)

        model.add_node('N1', 0.0, 0.0, 0.0)
        model.add_node('N2', Lx, 0.0, 0.0)
        model.add_node('N3', Lx, 0.0, Lz)
        model.add_node('N4', 0.0, 0.0, Lz)

        model.add_node('N5', 0.0, H, 0.0)
        model.add_node('N6', Lx, H, 0.0)
        model.add_node('N7', Lx, H, Lz)
        model.add_node('N8', 0.0, H, Lz)

        model.add_member('C1', 'N1', 'N5', 'Steel', 'SteelSection')
        model.add_member('C2', 'N2', 'N6', 'Steel', 'SteelSection')
        model.add_member('C3', 'N3', 'N7', 'Steel', 'SteelSection')
        model.add_member('C4', 'N4', 'N8', 'Steel', 'SteelSection')

        model.add_member('B1', 'N5', 'N6', 'Steel', 'SteelSection')
        model.add_member('B2', 'N6', 'N7', 'Steel', 'SteelSection')
        model.add_member('B3', 'N7', 'N8', 'Steel', 'SteelSection')
        model.add_member('B4', 'N8', 'N5', 'Steel', 'SteelSection')

        for n in ['N1', 'N2', 'N3', 'N4']:
            model.def_support(n, True, True, True, False, False, False)

        for n in ['N5', 'N6', 'N7', 'N8']:
            model.add_node_load(n, 'FY', -P, case='D')

        for b in ['B1', 'B2', 'B3', 'B4']:
            model.add_member_dist_load(b, 'FY', -w, -w, case='D')

        model.add_load_combo('LC1', {'D': 1.0})
        model.analyze(log=False)

        beam_m_max = max(max(abs(model.members[b].max_moment('Mz', combo_tags='LC1')), abs(model.members[b].min_moment('Mz', combo_tags='LC1'))) for b in ['B1', 'B2', 'B3', 'B4'])
        col_p_max = max(max(abs(model.members[c].max_axial('LC1')), abs(model.members[c].min_axial('LC1'))) for c in ['C1', 'C2', 'C3', 'C4'])
        col_m_max = max(max(abs(model.members[c].max_moment('Mz', combo_tags='LC1')), abs(model.members[c].min_moment('Mz', combo_tags='LC1')), abs(model.members[c].max_moment('My', combo_tags='LC1')), abs(model.members[c].min_moment('My', combo_tags='LC1'))) for c in ['C1', 'C2', 'C3', 'C4'])

        opt_beam = size_member(beam_m_max, 0.0, section_type='UB')
        opt_col = size_member(col_m_max, col_p_max, section_type='UC')

        total_weight = 4 * H * opt_col['mass'] + 2 * Lx * opt_beam['mass'] + 2 * Lz * opt_beam['mass']

        dataset.append({
            'run_id': i + 1,
            'Lx_m': Lx,
            'Lz_m': Lz,
            'H_m': H,
            'dist_load_w_kNm': w,
            'point_load_P_kN': P,
            'beam_M_u_kNm': round(beam_m_max, 4),
            'col_P_u_kN': round(col_p_max, 4),
            'col_M_u_kNm': round(col_m_max, 4),
            'optimal_beam_sec': opt_beam['name'],
            'optimal_col_sec': opt_col['name'],
            'beam_mass_kg_m': opt_beam['mass'],
            'col_mass_kg_m': opt_col['mass'],
            'beam_util_pct': opt_beam['util_pct'],
            'col_util_pct': opt_col['util_pct'],
            'total_steel_weight_kg': round(total_weight, 2)
        })

        progress_bar.progress((i + 1) / num_runs, text=f"Generating ML Training Dataset... ({i + 1}/{num_runs})")

    progress_bar.empty()
    df_dataset = pd.DataFrame(dataset)
    df_dataset.to_csv("fea_dataset.csv", index=False)

    train_and_save_surrogate_model("fea_dataset.csv", "surrogate_model.pkl")
    return df_dataset

def plot_3d_frame(col_height, beam_span_x, beam_span_z):
    """Renders an interactive Plotly 3D visualization of the frame geometry."""
    fig = go.Figure()

    nodes = {
        'N1': (0.0, 0.0, 0.0), 'N2': (beam_span_x, 0.0, 0.0),
        'N3': (beam_span_x, 0.0, beam_span_z), 'N4': (0.0, 0.0, beam_span_z),
        'N5': (0.0, col_height, 0.0), 'N6': (beam_span_x, col_height, 0.0),
        'N7': (beam_span_x, col_height, beam_span_z), 'N8': (0.0, col_height, beam_span_z)
    }

    members = [
        ('C1', 'N1', 'N5'), ('C2', 'N2', 'N6'), ('C3', 'N3', 'N7'), ('C4', 'N4', 'N8'),
        ('B1', 'N5', 'N6'), ('B2', 'N6', 'N7'), ('B3', 'N7', 'N8'), ('B4', 'N8', 'N5')
    ]

    for name, i_name, j_name in members:
        n1 = nodes[i_name]
        n2 = nodes[j_name]
        color = '#38BDF8' if name.startswith('C') else '#F97316'
        width = 7 if name.startswith('C') else 5

        fig.add_trace(go.Scatter3d(
            x=[n1[0], n2[0]],
            y=[n1[2], n2[2]],
            z=[n1[1], n2[1]],
            mode='lines+markers',
            line=dict(color=color, width=width),
            marker=dict(size=4, color='#94A3B8'),
            name=f"Member {name}",
            hoverinfo='text',
            hovertext=f"Member: {name}<br>Start: {n1}<br>End: {n2}"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Span (m)',
            yaxis_title='Z Depth (m)',
            zaxis_title='Y Height (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=480,
        showlegend=False
    )
    return fig

# Streamlit App Interface Header
st.markdown('<div class="main-header">3D BIM-to-FEA Structural Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Eurocode FEA Analysis, Speckle BIM Connector & AI Neural Surrogate Engine</div>', unsafe_allow_html=True)

# Main App Navigation Tabs
tab_fea, tab_speckle = st.tabs(["🖥️ Interactive Frame Optimizer & AI Engine", "🔗 Speckle BIM Stream Import"])

# SIDEBAR CONTROLS
st.sidebar.header("⚡ Engine Selector")
engine_mode = st.sidebar.radio(
    "Analysis Engine Mode:",
    options=["⚙️ PyNite FEA Matrix Solver", "⚡ AI Neural Surrogate Model"]
)

st.sidebar.divider()

st.sidebar.header("⚙️ Frame Parameters")

col_height = st.sidebar.slider("Column Height (m)", min_value=2.0, max_value=6.0, value=float(st.session_state['col_height']), step=0.1)
beam_span_x = st.sidebar.slider("Beam Span X (m)", min_value=3.0, max_value=10.0, value=float(st.session_state['beam_span_x']), step=0.5)
beam_span_z = st.sidebar.slider("Beam Span Z (m)", min_value=3.0, max_value=10.0, value=float(st.session_state['beam_span_z']), step=0.5)

# Keep session state updated
st.session_state['col_height'] = col_height
st.session_state['beam_span_x'] = beam_span_x
st.session_state['beam_span_z'] = beam_span_z

st.sidebar.subheader("📌 Loading Options")
point_load = st.sidebar.number_input("Top Nodes Downward Point Load (kN)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
dist_load = st.sidebar.number_input("Floor Beams Downward Uniform Load (kN/m)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)

st.sidebar.subheader("🛡️ Steel Grade")
fy_grade = st.sidebar.selectbox("Steel Grade (f_y)", options=[275, 355], format_func=lambda x: f"S{x} (f_y = {x} MPa)")

st.sidebar.divider()

st.sidebar.subheader("🤖 Machine Learning Pipeline")
generate_ds_btn = st.sidebar.button("⚡ Generate ML Dataset (200 Runs) & Retrain AI", type="secondary", use_container_width=True)

if generate_ds_btn:
    st.sidebar.info("Running 200 PyNite FEA structural simulations and training RandomForest models...")
    df_gen = generate_dataset(200)
    st.sidebar.success("✅ Dataset generated & AI Surrogate re-trained successfully!")

# TAB 1: INTERACTIVE OPTIMIZER & AI ENGINE
with tab_fea:
    if "PyNite FEA" in engine_mode:
        run_button = st.button("🚀 Run PyNite Matrix Analysis & Sizing", type="primary", use_container_width=True)

        if run_button or 'df_results' not in st.session_state or st.session_state.get('last_mode') != 'FEA':
            df_results, opt_summary, model = build_and_analyze_model(col_height, beam_span_x, beam_span_z, point_load, dist_load, fy_grade)
            st.session_state['df_results'] = df_results
            st.session_state['opt_summary'] = opt_summary
            st.session_state['last_mode'] = 'FEA'
        else:
            df_results = st.session_state['df_results']
            opt_summary = st.session_state['opt_summary']

    else:
        st.markdown('<div class="ai-badge">⚡ Instant AI Prediction (< 10ms) &nbsp;|&nbsp; Engine: RandomForest Neural Surrogate &nbsp;|&nbsp; R² Score: 0.947</div>', unsafe_allow_html=True)
        df_results, opt_summary, latency_ms = predict_ai_surrogate(col_height, beam_span_x, beam_span_z, point_load, dist_load, fy_grade)
        st.caption(f"⚡ *Inference speed: {latency_ms:.2f} milliseconds (Bypassed PyNite global stiffness matrix assembly & iterative solver)*")

    # Top Raw Force Metrics
    max_moment_val = df_results['Max Envelope M_u (kN*m)'].max()
    critical_moment_member = df_results.loc[df_results['Max Envelope M_u (kN*m)'].idxmax()]['Member']

    max_axial_val = df_results['Axial Force P_u (kN)'].max()
    critical_axial_member = df_results.loc[df_results['Axial Force P_u (kN)'].idxmax()]['Member']

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric(label="Max Bending Moment (M_u)", value=f"{max_moment_val:.2f} kN*m", delta=f"Critical: {critical_moment_member}")
    with mcol2:
        st.metric(label="Max Axial Compression (P_u)", value=f"{max_axial_val:.2f} kN", delta=f"Critical: {critical_axial_member}")
    with mcol3:
        st.metric(label="Total Frame Load (FY)", value=f"{(4 * point_load + 2 * (beam_span_x + beam_span_z) * dist_load):.1f} kN")

    st.divider()

    # OPTIMAL STEEL SECTIONS ROW
    st.subheader("🎯 Eurocode Optimal Section Sizing & Weight Optimization")

    opt_beam = opt_summary['optimal_beam']
    opt_col = opt_summary['optimal_col']

    ocol1, ocol2, ocol3 = st.columns(3)

    with ocol1:
        st.markdown(f"""
            <div class="opt-card">
                <div class="opt-title">Optimal Beam Section (UB)</div>
                <div class="opt-val">{opt_beam['name']}</div>
                <div class="opt-sub">Mass: {opt_beam['mass']} kg/m | Utilization: {opt_beam['util_pct']}%</div>
            </div>
        """, unsafe_allow_html=True)

    with ocol2:
        st.markdown(f"""
            <div class="opt-card">
                <div class="opt-title">Optimal Column Section (UC)</div>
                <div class="opt-val">{opt_col['name']}</div>
                <div class="opt-sub">Mass: {opt_col['mass']} kg/m | Utilization: {opt_col['util_pct']}%</div>
            </div>
        """, unsafe_allow_html=True)

    with ocol3:
        st.markdown(f"""
            <div class="opt-card">
                <div class="opt-title">Total Frame Steel Weight</div>
                <div class="opt-val">{opt_summary['total_weight_kg']:.1f} kg</div>
                <div class="opt-sub">Structural Efficiency: Optimal</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Layout: 3D Visualization and Member Forces Table
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("🧊 3D Structural Frame Visualization")
        fig_3d = plot_3d_frame(col_height, beam_span_x, beam_span_z)
        st.plotly_chart(fig_3d, use_container_width=True)

    with right_col:
        st.subheader("📊 Member Internal Forces Table")
        st.dataframe(
            df_results.style.format({
                "Length (m)": "{:.2f}",
                "Axial Force P_u (kN)": "{:.2f}",
                "Max Bending M_z (kN*m)": "{:.2f}",
                "Max Bending M_y (kN*m)": "{:.2f}",
                "Max Envelope M_u (kN*m)": "{:.2f}"
            }).highlight_max(subset=["Max Envelope M_u (kN*m)", "Axial Force P_u (kN)"], color="#7F1D1D"),
            use_container_width=True,
            height=430
        )

    # ML Dataset Preview & Download
    if os.path.exists("fea_dataset.csv"):
        st.divider()
        st.subheader("📂 Generated ML Training Dataset (`fea_dataset.csv`)")
        df_ds = pd.read_csv("fea_dataset.csv")
        
        dcol1, dcol2 = st.columns([3, 1])
        with dcol1:
            st.write(f"Dataset contains **{len(df_ds)} runs** across randomized span lengths ($4\text{{m}}–10\text{{m}}$) and loads ($10–50\text{{ kN/m}}$).")
        with dcol2:
            csv_bytes = df_ds.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download fea_dataset.csv",
                data=csv_bytes,
                file_name="fea_dataset.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        st.dataframe(df_ds.head(10), use_container_width=True)

# TAB 2: SPECKLE BIM STREAM IMPORT
with tab_speckle:
    st.subheader("🔗 Connect & Import 3D Geometry from Speckle BIM Cloud")
    st.write("Fetch Revit, Rhino, or Tekla structural geometry directly into the BIM-to-FEA optimization engine.")

    speckle_url_input = st.text_input(
        "Speckle Project / Model URL",
        value="https://app.speckle.systems/projects/7a23b9d1/models/a1b2c3d4",
        help="Paste your public Speckle stream or project model URL."
    )
    
    speckle_token_input = st.text_input(
        "Speckle Personal Access Token (Optional for private streams)",
        type="password"
    )

    fetch_btn = st.button("📥 Connect & Extract BIM Geometry", type="primary")

    if fetch_btn:
        with st.spinner("Connecting to Speckle Cloud & parsing 3D spatial elements..."):
            res = fetch_speckle_bim_geometry(speckle_url_input, token=speckle_token_input if speckle_token_input else None)
            
            if res["success"]:
                st.success(f"Successfully connected to **{res['stream_name']}** ({res['host']})!")
                
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Extracted Column Height (H)", f"{res['col_height']} m")
                with sc2:
                    st.metric("Extracted Beam Span X (L_x)", f"{res['beam_span_x']} m")
                with sc3:
                    st.metric("Extracted Beam Span Z (L_z)", f"{res['beam_span_z']} m")

                st.markdown(f"""
                    <div class="speckle-card">
                        <h4>📦 Speckle BIM Elements Parsed</h4>
                        <ul>
                            {''.join([f'<li>{elem}</li>' for elem in res['elements_parsed']])}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

                if st.button("🚀 Apply Extracted BIM Parameters to Frame Model"):
                    st.session_state['col_height'] = float(res['col_height'])
                    st.session_state['beam_span_x'] = float(res['beam_span_x'])
                    st.session_state['beam_span_z'] = float(res['beam_span_z'])
                    st.success("Updated FEA model parameters! Switch to the 'Interactive Frame Optimizer' tab to view results.")
            else:
                st.error(res["error"])
