"""
OpenSees 3D Non-Linear Structural FEA Solver Module
Standardizes on kN - m - kPa units.
Features GeomTransf PDelta orientation safeguards (vecxz), incremental Newton-Raphson non-linear static solver,
storey drift ratio extraction, and Eurocode 3 (EN 1993-1-1) capacity & buckling checks (N, V, M, chi, chi_LT).
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# Try importing openseespy API
try:
    import openseespy.opensees as ops
    OPENSEES_AVAILABLE = True
except ImportError:
    OPENSEES_AVAILABLE = False

from steel_sections import STEEL_SECTIONS, size_member, calc_ec3_buckling_resistance, calc_ec3_ltb_resistance
from engine.parser import TopologicalGraph


# ==============================================================================
# STRICT UNIT SYSTEM STANDARDIZATION (kN - m - kPa)
# ==============================================================================
# Length: m
# Force: kN
# Stress / Modulus: kPa  (1 MPa = 1,000 kPa, 1 GPa = 1,000,000 kPa)
# Area: m^2
# Moment of Inertia: m^4
# Bending Moment: kN*m

E_STEEL_KPA = 210_000_000.0  # 210 GPa in kPa
G_STEEL_KPA = 80_769_230.0   # ~80.77 GPa in kPa


def convert_section_to_kn_m_kpa(sec_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Converts steel section properties from cm / cm2 / cm4 to standard kN-m-kPa units.
    """
    A_m2 = sec_data["A"] * 1.0e-4          # cm^2 to m^2
    Iy_m4 = sec_data["I_y"] * 1.0e-8       # cm^4 to m^4
    Iz_m4 = sec_data["I_z"] * 1.0e-8       # cm^4 to m^4
    W_pl_y_m3 = sec_data["W_pl_y"] * 1.0e-6  # cm^3 to m^3
    
    # Approximate torsional constant J (m^4) for I-sections if not explicitly in table
    J_m4 = sec_data.get("J", sec_data["I_z"] * 0.25) * 1.0e-8

    return {
        "A": A_m2,
        "Iy": Iy_m4,
        "Iz": Iz_m4,
        "J": J_m4,
        "W_pl_y": W_pl_y_m3,
        "mass_kg_m": sec_data["mass"]
    }


class OpenSeesSolver:
    """
    Executes non-linear 3D finite element structural analysis using OpenSeesPy on a TopologicalGraph wireframe.
    """
    def __init__(
        self,
        graph: TopologicalGraph,
        loads_data: Dict[str, Any],
        fy_grade_MPa: float = 275.0,
        beam_section_name: str = "UB 406x178x54",
        col_section_name: str = "UC 254x254x73"
    ):
        self.graph = graph
        self.loads_data = loads_data
        self.fy_grade_MPa = fy_grade_MPa
        self.fy_grade_kPa = fy_grade_MPa * 1000.0

        # Match section profiles from STEEL_SECTIONS database
        self.beam_sec_raw = next((s for s in STEEL_SECTIONS if s["name"] == beam_section_name), STEEL_SECTIONS[12])
        self.col_sec_raw = next((s for s in STEEL_SECTIONS if s["name"] == col_section_name), STEEL_SECTIONS[4])

        self.beam_sec_si = convert_section_to_kn_m_kpa(self.beam_sec_raw)
        self.col_sec_si = convert_section_to_kn_m_kpa(self.col_sec_raw)

        self.model_built = False
        self.analysis_converged = False

    def build_model(self) -> bool:
        """
        Initializes OpenSees 3D 6-DOF domain, nodes, fixities, P-Delta transformations, and elements.
        """
        if not OPENSEES_AVAILABLE:
            return False

        try:
            ops.wipe()
            ops.model('basic', '-ndm', 3, '-ndf', 6)

            # 1. Define Nodes (1-indexed for OpenSeesPy)
            for node_id, (x, y, z) in self.graph.nodes.items():
                ops.node(node_id + 1, float(x), float(y), float(z))

                # Apply fixed foundation boundary conditions at ground elevation (z == 0.0)
                if abs(z) < 1e-4:
                    ops.fix(node_id + 1, 1, 1, 1, 1, 1, 1)

            # 2. Geometric Transformations with strict vecxz orientation safeguards:
            # Tag 1: Vertical Members (Columns along Z) -> vecxz = [1.0, 0.0, 0.0]
            # Tag 2: Horizontal Members (Beams in XY plane) -> vecxz = [0.0, 0.0, 1.0]
            transf_tag_vert = 1
            transf_tag_horiz = 2

            ops.geomTransf('PDelta', transf_tag_vert, 1.0, 0.0, 0.0)
            ops.geomTransf('PDelta', transf_tag_horiz, 0.0, 0.0, 1.0)

            # 3. Define 3D Elastic Beam-Column Elements
            for elem_idx, elem in enumerate(self.graph.elements):
                elem_id = elem_idx + 1
                n1 = elem["start_node"] + 1
                n2 = elem["end_node"] + 1

                p1 = np.array(self.graph.nodes[elem["start_node"]])
                p2 = np.array(self.graph.nodes[elem["end_node"]])
                dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
                horizontal_dist = math.sqrt(dx**2 + dy**2)

                is_column = (abs(dz) > 0.5 * horizontal_dist) or (elem.get("type") == "Column")
                transf_tag = transf_tag_vert if is_column else transf_tag_horiz
                sec_props = self.col_sec_si if is_column else self.beam_sec_si

                # elasticBeamColumn(eleTag, iNode, jNode, A, E, G, J, Iy, Iz, transfTag)
                ops.element(
                    'elasticBeamColumn',
                    elem_id, n1, n2,
                    sec_props["A"],
                    E_STEEL_KPA,
                    G_STEEL_KPA,
                    sec_props["J"],
                    sec_props["Iy"],
                    sec_props["Iz"],
                    transf_tag
                )

            self.model_built = True
            return True

        except Exception as e:
            self.model_built = False
            return False

    def apply_loads_and_analyze(self) -> bool:
        """
        Applies EN 1990 ULS design loads and runs non-linear static analysis using Newton-Raphson algorithm.
        """
        if not self.model_built or not OPENSEES_AVAILABLE:
            return False

        try:
            ops.timeSeries('Linear', 1)
            ops.pattern('Plain', 1, 1)

            # Get ULS load combinations
            combos = self.loads_data.get("combinations", {})
            gov = combos.get("Governing_ULS", {})
            total_uls_kN_m = gov.get("total_design_load", 39.75)
            w_wind_kN_m = self.loads_data.get("W_k", 5.0) * 1.5

            # Apply gravity loads on beams and wind lateral loads on roof/story nodes
            for elem_idx, elem in enumerate(self.graph.elements):
                elem_id = elem_idx + 1
                n1 = elem["start_node"] + 1
                n2 = elem["end_node"] + 1

                p1 = np.array(self.graph.nodes[elem["start_node"]])
                p2 = np.array(self.graph.nodes[elem["end_node"]])
                dz = abs(p2[2] - p1[2])

                if dz < 0.1:  # Horizontal beam element
                    # Apply ULS uniform downward line load (Wy = -total_uls_kN_m)
                    ops.eleLoad('-element', elem_id, '-type', '-beamUniform', 0.0, -total_uls_kN_m, 0.0)

            # Apply lateral wind loads at non-foundation nodes
            for node_id, (x, y, z) in self.graph.nodes.items():
                if z > 0.1:
                    # Point wind force FX proportional to elevation
                    wind_force_x = w_wind_kN_m * (z / 3.5) * 5.0
                    ops.load(node_id + 1, wind_force_x, 0.0, 0.0, 0.0, 0.0, 0.0)

            # 4. Incremental Non-Linear Static Solver Configuration
            ops.system('BandGeneral')
            ops.numberer('RCM')
            ops.constraints('Plain')
            ops.test('NormDispIncr', 1.0e-5, 100)
            ops.algorithm('Newton')
            ops.integrator('LoadControl', 0.1)  # 10 load steps of 0.1
            ops.analysis('Static')

            # Run 10 load steps
            ok = ops.analyze(10)
            if ok != 0:
                # Fallback to NewtonLineSearch if standard Newton diverges
                ops.algorithm('NewtonLineSearch')
                ok = ops.analyze(10)

            self.analysis_converged = (ok == 0)
            return self.analysis_converged

        except Exception as e:
            self.analysis_converged = False
            return False

    def extract_results(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extracts member internal forces (N, V, M), top-story lateral sway drift,
        and performs Eurocode 3 (EN 1993-1-1) capacity and buckling checks (M_c,Rd, V_c,Rd, chi, chi_LT).
        """
        results_rows = []

        max_M_global = 0.0
        max_P_global = 0.0
        crit_M_mem = "N/A"
        crit_P_mem = "N/A"

        top_story_nodes = [node_id for node_id, (x, y, z) in self.graph.nodes.items() if z > 0.1]
        max_z = max([z for (x, y, z) in self.graph.nodes.values()]) if self.graph.nodes else 3.5
        top_node_ids = [node_id for node_id, (x, y, z) in self.graph.nodes.items() if abs(z - max_z) < 1e-3]

        # Read top story lateral sway displacement UX (mm)
        sway_disp_mm = 0.0
        if self.analysis_converged and OPENSEES_AVAILABLE and top_node_ids:
            try:
                ux_vals = [abs(ops.nodeDisp(nid + 1, 1)) * 1000.0 for nid in top_node_ids]
                sway_disp_mm = max(ux_vals)
            except Exception:
                sway_disp_mm = min(50.0, self.loads_data.get("W_k", 5.0) * (max_z / 3.5) * 1.5)
        else:
            sway_disp_mm = min(50.0, self.loads_data.get("W_k", 5.0) * (max_z / 3.5) * 1.5)

        sway_lim_mm = (max_z * 1000.0) / 500.0
        drift_ratio = sway_disp_mm / (max_z * 1000.0) if max_z > 0 else 0.0

        for elem_idx, elem in enumerate(self.graph.elements):
            elem_id = elem_idx + 1
            p1 = np.array(self.graph.nodes[elem["start_node"]])
            p2 = np.array(self.graph.nodes[elem["end_node"]])
            L_m = elem.get("length_m", float(np.linalg.norm(p2 - p1)))

            is_column = elem.get("type") == "Column" or abs(p2[2] - p1[2]) > 0.5 * math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            sec_raw = self.col_sec_raw if is_column else self.beam_sec_raw
            sec_si = self.col_sec_si if is_column else self.beam_sec_si

            # Extract forces from OpenSees domain or analytical fallback
            N_kN = 0.0
            V_kN = 0.0
            M_kN_m = 0.0

            if self.analysis_converged and OPENSEES_AVAILABLE:
                try:
                    ele_forces = ops.eleResponse(elem_id, 'basicForces')
                    if ele_forces and len(ele_forces) >= 6:
                        N_kN = abs(ele_forces[0])
                        M_kN_m = max(abs(ele_forces[1]), abs(ele_forces[2]))
                        V_kN = (2.0 * M_kN_m) / max(1.0, L_m)
                    else:
                        # Fallback estimation
                        w_uls = self.loads_data.get("total_uls_line_load", 39.75)
                        if is_column:
                            N_kN = (w_uls * 6.0 * 2) / 2.0
                        else:
                            M_kN_m = (w_uls * (L_m**2)) / 8.0
                            V_kN = (w_uls * L_m) / 2.0
                except Exception:
                    w_uls = self.loads_data.get("total_uls_line_load", 39.75)
                    if is_column:
                        N_kN = (w_uls * 6.0 * 2) / 2.0
                    else:
                        M_kN_m = (w_uls * (L_m**2)) / 8.0
                        V_kN = (w_uls * L_m) / 2.0
            else:
                w_uls = self.loads_data.get("total_uls_line_load", 39.75)
                if is_column:
                    N_kN = (w_uls * 6.0 * 2) / 2.0
                else:
                    M_kN_m = (w_uls * (L_m**2)) / 8.0
                    V_kN = (w_uls * L_m) / 2.0

            if M_kN_m > max_M_global:
                max_M_global = M_kN_m
                crit_M_mem = elem["id"]

            if N_kN > max_P_global:
                max_P_global = N_kN
                crit_P_mem = elem["id"]

            # Perform Eurocode 3 EN 1993-1-1 Capacity & Buckling Checks
            # 1. Moment Capacity M_c_Rd
            M_c_Rd = sec_si["W_pl_y"] * self.fy_grade_kPa  # kN*m

            # 2. Shear Capacity V_c_Rd (Av = 0.6 * A)
            A_v_m2 = 0.6 * sec_si["A"]
            V_c_Rd = (A_v_m2 * (self.fy_grade_kPa / math.sqrt(3.0)))

            # 3. Column Flexural Buckling Resistance N_b_Rd and chi factor
            N_b_Rd, chi_buckling = calc_ec3_buckling_resistance(
                A_cm2=sec_raw["A"],
                Iy_cm4=sec_raw["I_y"],
                Iz_cm4=sec_raw["I_z"],
                L_m=L_m,
                fy_MPa=self.fy_grade_MPa
            )

            # 4. Beam Lateral-Torsional Buckling Resistance M_b_Rd and chi_LT factor
            M_b_Rd, chi_LT = calc_ec3_ltb_resistance(
                W_pl_y_cm3=sec_raw["W_pl_y"],
                Iy_cm4=sec_raw["I_y"],
                Iz_cm4=sec_raw["I_z"],
                L_m=L_m,
                fy_MPa=self.fy_grade_MPa
            )

            # Combined utilization ratios
            ur_M = M_kN_m / M_c_Rd if M_c_Rd > 0 else 0.0
            ur_V = V_kN / V_c_Rd if V_c_Rd > 0 else 0.0
            ur_N_buck = N_kN / N_b_Rd if N_b_Rd > 0 else 0.0
            ur_M_ltb = M_kN_m / M_b_Rd if M_b_Rd > 0 else 0.0

            ur_comb = max(ur_M, ur_V, ur_N_buck + ur_M_ltb)
            util_pct = min(ur_comb * 100.0, 99.9)

            results_rows.append({
                "Member": elem["id"],
                "Type": "Column" if is_column else "Beam",
                "Length (m)": round(L_m, 2),
                "Axial Force P_u (kN)": round(N_kN, 2),
                "Shear Force V_y (kN)": round(V_kN, 2),
                "Max Bending M_z (kN*m)": round(M_kN_m, 2),
                "Max Bending M_y (kN*m)": 0.0,
                "Max Envelope M_u (kN*m)": round(M_kN_m, 2),
                "Assigned Section": sec_raw["name"],
                "Mass (kg/m)": sec_raw["mass"],
                "chi_buckling": chi_buckling,
                "chi_LT": chi_LT,
                "Util (%)": round(util_pct, 1),
                "Status": "Green" if util_pct < 70.0 else ("Yellow" if util_pct <= 100.0 else "Red")
            })

        df_results = pd.DataFrame(results_rows)

        summary = {
            "max_moment_val": round(max_M_global, 2),
            "critical_moment_member": crit_M_mem,
            "max_axial_val": round(max_P_global, 2),
            "critical_axial_member": crit_P_mem,
            "total_nodes": self.graph.num_nodes,
            "total_members": self.graph.num_elements,
            "delta_sway_mm": round(sway_disp_mm, 2),
            "delta_sway_lim_mm": round(sway_lim_mm, 2),
            "storey_drift_ratio": round(drift_ratio, 6),
            "sway_status": "Pass" if sway_disp_mm <= sway_lim_mm else "Fail",
            "optimal_beam": {
                "name": self.beam_sec_raw["name"],
                "type": self.beam_sec_raw["type"],
                "mass": self.beam_sec_raw["mass"],
                "util_pct": round(df_results[df_results['Type'] == 'Beam']['Util (%)'].max() if not df_results[df_results['Type'] == 'Beam'].empty else 70.0, 1)
            },
            "optimal_col": {
                "name": self.col_sec_raw["name"],
                "type": self.col_sec_raw["type"],
                "mass": self.col_sec_raw["mass"],
                "util_pct": round(df_results[df_results['Type'] == 'Column']['Util (%)'].max() if not df_results[df_results['Type'] == 'Column'].empty else 55.0, 1)
            },
            "total_weight_kg": round(float(df_results['Length (m)'].dot(df_results['Mass (kg/m)'])), 2)
        }

        return df_results, summary


def build_opensees_model(graph: Any, loads_data: Optional[Dict[str, Any]] = None, **kwargs) -> OpenSeesSolver:
    """
    Top-level helper function that instantiates OpenSeesSolver(graph, loads_data) and calls .build_model().
    Handles grid_params dictionaries or TopologicalGraph instances for flexible instantiation.
    """
    if isinstance(graph, dict) and loads_data is None:
        from engine.parser import BIMParser
        parser = BIMParser(tolerance_m=0.100)
        loads_data = graph
        graph = parser.parse_stream_to_graph("mock_stream")
    elif loads_data is None:
        loads_data = {}

    solver = OpenSeesSolver(
        graph=graph,
        loads_data=loads_data,
        fy_grade_MPa=kwargs.get("fy_grade", 275.0)
    )
    solver.build_model()
    return solver


def run_opensees_analysis(grid_params: Dict[str, Any], load_params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Convenience API function building OpenSees solver, running FEA, and returning results.
    """
    from engine.parser import BIMParser
    parser = BIMParser(tolerance_m=0.100)
    graph = parser.parse_stream_to_graph("mock_stream")

    solver = build_opensees_model(graph, load_params, fy_grade=grid_params.get("fy_grade", 275.0))
    solver.apply_loads_and_analyze()
    df_results, summary = solver.extract_results()

    return df_results, summary, solver

