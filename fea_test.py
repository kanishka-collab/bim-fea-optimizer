"""
Enterprise Multi-Story Multi-Bay Structural FEA Engine (PyNite FEA Kernel)
Supports Eurocode EN 1990/EN 1993 Load Combinations, Flexural Buckling, SLS Deflection, and Top-Story Lateral Sway Drift (delta_sway <= H/500).
"""

import pandas as pd
import Pynite
from Pynite import FEModel3D
from steel_sections import size_member, STEEL_SECTIONS

class MultiStoryFEAEngine:
    def __init__(
        self,
        num_bays_x: int = 1,
        num_bays_z: int = 1,
        num_stories: int = 1,
        bay_width_x: float = 6.0,
        bay_width_z: float = 6.0,
        story_height: float = 3.5,
        G_k: float = 15.0,    # Dead Load (kN/m)
        Q_k: float = 10.0,    # Live Load (kN/m)
        W_k: float = 5.0,     # Wind Load (kN/m)
        fy_MPa: float = 275.0
    ):
        self.num_bays_x = int(max(1, num_bays_x))
        self.num_bays_z = int(max(1, num_bays_z))
        self.num_stories = int(max(1, num_stories))
        self.bay_width_x = float(bay_width_x)
        self.bay_width_z = float(bay_width_z)
        self.story_height = float(story_height)
        self.G_k = float(G_k)
        self.Q_k = float(Q_k)
        self.W_k = float(W_k)
        self.fy_MPa = float(fy_MPa)

        self.model = FEModel3D()
        self.df_results = None
        self.opt_summary = None

    def build_grid_only(self):
        """Builds structural grid nodes and members without analyzing."""
        model = self.model
        E, G, nu, rho = 200e6, 77e6, 0.3, 78.5
        model.add_material('Steel', E, G, nu, rho)
        model.add_section('SteelSection', 0.01, 1e-4, 2e-4, 5e-6)

        for k in range(self.num_stories + 1):
            for i in range(self.num_bays_x + 1):
                for j in range(self.num_bays_z + 1):
                    node_name = f"N_{i}_{k}_{j}"
                    model.add_node(node_name, i * self.bay_width_x, k * self.story_height, j * self.bay_width_z)
                    if k == 0:
                        model.def_support(node_name, True, True, True, False, False, False)

        for k in range(self.num_stories):
            for i in range(self.num_bays_x + 1):
                for j in range(self.num_bays_z + 1):
                    model.add_member(f"C_{i}_{k}_{j}", f"N_{i}_{k}_{j}", f"N_{i}_{k+1}_{j}", 'Steel', 'SteelSection')

        for k in range(1, self.num_stories + 1):
            for i in range(self.num_bays_x):
                for j in range(self.num_bays_z + 1):
                    model.add_member(f"BX_{i}_{k}_{j}", f"N_{i}_{k}_{j}", f"N_{i+1}_{k}_{j}", 'Steel', 'SteelSection')

        for k in range(1, self.num_stories + 1):
            for i in range(self.num_bays_x + 1):
                for j in range(self.num_bays_z):
                    model.add_member(f"BZ_{i}_{k}_{j}", f"N_{i}_{k}_{j}", f"N_{i}_{k}_{j+1}", 'Steel', 'SteelSection')

    def build_and_analyze(self):
        """Builds multi-story structural grid model, applies Eurocode load combinations, and solves."""
        model = self.model

        # 1. Define Material & Default Section
        E = 200e6    # kN/m^2 (200 GPa)
        G = 77e6     # kN/m^2 (77 GPa)
        nu = 0.3
        rho = 78.5   # kN/m^3
        model.add_material('Steel', E, G, nu, rho)

        A = 0.01      # m^2
        Iz = 2.0e-4   # m^4
        Iy = 1.0e-4   # m^4
        J = 5.0e-6    # m^4
        model.add_section('SteelSection', A, Iy, Iz, J)

        # 2. Build 3D Node Grid
        for k in range(self.num_stories + 1):
            for i in range(self.num_bays_x + 1):
                for j in range(self.num_bays_z + 1):
                    node_name = f"N_{i}_{k}_{j}"
                    x_pos = i * self.bay_width_x
                    y_pos = k * self.story_height
                    z_pos = j * self.bay_width_z
                    model.add_node(node_name, x_pos, y_pos, z_pos)

                    # Pinned supports at base level (k = 0)
                    if k == 0:
                        model.def_support(node_name, True, True, True, False, False, False)

        # 3. Add Columns (Vertical)
        for k in range(self.num_stories):
            for i in range(self.num_bays_x + 1):
                for j in range(self.num_bays_z + 1):
                    col_name = f"C_{i}_{k}_{j}"
                    n1 = f"N_{i}_{k}_{j}"
                    n2 = f"N_{i}_{k+1}_{j}"
                    model.add_member(col_name, n1, n2, 'Steel', 'SteelSection')

        # 4. Add Beams X (Horizontal X-Direction)
        for k in range(1, self.num_stories + 1):
            for i in range(self.num_bays_x):
                for j in range(self.num_bays_z + 1):
                    beam_name = f"BX_{i}_{k}_{j}"
                    n1 = f"N_{i}_{k}_{j}"
                    n2 = f"N_{i+1}_{k}_{j}"
                    model.add_member(beam_name, n1, n2, 'Steel', 'SteelSection')

        # 5. Add Beams Z (Horizontal Z-Direction)
        for k in range(1, self.num_stories + 1):
            for i in range(self.num_bays_x + 1):
                for j in range(self.num_bays_z):
                    beam_name = f"BZ_{i}_{k}_{j}"
                    n1 = f"N_{i}_{k}_{j}"
                    n2 = f"N_{i}_{k}_{j+1}"
                    model.add_member(beam_name, n1, n2, 'Steel', 'SteelSection')

        # 6. Apply Distributed Loads
        # Floor Beams (Dead G_k and Live Q_k)
        for m_name, member in model.members.items():
            if m_name.startswith('B'):
                if self.G_k > 0:
                    model.add_member_dist_load(m_name, 'FY', -abs(self.G_k), -abs(self.G_k), case='G')
                if self.Q_k > 0:
                    model.add_member_dist_load(m_name, 'FY', -abs(self.Q_k), -abs(self.Q_k), case='Q')

        # Windward Columns (Wind W_k in X-Direction at i = 0)
        if self.W_k > 0:
            for k in range(self.num_stories):
                for j in range(self.num_bays_z + 1):
                    col_name = f"C_0_{k}_{j}"
                    if col_name in model.members:
                        model.add_member_dist_load(col_name, 'FX', abs(self.W_k), abs(self.W_k), case='W')

        # 7. Add Eurocode Load Combinations
        model.add_load_combo('LC_ULS1', {'G': 1.35, 'Q': 1.50, 'W': 0.90})
        model.add_load_combo('LC_ULS2', {'G': 1.00, 'Q': 1.05, 'W': 1.50})
        model.add_load_combo('LC_SLS', {'G': 1.00, 'Q': 1.00, 'W': 1.00})

        # 8. Execute Solver
        model.analyze(log=False)

        # 9. Extract Top-Story Lateral Sway Drift (delta_sway <= H / 500)
        top_k = self.num_stories
        total_height_m = top_k * self.story_height
        sway_limit_mm = (total_height_m * 1000.0) / 500.0

        max_dx_m = 0.0
        for i in range(self.num_bays_x + 1):
            for j in range(self.num_bays_z + 1):
                node_name = f"N_{i}_{top_k}_{j}"
                if node_name in model.nodes:
                    dx = abs(model.nodes[node_name].DX.get('LC_SLS', 0.0))
                    if dx > max_dx_m:
                        max_dx_m = dx

        delta_sway_mm = max_dx_m * 1000.0
        util_sway = delta_sway_mm / sway_limit_mm if sway_limit_mm > 0 else 0.0
        sway_status = "Pass" if util_sway <= 1.0 else "Fail"

        # 10. Extract Member Forces & Optimal Eurocode 3 Sizing
        results = []
        max_beam_M = 0.0
        max_col_P = 0.0
        max_col_M = 0.0

        for name, member in model.members.items():
            mtype = "Column" if name.startswith('C') else "Beam"
            L_m = member.L()

            # Envelope across ULS combinations
            p_uls1 = max(abs(member.max_axial('LC_ULS1')), abs(member.min_axial('LC_ULS1')))
            p_uls2 = max(abs(member.max_axial('LC_ULS2')), abs(member.min_axial('LC_ULS2')))
            max_p = max(p_uls1, p_uls2)

            mz_uls1 = max(abs(member.max_moment('Mz', combo_tags='LC_ULS1')), abs(member.min_moment('Mz', combo_tags='LC_ULS1')))
            mz_uls2 = max(abs(member.max_moment('Mz', combo_tags='LC_ULS2')), abs(member.min_moment('Mz', combo_tags='LC_ULS2')))
            max_mz = max(mz_uls1, mz_uls2)

            my_uls1 = max(abs(member.max_moment('My', combo_tags='LC_ULS1')), abs(member.min_moment('My', combo_tags='LC_ULS1')))
            my_uls2 = max(abs(member.max_moment('My', combo_tags='LC_ULS2')), abs(member.min_moment('My', combo_tags='LC_ULS2')))
            max_my = max(my_uls1, my_uls2)

            max_m = max(max_mz, max_my)

            if mtype == "Beam":
                if max_m > max_beam_M:
                    max_beam_M = max_m
            else:
                if max_p > max_col_P:
                    max_col_P = max_p
                if max_m > max_col_M:
                    max_col_M = max_m

            # Perform section sizing
            sec_type = 'UC' if mtype == 'Column' else 'UB'
            total_load_w = self.G_k + self.Q_k if mtype == 'Beam' else 0.0
            opt_sec = size_member(max_m, max_p, L_m=L_m, w_kNm=total_load_w, yield_strength_MPa=self.fy_MPa, member_type=mtype, section_type=sec_type)

            util_pct = opt_sec['util_pct']
            status_color = "Green" if util_pct < 70.0 else ("Yellow" if util_pct <= 100.0 else "Red")

            results.append({
                "Member": name,
                "Type": mtype,
                "Length (m)": round(L_m, 2),
                "Axial Force P_u (kN)": round(max_p, 2),
                "Max Bending M_z (kN*m)": round(max_mz, 2),
                "Max Bending M_y (kN*m)": round(max_my, 2),
                "Max Envelope M_u (kN*m)": round(max_m, 2),
                "Assigned Section": opt_sec['name'],
                "Mass (kg/m)": opt_sec['mass'],
                "Util (%)": util_pct,
                "Status": status_color
            })

        df_results = pd.DataFrame(results)

        # Global optimal sections
        total_w_kNm = self.G_k + self.Q_k
        optimal_beam = size_member(max_beam_M, 0.0, L_m=self.bay_width_x, w_kNm=total_w_kNm, yield_strength_MPa=self.fy_MPa, member_type='beam', section_type='UB')
        optimal_col = size_member(max_col_M, max_col_P, L_m=self.story_height, yield_strength_MPa=self.fy_MPa, member_type='column', section_type='UC')

        # Total Weight calculation
        total_col_length = (self.num_bays_x + 1) * (self.num_bays_z + 1) * self.num_stories * self.story_height
        total_beam_x_length = self.num_bays_x * (self.num_bays_z + 1) * self.num_stories * self.bay_width_x
        total_beam_z_length = (self.num_bays_x + 1) * self.num_bays_z * self.num_stories * self.bay_width_z

        total_weight_kg = (total_col_length * optimal_col['mass']) + ((total_beam_x_length + total_beam_z_length) * optimal_beam['mass'])

        self.df_results = df_results
        self.opt_summary = {
            "optimal_beam": optimal_beam,
            "optimal_col": optimal_col,
            "total_weight_kg": round(total_weight_kg, 2),
            "max_beam_M": round(max_beam_M, 2),
            "max_col_P": round(max_col_P, 2),
            "total_nodes": len(model.nodes),
            "total_members": len(model.members),
            "total_height_m": round(total_height_m, 2),
            "delta_sway_mm": round(delta_sway_mm, 2),
            "delta_sway_lim_mm": round(sway_limit_mm, 2),
            "util_sway": round(util_sway, 4),
            "sway_status": sway_status
        }

        return df_results, self.opt_summary, model

def main():
    print("Executing Enterprise MultiStoryFEAEngine Test with Lateral Sway Drift...")
    engine = MultiStoryFEAEngine(
        num_bays_x=2,
        num_bays_z=1,
        num_stories=3,
        bay_width_x=6.0,
        bay_width_z=5.0,
        story_height=3.5,
        G_k=15.0,
        Q_k=10.0,
        W_k=12.0
    )
    df, opt, model = engine.build_and_analyze()

    print("\n==================================================")
    print("      ENTERPRISE MULTI-STORY STRUCTURAL FEA       ")
    print("==================================================")
    print(f"Total Nodes: {opt['total_nodes']} | Total Members: {opt['total_members']}")
    print(f"Total Height: {opt['total_height_m']} m")
    print(f"Top-Story Sway Drift: {opt['delta_sway_mm']} mm (Limit H/500 = {opt['delta_sway_lim_mm']} mm) -> Status: {opt['sway_status']}")
    print(f"Optimal Beam Section (UB) : {opt['optimal_beam']['name']} ({opt['optimal_beam']['mass']} kg/m)")
    print(f"Optimal Column Section (UC): {opt['optimal_col']['name']} ({opt['optimal_col']['mass']} kg/m)")
    print(f"Total Steel Weight         : {opt['total_weight_kg']:.2f} kg")
    print("==================================================\n")

if __name__ == '__main__':
    main()
