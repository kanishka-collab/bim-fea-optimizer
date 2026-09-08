import Pynite
from Pynite import FEModel3D

def main():
    # 1. Initialize 3D FEA Model
    model = FEModel3D()

    # 2. Define Material (Steel: E = 200 GPa = 200e6 kN/m^2, G = 77 GPa = 77e6 kN/m^2, nu = 0.3, rho = 78.5 kN/m^3)
    E = 200e6    # kN/m^2
    G = 77e6     # kN/m^2
    nu = 0.3
    rho = 78.5   # kN/m^3
    model.add_material('Steel', E, G, nu, rho)

    # 3. Define Section Properties (SI units: m^2 and m^4)
    A = 0.01      # Cross-sectional Area (m^2)
    Iz = 2.0e-4   # Strong axis Moment of Inertia (m^4)
    Iy = 1.0e-4   # Weak axis Moment of Inertia (m^4)
    J = 5.0e-6    # Torsional constant (m^4)
    model.add_section('SteelSection', A, Iy, Iz, J)

    # 4. Define Nodes for 3D Portal Frame
    # 4 Base nodes (Y = 0.0 m)
    model.add_node('N1', 0.0, 0.0, 0.0)
    model.add_node('N2', 6.0, 0.0, 0.0)
    model.add_node('N3', 6.0, 0.0, 6.0)
    model.add_node('N4', 0.0, 0.0, 6.0)

    # 4 Top nodes (Y = 3.5 m)
    model.add_node('N5', 0.0, 3.5, 0.0)
    model.add_node('N6', 6.0, 3.5, 0.0)
    model.add_node('N7', 6.0, 3.5, 6.0)
    model.add_node('N8', 0.0, 3.5, 6.0)

    # 5. Add Members (4 Columns of 3.5m height, 4 Beams of 6.0m span)
    # Columns
    model.add_member('C1', 'N1', 'N5', 'Steel', 'SteelSection')
    model.add_member('C2', 'N2', 'N6', 'Steel', 'SteelSection')
    model.add_member('C3', 'N3', 'N7', 'Steel', 'SteelSection')
    model.add_member('C4', 'N4', 'N8', 'Steel', 'SteelSection')

    # Beams
    model.add_member('B1', 'N5', 'N6', 'Steel', 'SteelSection')
    model.add_member('B2', 'N6', 'N7', 'Steel', 'SteelSection')
    model.add_member('B3', 'N7', 'N8', 'Steel', 'SteelSection')
    model.add_member('B4', 'N8', 'N5', 'Steel', 'SteelSection')

    # 6. Apply Pinned Supports at Base Nodes (Restrain DX, DY, DZ; Free RX, RY, RZ)
    base_nodes = ['N1', 'N2', 'N3', 'N4']
    for node in base_nodes:
        model.def_support(node, True, True, True, False, False, False)

    # 7. Apply Downwards Vertical Point Load of 50 kN on Top Nodes
    top_nodes = ['N5', 'N6', 'N7', 'N8']
    for node in top_nodes:
        model.add_node_load(node, 'FY', -50.0, case='D')

    # 8. Apply Downwards Distributed Load of 15 kN/m on all 4 Floor Beams (B1, B2, B3, B4)
    floor_beams = ['B1', 'B2', 'B3', 'B4']
    for beam in floor_beams:
        model.add_member_dist_load(beam, 'FY', -15.0, -15.0, case='D')

    # 9. Add Load Combination
    model.add_load_combo('LC1', {'D': 1.0})

    # 10. Run FEA Analysis
    print("==================================================")
    print("       3D STRUCTURAL STEEL PORTAL FRAME FEA       ")
    print("==================================================")
    print("Analyzing 3D Portal Frame model with distributed beam loads...")
    model.analyze(log=False)
    print("Analysis complete.\n")

    # 11. Summary Results
    print("Member Results Summary (Load Combo LC1):")
    print(f"{'Member':<8} | {'Type':<8} | {'Axial P (kN)':<14} | {'Max Mz (kN*m)':<14} | {'Max My (kN*m)':<14}")
    print("-" * 70)

    max_M_u = 0.0
    critical_member_M = None
    max_P_u = 0.0
    critical_member_P = None

    for name, member in model.members.items():
        mtype = "Column" if name.startswith('C') else "Beam"
        
        # Axial force P
        p_max = abs(member.max_axial('LC1'))
        p_min = abs(member.min_axial('LC1'))
        max_p = max(p_max, p_min)

        # Bending moments
        mz_max = abs(member.max_moment('Mz', combo_tags='LC1'))
        mz_min = abs(member.min_moment('Mz', combo_tags='LC1'))
        my_max = abs(member.max_moment('My', combo_tags='LC1'))
        my_min = abs(member.min_moment('My', combo_tags='LC1'))

        max_m = max(mz_max, mz_min, my_max, my_min)

        print(f"{name:<8} | {mtype:<8} | {max_p:<14.4f} | {max(mz_max, mz_min):<14.4f} | {max(my_max, my_min):<14.4f}")

        if max_m >= max_M_u:
            max_M_u = max_m
            critical_member_M = name

        if max_p >= max_P_u:
            max_P_u = max_p
            critical_member_P = name

    print("-" * 70)
    print(f"\nUPDATED RESULTS OVERVIEW:")
    print(f"  Maximum Bending Moment (M_u) : {max_M_u:.4f} kN*m (Critical Member: {critical_member_M})")
    print(f"  Maximum Axial Force (P_u)    : {max_P_u:.4f} kN (Critical Member: {critical_member_P})")
    print(f"==================================================\n")

if __name__ == '__main__':
    main()
