"""
Database of Standard UK/European Structural Steel Sections (UB and UC)
Includes Eurocode 3 (EN 1993-1-1) Flexural Buckling Checks and SLS Deflection Verification.
"""

import math

STEEL_SECTIONS = [
    # Universal Columns (UC)
    {"name": "UC 152x152x23", "type": "UC", "mass": 23.0, "A": 29.2, "W_pl_y": 182.0, "I_y": 1250.0, "I_z": 400.0},
    {"name": "UC 152x152x30", "type": "UC", "mass": 30.0, "A": 38.3, "W_pl_y": 247.0, "I_y": 1750.0, "I_z": 560.0},
    {"name": "UC 203x203x46", "type": "UC", "mass": 46.1, "A": 58.7, "W_pl_y": 497.0, "I_y": 4570.0, "I_z": 1550.0},
    {"name": "UC 203x203x60", "type": "UC", "mass": 60.0, "A": 76.4, "W_pl_y": 656.0, "I_y": 6120.0, "I_z": 2060.0},
    {"name": "UC 254x254x73", "type": "UC", "mass": 73.1, "A": 93.1, "W_pl_y": 992.0, "I_y": 11400.0, "I_z": 3910.0},
    {"name": "UC 254x254x89", "type": "UC", "mass": 89.0, "A": 113.0, "W_pl_y": 1230.0, "I_y": 14300.0, "I_z": 4860.0},
    {"name": "UC 305x305x97", "type": "UC", "mass": 96.9, "A": 123.0, "W_pl_y": 1590.0, "I_y": 22200.0, "I_z": 7310.0},
    {"name": "UC 305x305x118", "type": "UC", "mass": 118.0, "A": 150.0, "W_pl_y": 1960.0, "I_y": 27600.0, "I_z": 9070.0},

    # Universal Beams (UB)
    {"name": "UB 203x133x25", "type": "UB", "mass": 25.1, "A": 32.0, "W_pl_y": 262.0, "I_y": 2340.0, "I_z": 308.0},
    {"name": "UB 254x146x31", "type": "UB", "mass": 31.1, "A": 39.7, "W_pl_y": 396.0, "I_y": 4410.0, "I_z": 448.0},
    {"name": "UB 305x165x40", "type": "UB", "mass": 40.5, "A": 51.6, "W_pl_y": 625.0, "I_y": 8500.0, "I_z": 764.0},
    {"name": "UB 356x171x45", "type": "UB", "mass": 45.0, "A": 57.3, "W_pl_y": 775.0, "I_y": 12100.0, "I_z": 811.0},
    {"name": "UB 406x178x54", "type": "UB", "mass": 54.3, "A": 69.2, "W_pl_y": 1090.0, "I_y": 18700.0, "I_z": 880.0},
    {"name": "UB 457x191x67", "type": "UB", "mass": 67.1, "A": 85.5, "W_pl_y": 1450.0, "I_y": 29400.0, "I_z": 1450.0},
    {"name": "UB 533x210x82", "type": "UB", "mass": 82.2, "A": 105.0, "W_pl_y": 2060.0, "I_y": 47500.0, "I_z": 2010.0},
    {"name": "UB 610x229x101", "type": "UB", "mass": 101.2, "A": 129.0, "W_pl_y": 2880.0, "I_y": 75700.0, "I_z": 2770.0},
]

def calc_ec3_buckling_resistance(A_cm2: float, Iy_cm4: float, Iz_cm4: float, L_m: float, fy_MPa: float = 275.0, E_GPa: float = 200.0):
    """
    Calculates Eurocode 3 (EN 1993-1-1 §6.3.1.2) flexural buckling resistance N_b_Rd (kN)
    and reduction factor chi.
    """
    A = A_cm2 * 1.0e-4          # m^2
    Iy = Iy_cm4 * 1.0e-8        # m^4
    Iz = Iz_cm4 * 1.0e-8        # m^4
    fy = fy_MPa * 1.0e6         # N/m^2
    E = E_GPa * 1.0e9           # N/m^2

    iy = math.sqrt(Iy / A)
    iz = math.sqrt(Iz / A)

    eps = math.sqrt(235.0 / fy_MPa)
    lambda_1 = 93.9 * eps

    lambda_bar_y = (L_m / iy) / lambda_1
    lambda_bar_z = (L_m / iz) / lambda_1

    # Imperfection factors: Curve b (alpha=0.34) strong axis, Curve c (alpha=0.49) weak axis
    alpha_y, alpha_z = 0.34, 0.49

    phi_y = 0.5 * (1.0 + alpha_y * (lambda_bar_y - 0.2) + lambda_bar_y**2)
    phi_z = 0.5 * (1.0 + alpha_z * (lambda_bar_z - 0.2) + lambda_bar_z**2)

    chi_y = min(1.0, 1.0 / (phi_y + math.sqrt(max(0.0001, phi_y**2 - lambda_bar_y**2))))
    chi_z = min(1.0, 1.0 / (phi_z + math.sqrt(max(0.0001, phi_z**2 - lambda_bar_z**2))))

    chi = min(chi_y, chi_z)
    N_b_Rd_kN = (chi * A * fy) / 1000.0
    return round(N_b_Rd_kN, 2), round(chi, 4)

def calc_ec3_ltb_resistance(W_pl_y_cm3: float, Iy_cm4: float, Iz_cm4: float, L_m: float, fy_MPa: float = 275.0, E_GPa: float = 200.0):
    """
    Calculates Eurocode 3 (EN 1993-1-1 §6.3.2) Lateral-Torsional Buckling (LTB) moment resistance M_b_Rd (kN*m)
    and reduction factor chi_LT.
    """
    W_pl = W_pl_y_cm3 * 1.0e-6    # m^3
    Iz = Iz_cm4 * 1.0e-8          # m^4
    Iy = Iy_cm4 * 1.0e-8          # m^4
    fy = fy_MPa * 1.0e6           # N/m^2
    E = E_GPa * 1.0e9             # N/m^2

    # Elastic Critical Moment Mcr approximation (C1 = 1.13 for ULS gravity envelope)
    C1 = 1.13
    # Effective torsional length ratio approximation
    M_cr_Nm = C1 * ((math.pi**2 * E * Iz) / (L_m**2)) * math.sqrt(max(1.0, Iy / Iz)) * 0.45
    M_cr_kNm = max(1.0, M_cr_Nm / 1000.0)

    M_Rd_kNm = (W_pl * fy) / 1000.0

    lambda_bar_LT = math.sqrt(M_Rd_kNm / M_cr_kNm)

    if lambda_bar_LT <= 0.4:
        chi_LT = 1.0
    else:
        alpha_LT = 0.34  # Rolled section curve b
        phi_LT = 0.5 * (1.0 + alpha_LT * (lambda_bar_LT - 0.2) + lambda_bar_LT**2)
        chi_LT = min(1.0, 1.0 / (phi_LT + math.sqrt(max(0.0001, phi_LT**2 - lambda_bar_LT**2))))

    M_b_Rd_kNm = chi_LT * M_Rd_kNm
    return round(M_b_Rd_kNm, 2), round(chi_LT, 4)

def calc_sls_deflection(w_kNm: float, L_m: float, Iy_cm4: float, E_GPa: float = 200.0):
    """
    Calculates max SLS midspan deflection delta_max (mm) for a beam under uniform load w
    and compares against Eurocode limit delta_lim = L / 360 (mm).
    """
    w_Nm = abs(w_kNm) * 1000.0
    Iy = Iy_cm4 * 1.0e-8
    E = E_GPa * 1.0e9

    delta_max_m = (5.0 * w_Nm * (L_m**4)) / (384.0 * E * Iy)
    delta_max_mm = delta_max_m * 1000.0

    delta_lim_mm = (L_m * 1000.0) / 360.0
    util_def = delta_max_mm / delta_lim_mm if delta_lim_mm > 0 else 0.0

    return round(delta_max_mm, 2), round(delta_lim_mm, 2), round(util_def, 4)

def size_member(
    M_u_kNm: float,
    P_u_kN: float,
    L_m: float = 6.0,
    delta_max_mm: float = None,
    yield_strength_MPa: float = 275.0,
    member_type: str = "beam",
    **kwargs
):
    """
    Finds the lightest compliant Eurocode 3 section enforcing EN 1993-1-1 Section 6.3.3:
    - Cross-section moment capacity utilization M_u / M_Rd <= 1.0
    - Axial compression capacity utilization P_u / N_Rd <= 1.0
    - Flexural Column Buckling utilization P_u / N_b_Rd <= 1.0
    - Lateral-Torsional Buckling utilization M_u / M_b_Rd <= 1.0
    - Combined EN 1993-1-1 §6.3.3 interaction: UR_combined = (P_u / N_b_Rd) + (M_u / M_b_Rd) <= 1.0
    - SLS Deflection utilization delta_max / (L / 360) <= 1.0

    Handles any additional keyword arguments (e.g. section_type, w_kNm) via **kwargs.
    """
    M_u = abs(M_u_kNm)
    P_u = abs(P_u_kN)
    f_y = float(yield_strength_MPa)

    # Determine section_type ('UB' or 'UC')
    sec_type_filter = kwargs.get("section_type", None)
    if not sec_type_filter and member_type:
        m_str = str(member_type).lower()
        if "beam" in m_str or m_str == "ub":
            sec_type_filter = "UB"
        elif "col" in m_str or m_str == "uc":
            sec_type_filter = "UC"

    candidates = STEEL_SECTIONS
    if sec_type_filter:
        candidates = [s for s in candidates if s["type"].upper() == sec_type_filter.upper()]

    candidates = sorted(candidates, key=lambda x: x["mass"])

    delta_limit = (L_m * 1000.0) / 360.0

    for sec in candidates:
        M_Rd = sec["W_pl_y"] * f_y * 1.0e-3
        N_Rd = sec["A"] * f_y * 1.0e-1

        N_b_Rd, chi_buck = calc_ec3_buckling_resistance(sec["A"], sec["I_y"], sec["I_z"], L_m, f_y)
        M_b_Rd, chi_ltb = calc_ec3_ltb_resistance(sec["W_pl_y"], sec["I_y"], sec["I_z"], L_m, f_y)

        if delta_max_mm is not None:
            d_max = float(delta_max_mm)
            d_lim = delta_limit
            util_def = d_max / d_lim if d_lim > 0 else 0.0
        elif "w_kNm" in kwargs and kwargs["w_kNm"] > 0:
            d_max, d_lim, util_def = calc_sls_deflection(kwargs["w_kNm"], L_m, sec["I_y"])
        else:
            d_max = 0.0
            d_lim = delta_limit
            util_def = 0.0

        util_M = M_u / M_Rd if M_Rd > 0 else 999.0
        util_N = P_u / N_Rd if N_Rd > 0 else 999.0
        util_buck = P_u / N_b_Rd if N_b_Rd > 0 else 999.0
        util_ltb = M_u / M_b_Rd if M_b_Rd > 0 else 999.0

        # Eurocode 3 EN 1993-1-1 Section 6.3.3 combined buckling interaction check:
        # UR_combined = (P_u / N_b_Rd) + (M_u / M_b_Rd)
        util_comb_uls = util_buck + util_ltb
        util_comb = max(util_comb_uls, util_def)

        if util_M <= 1.0 and util_N <= 1.0 and util_buck <= 1.0 and util_ltb <= 1.0 and util_def <= 1.0 and util_comb <= 1.0:
            return {
                "name": sec["name"],
                "type": sec["type"],
                "mass": sec["mass"],
                "A_cm2": sec["A"],
                "W_pl_y_cm3": sec["W_pl_y"],
                "M_Rd_kNm": round(M_Rd, 2),
                "N_Rd_kN": round(N_Rd, 2),
                "N_b_Rd_kN": round(N_b_Rd, 2),
                "M_b_Rd_kNm": round(M_b_Rd, 2),
                "chi_buckling": round(chi_buck, 4),
                "chi_ltb": round(chi_ltb, 4),
                "delta_max_mm": round(d_max, 2),
                "delta_lim_mm": round(d_lim, 2),
                "util_M": round(util_M, 4),
                "util_N": round(util_N, 4),
                "util_buck": round(util_buck, 4),
                "util_ltb": round(util_ltb, 4),
                "util_def": round(util_def, 4),
                "util_comb": round(util_comb, 4),
                "util_pct": round(util_comb * 100, 2)
            }

    # Fallback to largest available section
    largest = candidates[-1]
    M_Rd = largest["W_pl_y"] * f_y * 1.0e-3
    N_Rd = largest["A"] * f_y * 1.0e-1
    N_b_Rd, chi_buck = calc_ec3_buckling_resistance(largest["A"], largest["I_y"], largest["I_z"], L_m, f_y)
    M_b_Rd, chi_ltb = calc_ec3_ltb_resistance(largest["W_pl_y"], largest["I_y"], largest["I_z"], L_m, f_y)

    if delta_max_mm is not None:
        d_max = float(delta_max_mm)
        d_lim = delta_limit
        util_def = d_max / d_lim if d_lim > 0 else 0.0
    elif "w_kNm" in kwargs and kwargs["w_kNm"] > 0:
        d_max, d_lim, util_def = calc_sls_deflection(kwargs["w_kNm"], L_m, largest["I_y"])
    else:
        d_max = 0.0
        d_lim = delta_limit
        util_def = 0.0

    util_buck = P_u / N_b_Rd if N_b_Rd > 0 else 999.0
    util_ltb = M_u / M_b_Rd if M_b_Rd > 0 else 999.0
    util_comb = max((P_u / N_b_Rd) + (M_u / M_b_Rd), util_def)

    return {
        "name": largest["name"],
        "type": largest["type"],
        "mass": largest["mass"],
        "A_cm2": largest["A"],
        "W_pl_y_cm3": largest["W_pl_y"],
        "M_Rd_kNm": round(M_Rd, 2),
        "N_Rd_kN": round(N_Rd, 2),
        "N_b_Rd_kN": round(N_b_Rd, 2),
        "M_b_Rd_kNm": round(M_b_Rd, 2),
        "chi_buckling": round(chi_buck, 4),
        "chi_ltb": round(chi_ltb, 4),
        "delta_max_mm": round(d_max, 2),
        "delta_lim_mm": round(d_lim, 2),
        "util_M": round(M_u / M_Rd, 4),
        "util_N": round(P_u / N_Rd, 4),
        "util_buck": round(util_buck, 4),
        "util_ltb": round(util_ltb, 4),
        "util_def": round(util_def, 4),
        "util_comb": round(util_comb, 4),
        "util_pct": round(util_comb * 100, 2)
    }

