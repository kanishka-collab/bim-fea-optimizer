"""
Database of Standard UK/European Structural Steel Sections (UB and UC)
Properties include:
- mass: Mass per meter (kg/m)
- A: Cross-sectional Area (cm^2)
- W_pl_y: Plastic Section Modulus strong axis (cm^3)
- I_y: Second moment of area strong axis (cm^4)
- I_z: Second moment of area weak axis (cm^4)
"""

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

def size_member(M_u_kNm: float, P_u_kN: float, yield_strength_MPa: float = 275.0, section_type: str = None):
    """
    Finds the lightest compliant section where:
    - Moment capacity utilization M_u / M_Rd <= 1.0
    - Axial capacity utilization P_u / N_Rd <= 1.0
    - Combined interaction P_u / N_Rd + M_u / M_Rd <= 1.0

    Parameters:
    - M_u_kNm: Ultimate Bending Moment (kN*m)
    - P_u_kN: Ultimate Axial Force (kN)
    - yield_strength_MPa: Yield strength f_y (MPa), default 275 (S275 steel)
    - section_type: Optional filter 'UB' or 'UC'

    Returns:
    - Dict with optimal section details, capacities M_Rd, N_Rd, and utilization ratio
    """
    M_u = abs(M_u_kNm)
    P_u = abs(P_u_kN)
    f_y = float(yield_strength_MPa)

    # Filter candidate list
    candidates = STEEL_SECTIONS
    if section_type:
        candidates = [s for s in candidates if s["type"].upper() == section_type.upper()]

    # Sort by mass ascending (lightest first)
    candidates = sorted(candidates, key=lambda x: x["mass"])

    for sec in candidates:
        # Capacities:
        # M_Rd (kN*m) = W_pl_y (cm^3) * 10^-6 m^3 * (f_y * 10^3 kN/m^2) = W_pl_y * f_y * 10^-3
        M_Rd = sec["W_pl_y"] * f_y * 1.0e-3

        # N_Rd (kN) = A (cm^2) * 10^-4 m^2 * (f_y * 10^3 kN/m^2) = A * f_y * 10^-1
        N_Rd = sec["A"] * f_y * 1.0e-1

        util_M = M_u / M_Rd if M_Rd > 0 else 999.0
        util_N = P_u / N_Rd if N_Rd > 0 else 999.0
        util_comb = util_M + util_N

        if util_M <= 1.0 and util_N <= 1.0 and util_comb <= 1.0:
            return {
                "name": sec["name"],
                "type": sec["type"],
                "mass": sec["mass"],
                "A_cm2": sec["A"],
                "W_pl_y_cm3": sec["W_pl_y"],
                "M_Rd_kNm": round(M_Rd, 2),
                "N_Rd_kN": round(N_Rd, 2),
                "util_M": round(util_M, 4),
                "util_N": round(util_N, 4),
                "util_comb": round(util_comb, 4),
                "util_pct": round(util_comb * 100, 2)
            }

    # If no section is large enough in the subset, return largest available
    largest = candidates[-1]
    M_Rd = largest["W_pl_y"] * f_y * 1.0e-3
    N_Rd = largest["A"] * f_y * 1.0e-1
    util_comb = (M_u / M_Rd) + (P_u / N_Rd)
    return {
        "name": largest["name"],
        "type": largest["type"],
        "mass": largest["mass"],
        "A_cm2": largest["A"],
        "W_pl_y_cm3": largest["W_pl_y"],
        "M_Rd_kNm": round(M_Rd, 2),
        "N_Rd_kN": round(N_Rd, 2),
        "util_M": round(M_u / M_Rd, 4),
        "util_N": round(P_u / N_Rd, 4),
        "util_comb": round(util_comb, 4),
        "util_pct": round(util_comb * 100, 2)
    }
