"""
Eurocode Load Calculation Engine (EN 1990 / EN 1991-1-1 / EN 1991-1-4)
Includes Sri Lanka National Annex Wind Loading & Occupancy Imposed Load Mapping.
"""

import math
from typing import Dict, Any, Union, Optional


# ==============================================================================
# 1. EN 1991-1-1 OCCUPANCY CATEGORIES & IMPOSED LOADS (q_k, psi_0, psi_1, psi_2)
# ==============================================================================

OCCUPANCY_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "A": {
        "description": "Domestic and residential areas",
        "q_k": 2.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.5,
        "psi_2": 0.3,
    },
    "B": {
        "description": "Office areas",
        "q_k": 3.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.5,
        "psi_2": 0.3,
    },
    "C1": {
        "description": "Congregation areas - Tables (e.g. schools, cafes, restaurants)",
        "q_k": 3.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
    "C2": {
        "description": "Congregation areas - Fixed seats (e.g. theaters, cinemas, churches)",
        "q_k": 4.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
    "C3": {
        "description": "Congregation areas - Without obstacles (e.g. museums, exhibition halls, corridors)",
        "q_k": 5.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
    "C4": {
        "description": "Congregation areas - Physical activities (e.g. dance halls, gyms, stages)",
        "q_k": 5.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
    "C5": {
        "description": "Congregation areas - Large crowds (e.g. concert halls, sports grandstands)",
        "q_k": 5.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
    "D1": {
        "description": "Shopping areas - General retail / department stores",
        "q_k": 4.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
    "D2": {
        "description": "Shopping areas - Supermarkets / Wholesale department stores",
        "q_k": 5.0,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
    "E1": {
        "description": "Storage & Industrial areas - General storage, warehousing, industrial space",
        "q_k": 7.5,  # kN/m2
        "psi_0": 1.0,
        "psi_1": 0.9,
        "psi_2": 0.8,
    },
    "F": {
        "description": "Traffic & Parking areas - Light vehicles (gross weight <= 30 kN)",
        "q_k": 2.5,  # kN/m2
        "psi_0": 0.7,
        "psi_1": 0.7,
        "psi_2": 0.6,
    },
}


def get_occupancy_category_defaults(category_code: str = "B") -> Dict[str, Any]:
    """
    Returns default characteristic imposed load (q_k) and combination factors (psi_0, psi_1, psi_2)
    for EN 1991-1-1 occupancy categories (Category A through F).
    """
    code_upper = category_code.upper().strip()
    if code_upper not in OCCUPANCY_CATEGORIES:
        # Fallback to Category B (Office) if unknown
        code_upper = "B"
    return {
        "category": code_upper,
        **OCCUPANCY_CATEGORIES[code_upper]
    }


# ==============================================================================
# 2. EN 1991-1-4 WIND LOADING & SRI LANKA NATIONAL ANNEX PARAMETERS
# ==============================================================================

# Sri Lanka National Annex Wind Speed Zones (v_b,0 in m/s)
SRI_LANKA_WIND_ZONES: Dict[str, Dict[str, Any]] = {
    "Zone 1": {
        "description": "Inland / Low Wind Zone (Sri Lanka NA)",
        "v_b0": 23.0,  # m/s
    },
    "Zone 2": {
        "description": "Intermediate / Coastal Zone (Sri Lanka NA)",
        "v_b0": 29.0,  # m/s
    },
    "Zone 3": {
        "description": "High Wind / Northern & Eastern Coastal Zone (Sri Lanka NA)",
        "v_b0": 35.0,  # m/s
    },
}

# Terrain Categories according to EN 1991-1-4 Table 4.1
TERRAIN_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "0": {
        "description": "Sea or coastal area exposed to open sea",
        "z_0": 0.003,   # m
        "z_min": 1.0,   # m
    },
    "I": {
        "description": "Lakes or flat area with negligible vegetation & without obstacles",
        "z_0": 0.01,    # m
        "z_min": 1.0,   # m
    },
    "II": {
        "description": "Area with low vegetation such as grass and isolated obstacles (trees, buildings)",
        "z_0": 0.05,    # m
        "z_min": 2.0,   # m
    },
    "III": {
        "description": "Area with regular cover of vegetation or buildings or isolated obstacles",
        "z_0": 0.30,    # m
        "z_min": 5.0,   # m
    },
    "IV": {
        "description": "Area in which at least 15% of the surface is covered with buildings h > 15 m",
        "z_0": 1.00,    # m
        "z_min": 10.0,  # m
    },
}


def calculate_peak_velocity_pressure(
    z: float,
    v_b0: float = 23.0,
    terrain_category: str = "II",
    c_o: float = 1.0,
    rho: float = 1.25,
    c_dir: float = 1.0,
    c_season: float = 1.0,
    k_I: float = 1.0
) -> Dict[str, float]:
    """
    Computes EN 1991-1-4 peak velocity pressure q_p(z) at height z.
    
    :param z: Elevation height above ground level (m).
    :param v_b0: Fundamental basic wind velocity (m/s) [e.g. 23, 29, 35 m/s for Sri Lanka NA].
    :param terrain_category: EN 1991-1-4 Terrain category ('0', 'I', 'II', 'III', 'IV').
    :param c_o: Orography factor (c_o >= 1.0 for hills/cliffs/ridges).
    :param rho: Air density (default 1.25 kg/m3).
    :param c_dir: Directional factor (default 1.0).
    :param c_season: Season factor (default 1.0).
    :param k_I: Turbulence factor (default 1.0).
    :return: Dictionary containing q_p(z) in N/m2 and kN/m2 (kPa), v_m(z), I_v(z), c_r(z).
    """
    z_eval = max(0.1, z)
    c_o_eval = max(1.0, c_o)

    # 1. Basic wind velocity
    v_b = c_dir * c_season * v_b0

    # 2. Terrain roughness parameters
    tc_key = str(terrain_category).upper().strip()
    tc_data = TERRAIN_CATEGORIES.get(tc_key, TERRAIN_CATEGORIES["II"])
    z_0 = tc_data["z_0"]
    z_min = tc_data["z_min"]

    # Terrain factor k_r = 0.19 * (z_0 / 0.05)**0.07
    k_r = 0.19 * math.pow(z_0 / 0.05, 0.07)

    # Roughness factor c_r(z)
    z_for_cr = max(z_eval, z_min)
    c_r_z = k_r * math.log(z_for_cr / z_0)

    # Mean wind velocity v_m(z)
    v_m_z = c_r_z * c_o_eval * v_b

    # Turbulence intensity I_v(z)
    I_v_z = k_I / (c_o_eval * math.log(z_for_cr / z_0))

    # Peak velocity pressure q_p(z) = [1 + 7 * I_v(z)] * 0.5 * rho * v_m(z)^2
    q_p_N_m2 = (1.0 + 7.0 * I_v_z) * 0.5 * rho * (v_m_z ** 2)
    q_p_kPa = q_p_N_m2 / 1000.0

    return {
        "height_z_m": z_eval,
        "v_b0_m_s": v_b0,
        "v_b_m_s": v_b,
        "terrain_category": tc_key,
        "k_r": k_r,
        "c_r_z": c_r_z,
        "c_o": c_o_eval,
        "v_m_z_m_s": v_m_z,
        "I_v_z": I_v_z,
        "q_p_N_m2": q_p_N_m2,
        "q_p_kPa": q_p_kPa,
    }


# ==============================================================================
# 3. EN 1990 Eq. 6.10 ULS LOAD COMBINATIONS GENERATOR
# ==============================================================================

def generate_en1990_eq610_combinations(
    G_k: float,
    Q_k: float,
    W_k: float,
    psi_0_Q: float = 0.7,
    psi_0_W: float = 0.6,
    gamma_G: float = 1.35,
    gamma_Q: float = 1.50,
    gamma_W: float = 1.50
) -> Dict[str, Dict[str, Any]]:
    """
    Computes EN 1990 Eq. 6.10 ULS load combinations:
    - Comb 1 (Dominant Imposed Load Q_k): 1.35 G_k + 1.50 Q_k + 1.50 * psi_0_W * W_k  (1.35 G_k + 1.5 Q_k + 0.9 W_k when psi_0_W=0.6)
    - Comb 2 (Dominant Wind Load W_k):    1.35 G_k + 1.50 W_k + 1.50 * psi_0_Q * Q_k
    - SLS Characteristic:                 1.00 G_k + 1.00 Q_k + 1.00 W_k
    """
    # ULS Combination 1: Dominant Variable Imposed Load Q_k
    uls_comb1_val = gamma_G * G_k + gamma_Q * Q_k + gamma_W * psi_0_W * W_k

    # ULS Combination 2: Dominant Variable Wind Load W_k
    uls_comb2_val = gamma_G * G_k + gamma_W * W_k + gamma_Q * psi_0_Q * Q_k

    # SLS Characteristic Combination
    sls_char_val = 1.00 * G_k + 1.00 * Q_k + 1.00 * W_k

    # Governing ULS Total Load Action
    governing_uls_val = max(uls_comb1_val, uls_comb2_val)
    governing_comb_name = "Comb 1 (Dominant Q_k)" if uls_comb1_val >= uls_comb2_val else "Comb 2 (Dominant W_k)"

    return {
        "Comb1_Dominant_Q": {
            "name": "EN 1990 Eq. 6.10 (Comb 1: Dominant Imposed Q_k)",
            "formula": f"{gamma_G} G_k + {gamma_Q} Q_k + {gamma_W * psi_0_W:.2f} W_k",
            "dead_component": gamma_G * G_k,
            "live_component": gamma_Q * Q_k,
            "wind_component": gamma_W * psi_0_W * W_k,
            "total_design_load": uls_comb1_val,
        },
        "Comb2_Dominant_W": {
            "name": "EN 1990 Eq. 6.10 (Comb 2: Dominant Wind W_k)",
            "formula": f"{gamma_G} G_k + {gamma_W} W_k + {gamma_Q * psi_0_Q:.2f} Q_k",
            "dead_component": gamma_G * G_k,
            "wind_component": gamma_W * W_k,
            "live_component": gamma_Q * psi_0_Q * Q_k,
            "total_design_load": uls_comb2_val,
        },
        "SLS_Characteristic": {
            "name": "EN 1990 SLS Characteristic Combination",
            "formula": "1.0 G_k + 1.0 Q_k + 1.0 W_k",
            "total_design_load": sls_char_val,
        },
        "Governing_ULS": {
            "governing_comb": governing_comb_name,
            "total_design_load": governing_uls_val,
        }
    }


def get_load_combinations(G_k: float, Q_k: float, W_k: float) -> Dict[str, float]:
    """
    Backwards-compatible API wrapper returning key load values.
    """
    combos = generate_en1990_eq610_combinations(G_k, Q_k, W_k)
    comb1 = combos["Comb1_Dominant_Q"]
    comb2 = combos["Comb2_Dominant_W"]
    gov = combos["Governing_ULS"]

    return {
        "gamma_G": 1.35,
        "gamma_Q": 1.50,
        "gamma_W": 1.50,
        "uls_comb1": comb1["total_design_load"],
        "uls_comb2": comb2["total_design_load"],
        "uls_dead": comb1["dead_component"],
        "uls_live": comb1["live_component"],
        "uls_wind": comb1["wind_component"],
        "uls_governing_load": gov["total_design_load"],
    }


def calculate_eurocode_loads(
    G_k: float,
    Q_k: float,
    W_k: float,
    num_stories: int = 1,
    story_height: float = 3.5,
    occupancy_category: str = "B",
    sri_lanka_zone: str = "Zone 1",
    terrain_category: str = "II",
    c_o: float = 1.0
) -> Dict[str, Any]:
    """
    Comprehensive Eurocode load analysis function computing occupancy defaults,
    height-dependent peak velocity pressure q_p(z), and EN 1990 Eq. 6.10 ULS/SLS combinations.
    """
    occ_data = get_occupancy_category_defaults(occupancy_category)
    psi_0_Q = occ_data["psi_0"]

    zone_info = SRI_LANKA_WIND_ZONES.get(sri_lanka_zone, SRI_LANKA_WIND_ZONES["Zone 1"])
    v_b0 = zone_info["v_b0"]

    total_height = num_stories * story_height
    q_p_roof = calculate_peak_velocity_pressure(
        z=total_height,
        v_b0=v_b0,
        terrain_category=terrain_category,
        c_o=c_o
    )

    combinations = generate_en1990_eq610_combinations(
        G_k=G_k,
        Q_k=Q_k,
        W_k=W_k,
        psi_0_Q=psi_0_Q,
        psi_0_W=0.6
    )

    return {
        "G_k": G_k,
        "Q_k": Q_k,
        "W_k": W_k,
        "num_stories": num_stories,
        "story_height": story_height,
        "total_height": total_height,
        "occupancy": occ_data,
        "sri_lanka_wind_zone": {
            "zone": sri_lanka_zone,
            **zone_info
        },
        "peak_velocity_pressure_roof": q_p_roof,
        "combinations": combinations,
        "total_uls_line_load": combinations["Governing_ULS"]["total_design_load"]
    }
