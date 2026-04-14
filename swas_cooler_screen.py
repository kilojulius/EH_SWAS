#!/usr/bin/env python3
"""
SWAS Sample Cooler Screening Tool
===================================
Consumes heat-duty results from swas_heat_duty.calc_point() and screens a
library of candidate coolers (cooler_candidates.json) for each sample point.

Two-stage screening per sample point
-------------------------------------
Stage 1 -- Primary CW Cooler
  Steam points  : three thermal zones (desuperheat / condensation / subcooling)
                  Area required = sum of zone areas (conservative)
  Liquid points : single-phase counterflow LMTD
  UA = Q / (F * LMTD),  A_required = UA / U_assumed
  U values are conservative named constants defined below.

Stage 2 -- Secondary Chiller
  Always single-phase (sample arrives subcooled from Stage 1).
  Same LMTD / area approach.

For candidates that pass the area screen (Pass 1), an epsilon-NTU check
(Pass 2) predicts the actual sample outlet temperature achievable with the
candidate's published area, confirming thermal margin.

Usage
-----
    python swas_cooler_screen.py
    python swas_cooler_screen.py --config swas_config.yaml --screen-output screen_results.csv
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Import unit helpers and duty engine from the companion module.
# No steam properties are re-derived here; all Q values come from calc_point().
from swas_heat_duty import (
    calc_point,
    F_to_K, K_to_F,
    psig_to_Pa, W_to_BTU_hr,
    lbm_min_to_kg_s, kg_s_to_lbm_min,
    STEAM_PHASES, SUPERHEATED, SATURATED, SUBCOOLED, SUPERCRITICAL,
)

# ---------------------------------------------------------------------------
# Physical / conversion constants
# ---------------------------------------------------------------------------

CP_WATER_BTU_LBF   = 1.0        # Btu / (lbm · °F)  — liquid water specific heat
RHO_WATER_LB_GAL   = 8.34       # lbm / US gallon
GAL_PER_MIN_TO_LB_HR = RHO_WATER_LB_GAL * 60.0   # lbm/hr per GPM
LBM_PER_MIN_TO_LBM_HR = 60.0   # lbm/min → lbm/hr
MIN_LMTD_F         = 0.5        # LMTD below this (°F) is thermally indeterminate
MIN_Q_BTU_HR       = 1.0        # Skip a zone if its duty is below this threshold

# ---------------------------------------------------------------------------
# Conservative overall heat-transfer coefficient (U) assumptions
# ---------------------------------------------------------------------------
# All values are deliberately low to ensure the area screen is conservative.
# Basis: Sentry helical coil tube-in-shell geometry; small tube diameters
# reduce tube-side h compared to plain tube bundles.  Values are roughly
# 1/3–1/2 of Perry's typical ranges for clean water service.
#
# Zone definitions
#   DESUPERHEAT  : superheated steam (gas phase) on tube side → CW shell side.
#                  Gas-side film coefficient dominates; h is poor.
#   CONDENSATION : condensing steam on tube side → CW shell side.
#                  Condensation h is high; shell-side CW is controlling.
#   SUBCOOLING   : hot condensate (pressurized liquid) → CW.
#                  Both sides liquid; moderate turbulence in helical coil.
#   LIQUID_HOT   : single-phase liquid sample > 300 °F → CW.
#   LIQUID_WARM  : single-phase liquid sample 150–300 °F → CW.
#   LIQUID_COOL  : single-phase liquid sample < 150 °F, or any chiller stage.
#
# Reference for adjustment: TEMA standards, Perry's Chemical Engineers' Handbook,
# and Sentry equipment IOM for typical operating conditions.

U_DESUPERHEAT_BTU_HR_FT2_F   =  60.0   # steam gas phase → CW
U_CONDENSATION_BTU_HR_FT2_F  = 500.0   # condensing steam → CW
U_SUBCOOLING_BTU_HR_FT2_F    = 150.0   # hot condensate → CW (within steam primary)
U_LIQUID_HOT_BTU_HR_FT2_F    = 130.0   # liquid sample > 300 F → CW
U_LIQUID_WARM_BTU_HR_FT2_F   = 100.0   # liquid sample 150–300 F → CW
U_LIQUID_COOL_BTU_HR_FT2_F   =  70.0   # liquid sample < 150 F, or chiller stage

# ---------------------------------------------------------------------------
# U selection logic
# ---------------------------------------------------------------------------

def select_U_assumed(is_steam_zone: bool,
                     T_sample_in_F: float,
                     zone: Optional[str] = None) -> float:
    """
    Return a conservative overall heat-transfer coefficient (Btu/hr·ft²·°F).

    Parameters
    ----------
    is_steam_zone : bool
        True when the zone involves steam (gas or condensing) on the sample side.
    T_sample_in_F : float
        Sample inlet temperature for the zone (°F).  Used for liquid service
        to pick the appropriate liquid U tier.
    zone : str or None
        For steam zones pass one of: 'desuperheat', 'condensation', 'subcooling'.
        For liquid or chiller zones leave as None (T_sample_in_F drives selection).

    Returns
    -------
    float
        U in Btu / (hr · ft² · °F).
    """
    if is_steam_zone:
        if zone == "desuperheat":
            return U_DESUPERHEAT_BTU_HR_FT2_F
        if zone == "condensation":
            return U_CONDENSATION_BTU_HR_FT2_F
        # Default for steam zones not otherwise specified (e.g. subcooling within primary)
        return U_SUBCOOLING_BTU_HR_FT2_F
    # Liquid (single-phase) service — tier by sample temperature
    if T_sample_in_F > 300.0:
        return U_LIQUID_HOT_BTU_HR_FT2_F
    if T_sample_in_F > 150.0:
        return U_LIQUID_WARM_BTU_HR_FT2_F
    return U_LIQUID_COOL_BTU_HR_FT2_F

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class CandidateCooler:
    """
    One candidate cooler entry from the JSON database.
    All fields optional except model_name so missing keys don't crash the load.
    """
    model_name: str = ""
    service_type: str = "both"          # "steam", "liquid", or "both"
    max_pressure_psig: Optional[float] = None
    max_temp_F: Optional[float] = None
    heat_transfer_area_ft2: Optional[float] = None
    min_sample_flow_lbh: Optional[float] = None
    max_sample_flow_lbh: Optional[float] = None
    notes: str = ""


@dataclass
class ZoneUA:
    """Thermal calculation results for one heat-exchanger zone."""
    zone_name: str          # e.g. "Desuperheat", "Condensation", "Subcooling"
    Q_BTU_hr: float         # Heat duty for this zone (Btu/hr)
    lmtd_F: Optional[float]                  # Log-mean temperature difference (°F)
    U_assumed: float                          # U value used (Btu/hr·ft²·°F)
    UA_required: Optional[float]             # Q / (F·LMTD)  (Btu/hr·°F)
    A_required_ft2: Optional[float]          # UA / U  (ft²)
    T_predicted_out_F: Optional[float] = None  # eNTU predicted sample outlet (°F)
    note: str = ""


@dataclass
class CoolerMatch:
    """A candidate cooler that passed all screening checks for one stage."""
    candidate: CandidateCooler
    A_required_ft2: float
    oversize_factor: Optional[float]    # candidate area / required area
    T_predicted_out_F: Optional[float]  # eNTU predicted sample outlet (°F)
    notes: str = ""


@dataclass
class CoolerReject:
    """A candidate cooler that failed one or more checks."""
    candidate: CandidateCooler
    fail_reason: str


@dataclass
class PointScreenResult:
    """All screening outputs for one sample point."""
    # --- Identity and phase ---
    name: str
    phase: str
    is_steam: bool
    T_in_F: float
    P_psig: float
    T_prim_F: float
    T_out_F: float

    # --- Duties (Btu/hr) ---
    Q_primary_required: float
    Q_primary_max_utility: float
    primary_utility_limited: bool
    Q_secondary_required: float

    # --- Primary stage thermal analysis ---
    ua_primary_required: Optional[float]      # total required UA (Btu/hr·°F)
    A_primary_required_ft2: Optional[float]   # total required area (ft²)
    steam_zones: Optional[list]               # list[ZoneUA]; None for liquid points

    # --- Secondary stage thermal analysis ---
    ua_secondary_required: Optional[float]
    A_secondary_required_ft2: Optional[float]

    # --- Service type note ---
    service_type_note: str

    # --- Candidate screening results ---
    passing_primary: list = field(default_factory=list)    # list[CoolerMatch]
    passing_secondary: list = field(default_factory=list)
    rejected_primary: list = field(default_factory=list)   # list[CoolerReject]
    rejected_secondary: list = field(default_factory=list)

# ---------------------------------------------------------------------------
# Utility limit functions
# ---------------------------------------------------------------------------

def primary_max_duty_BTU_hr(gpm: float,
                             T_cw_in_F: float,
                             T_cw_out_max_F: float,
                             Cp: float = CP_WATER_BTU_LBF,
                             rho_lb_gal: float = RHO_WATER_LB_GAL) -> float:
    """
    Maximum heat the primary CW utility can remove given its flow and
    allowable temperature rise.

        Q_max = m_dot_cw * Cp * (T_cw_out_max - T_cw_in)
        m_dot_cw [lbm/hr] = gpm * rho [lbm/gal] * 60 [min/hr]
    """
    m_dot_cw_lbh = gpm * rho_lb_gal * 60.0   # lbm/hr
    return m_dot_cw_lbh * Cp * (T_cw_out_max_F - T_cw_in_F)


def secondary_required_BTU_hr(Q_total: float, Q_primary_actual: float) -> float:
    """Remaining duty that must be handled by the chiller after the primray cooler."""
    return max(0.0, Q_total - Q_primary_actual)

# ---------------------------------------------------------------------------
# LMTD helpers
# ---------------------------------------------------------------------------

def lmtd_counterflow(T_h_in: float, T_h_out: float,
                     T_c_in: float,  T_c_out: float) -> Optional[float]:
    """
    Log-Mean Temperature Difference for a counterflow heat exchanger.

    In counterflow the hot inlet faces the cold outlet (same end):
        dT1 = T_h_in  - T_c_out   (hot-inlet end)
        dT2 = T_h_out - T_c_in    (hot-outlet end)

    Returns None if either endpoint dT <= 0 (temperature cross or equal temps).
    Edge case dT1 == dT2: L'Hôpital limit gives LMTD = dT1.
    """
    dT1 = T_h_in  - T_c_out
    dT2 = T_h_out - T_c_in
    if dT1 <= 0.0 or dT2 <= 0.0:
        return None   # temperature cross — exchanger cannot achieve this duty
    if abs(dT1 - dT2) < 1e-6:
        return dT1    # L'Hôpital limit: LMTD → dT when both deltas are equal
    return (dT1 - dT2) / math.log(dT1 / dT2)


def lmtd_condensing(T_sat_F: float,
                    T_c_in_F: float,
                    T_c_out_F: float) -> Optional[float]:
    """
    LMTD for condensing (or evaporating) service where the hot side is
    isothermal at T_sat_F.

        dT1 = T_sat - T_c_out   (condensate outlet end — approaching cold outlet)
        dT2 = T_sat - T_c_in    (feed end — approaching cold inlet)

    Returns None if either delta <= 0 (utility temperature crosses saturation).
    """
    dT1 = T_sat_F - T_c_out_F
    dT2 = T_sat_F - T_c_in_F
    if dT1 <= 0.0 or dT2 <= 0.0:
        return None
    if abs(dT1 - dT2) < 1e-6:
        return dT1
    return (dT1 - dT2) / math.log(dT1 / dT2)


def ua_from_lmtd(Q_BTU_hr: float,
                 lmtd_F: Optional[float],
                 F_correction: float = 1.0) -> Optional[float]:
    """
    Required UA from the basic exchanger equation:  Q = U*A*F*LMTD.

        UA_required = Q / (F * LMTD)

    Returns None if LMTD is indeterminate or below MIN_LMTD_F.
    """
    if lmtd_F is None or lmtd_F < MIN_LMTD_F:
        return None
    return Q_BTU_hr / (F_correction * lmtd_F)


def area_required(UA: Optional[float], U: float) -> Optional[float]:
    """
    Required heat-transfer area:  A = UA / U.
    Returns None if UA is None (propagates indeterminate LMTD).
    """
    if UA is None:
        return None
    return UA / U

# ---------------------------------------------------------------------------
# epsilon-NTU helpers
# ---------------------------------------------------------------------------

def epsilon_counterflow(NTU: float, R: float) -> float:
    """
    Heat-exchanger effectiveness for a counterflow configuration.

    Parameters
    ----------
    NTU : float   Number of Transfer Units  =  UA / C_min
    R   : float   Heat capacity rate ratio  =  C_min / C_max  (0 <= R <= 1)

    Special cases
    -------------
    R = 0  (one side isothermal, e.g. condensing/evaporating):
        epsilon = 1 - exp(-NTU)

    R = 1  (balanced flow, C_min = C_max):
        epsilon = NTU / (1 + NTU)

    General (0 < R < 1):
        epsilon = (1 - exp(-NTU*(1-R))) / (1 - R*exp(-NTU*(1-R)))
    """
    if R < 1e-9:
        # Condensing / evaporating side — isothermal, infinite capacity rate
        return 1.0 - math.exp(-NTU)
    if abs(R - 1.0) < 1e-6:
        return NTU / (1.0 + NTU)
    factor = math.exp(-NTU * (1.0 - R))
    return (1.0 - factor) / (1.0 - R * factor)


def predict_outlet_eNTU(A_candidate_ft2: float,
                        U_assumed: float,
                        m_sample_lbhr: float,
                        T_sample_in_F: float,
                        T_utility_in_F: float,
                        is_condensing: bool = False,
                        Cp_sample: float = CP_WATER_BTU_LBF,
                        Cp_utility: float = CP_WATER_BTU_LBF,
                        m_utility_lbhr: Optional[float] = None) -> dict:
    """
    Predict the sample outlet temperature achievable with a candidate cooler
    of known area using the epsilon-NTU method.

    Used as a Pass-2 check for candidates that have already passed the LMTD
    area screen.  Tells the engineer not just pass/fail but HOW CLOSE to the
    target the candidate gets.

    Parameters
    ----------
    A_candidate_ft2   : published heat-transfer area of the candidate cooler
    U_assumed         : overall U coefficient (Btu/hr·ft²·°F)
    m_sample_lbhr     : sample mass flow rate (lbm/hr)
    T_sample_in_F     : sample inlet temperature for this stage (°F)
    T_utility_in_F    : utility inlet temperature (°F)
    is_condensing     : True for condensation zone (utility side isothermal → R=0)
    Cp_sample         : sample specific heat (default water, Btu/lbm·°F)
    Cp_utility        : utility specific heat (default water, Btu/lbm·°F)
    m_utility_lbhr    : utility mass flow (lbm/hr); None treats utility as infinite
                        (R=0, isothermal) which is conservative for condensing service.

    Returns
    -------
    dict with keys:
        T_sample_out_F    : predicted sample outlet temperature (°F)
        Q_predicted_BTU_hr: predicted heat duty (Btu/hr)
        epsilon           : heat-exchanger effectiveness
        NTU               : number of transfer units
        C_min             : minimum heat capacity rate (Btu/hr·°F)
        C_max             : maximum heat capacity rate (same units)
        R                 : C_min / C_max
    """
    UA_candidate = U_assumed * A_candidate_ft2
    C_sample     = m_sample_lbhr * Cp_sample          # Btu/hr/°F

    if is_condensing or m_utility_lbhr is None:
        # Condensing service: utility side is isothermal → R = 0
        R       = 0.0
        C_min   = C_sample
        C_max   = float("inf")
    else:
        C_utility = m_utility_lbhr * Cp_utility
        C_min  = min(C_sample, C_utility)
        C_max  = max(C_sample, C_utility)
        R      = C_min / C_max if C_max > 0 else 0.0

    NTU     = UA_candidate / C_min if C_min > 0 else 0.0
    epsilon = epsilon_counterflow(NTU, R)

    # Maximum possible heat transfer (if epsilon = 1)
    Q_max   = C_min * abs(T_sample_in_F - T_utility_in_F)
    Q_pred  = epsilon * Q_max

    # Predicted sample outlet: sample is losing heat so T decreases
    T_sample_out_F = T_sample_in_F - Q_pred / C_sample if C_sample > 0 else T_sample_in_F

    return {
        "T_sample_out_F":     round(T_sample_out_F, 1),
        "Q_predicted_BTU_hr": round(Q_pred),
        "epsilon":            round(epsilon, 4),
        "NTU":                round(NTU, 3),
        "C_min":              round(C_min, 1),
        "C_max":              round(C_max, 1) if C_max != float("inf") else None,
        "R":                  round(R, 4),
    }

# ---------------------------------------------------------------------------
# Steam three-zone primary cooler analysis
# ---------------------------------------------------------------------------

def decompose_steam_primary_zones(r: dict,
                                  cw_in_F: float,
                                  cw_flow_gpm: float,
                                  F_correction: float = 1.0) -> list:
    """
    Split the steam primary cooler duty into three sequential thermal zones
    and compute the required heat-transfer area for each.

    Zone order (counterflow: CW enters at the subcooling end):
        Zone 1 — Subcooling   : T_sat → T_prim_F  (condensate cools to primary outlet)
        Zone 2 — Condensation : T_sat → T_sat      (isothermal, latent heat)
        Zone 3 — Desuperheat  : T_in  → T_sat      (superheated steam cools to T_sat)

    CW temperature rise is allocated proportionally to zone duty fractions so
    that the zone-by-zone LMTD calculations are self-consistent.

    The total required area (sum of zones) is conservative because each zone
    uses its own conservative U.  This is appropriate for a screening tool.

    Parameters
    ----------
    r            : result dict from calc_point() for a steam point
    cw_in_F      : cooling water supply temperature (°F)
    cw_flow_gpm  : cooling water flow to this cooler (GPM)
    F_correction : LMTD correction factor (1.0 = pure counterflow)

    Returns
    -------
    list of ZoneUA objects (only zones with Q > MIN_Q_BTU_HR are returned).
    """
    Q_prim     = r["Q_prim"]        # total primary cooler duty (Btu/hr)
    Q_dsh      = r["Q_dsh"]         # desuperheat zone
    Q_cond     = r["Q_cond"]        # condensation zone
    # Subcooling within the primary cooler = total primary - desuperheat - condensation
    Q_subcool  = Q_prim - Q_dsh - Q_cond

    T_in_F     = r["T_in_F"]
    T_sat_F    = r["T_sat_F"]       # saturation temperature at sample pressure
    T_prim_F   = r["T_prim_F"]      # primary cooler sample outlet (subcooled target)
    cw_dT      = r["cw_dT"]         # actual CW temperature rise across entire primary

    # CW outlet temperature (overall)
    cw_out_F   = cw_in_F + cw_dT

    # Allocate CW temperature rise proportionally to zone duties.
    # CW flows counterflow: coldest CW at the subcooling end,
    # warmest CW exits at the desuperheat end.
    if Q_prim > MIN_Q_BTU_HR:
        cw_dT_sc   = cw_dT * (Q_subcool  / Q_prim) if Q_subcool  > MIN_Q_BTU_HR else 0.0
        cw_dT_cond = cw_dT * (Q_cond     / Q_prim) if Q_cond     > MIN_Q_BTU_HR else 0.0
        cw_dT_dsh  = cw_dT * (Q_dsh      / Q_prim) if Q_dsh      > MIN_Q_BTU_HR else 0.0
    else:
        return []

    # CW temperature profile (counterflow boundaries)
    # CW enters at cw_in_F on the subcooling side and leaves at cw_out_F on the DSH side
    T_cw_0     = cw_in_F                       # CW enters at subcooling zone cold end
    T_cw_1     = T_cw_0 + cw_dT_sc             # CW between subcooling and condensation zones
    T_cw_2     = T_cw_1 + cw_dT_cond           # CW between condensation and desuperheat zones
    T_cw_3     = cw_out_F                       # CW leaves after desuperheat zone (verify ≈ T_cw_2 + dT_dsh)

    zones = []

    # --- Zone 1: Subcooling (condensate T_sat → T_prim_F) ---
    if Q_subcool > MIN_Q_BTU_HR:
        lmtd_sc = lmtd_counterflow(T_sat_F, T_prim_F, T_cw_0, T_cw_1)
        U_sc     = select_U_assumed(is_steam_zone=True, T_sample_in_F=T_sat_F,
                                    zone="subcooling")
        UA_sc    = ua_from_lmtd(Q_subcool, lmtd_sc, F_correction)
        A_sc     = area_required(UA_sc, U_sc)
        zones.append(ZoneUA(
            zone_name     = "Subcooling",
            Q_BTU_hr      = Q_subcool,
            lmtd_F        = round(lmtd_sc, 2) if lmtd_sc else None,
            U_assumed     = U_sc,
            UA_required   = round(UA_sc, 1) if UA_sc else None,
            A_required_ft2= round(A_sc, 3) if A_sc else None,
            note          = "" if lmtd_sc else "LMTD indeterminate — check CW inlet vs T_prim",
        ))

    # --- Zone 2: Condensation (isothermal at T_sat) ---
    if Q_cond > MIN_Q_BTU_HR:
        lmtd_cd  = lmtd_condensing(T_sat_F, T_cw_1, T_cw_2)
        U_cd     = select_U_assumed(is_steam_zone=True, T_sample_in_F=T_sat_F,
                                    zone="condensation")
        UA_cd    = ua_from_lmtd(Q_cond, lmtd_cd, F_correction)
        A_cd     = area_required(UA_cd, U_cd)
        zones.append(ZoneUA(
            zone_name     = "Condensation",
            Q_BTU_hr      = Q_cond,
            lmtd_F        = round(lmtd_cd, 2) if lmtd_cd else None,
            U_assumed     = U_cd,
            UA_required   = round(UA_cd, 1) if UA_cd else None,
            A_required_ft2= round(A_cd, 3) if A_cd else None,
            note          = "" if lmtd_cd else "LMTD indeterminate — CW may exceed T_sat",
        ))

    # --- Zone 3: Desuperheat (T_in → T_sat) ---
    if Q_dsh > MIN_Q_BTU_HR:
        lmtd_dsh = lmtd_counterflow(T_in_F, T_sat_F, T_cw_2, T_cw_3)
        U_dsh    = select_U_assumed(is_steam_zone=True, T_sample_in_F=T_in_F,
                                    zone="desuperheat")
        UA_dsh   = ua_from_lmtd(Q_dsh, lmtd_dsh, F_correction)
        A_dsh    = area_required(UA_dsh, U_dsh)
        zones.append(ZoneUA(
            zone_name     = "Desuperheat",
            Q_BTU_hr      = Q_dsh,
            lmtd_F        = round(lmtd_dsh, 2) if lmtd_dsh else None,
            U_assumed     = U_dsh,
            UA_required   = round(UA_dsh, 1) if UA_dsh else None,
            A_required_ft2= round(A_dsh, 3) if A_dsh else None,
            note          = "" if lmtd_dsh else "LMTD indeterminate — check steam vs CW temps",
        ))

    return zones

# ---------------------------------------------------------------------------
# Candidate database loader
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    """Convert a JSON value to float, returning None for null/missing."""
    return float(val) if val is not None else None


def load_candidates(json_path: str) -> list:
    """
    Load the candidate cooler database from a JSON file.

    Unknown or missing JSON keys are silently defaulted so that partial
    entries do not crash the tool.  Entries whose 'model_name' starts with
    '_comment' are skipped (they are documentation-only nodes).

    Returns a list of CandidateCooler objects.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    candidates = []
    for entry in raw:
        if any(str(k).startswith("_comment") for k in entry):
            continue    # skip documentation nodes
        candidates.append(CandidateCooler(
            model_name            = str(entry.get("model_name", "Unknown")),
            service_type          = str(entry.get("service_type", "both")).lower(),
            max_pressure_psig     = _safe_float(entry.get("max_pressure_psig")),
            max_temp_F            = _safe_float(entry.get("max_temp_F")),
            heat_transfer_area_ft2= _safe_float(entry.get("heat_transfer_area_ft2")),
            min_sample_flow_lbh   = _safe_float(entry.get("min_sample_flow_lbh")),
            max_sample_flow_lbh   = _safe_float(entry.get("max_sample_flow_lbh")),
            notes                 = str(entry.get("notes", "")),
        ))
    return candidates

# ---------------------------------------------------------------------------
# Atomic candidate check functions
# ---------------------------------------------------------------------------

def check_pressure(c: CandidateCooler, P_sample_psig: float):
    """Fail if the candidate's max pressure rating is below the sample pressure."""
    if c.max_pressure_psig is None:
        return True, "Pressure rating not specified — assume OK (verify with vendor)"
    if P_sample_psig > c.max_pressure_psig:
        return False, (f"Pressure {P_sample_psig:.0f} psig exceeds rating "
                       f"{c.max_pressure_psig:.0f} psig")
    return True, ""


def check_temperature(c: CandidateCooler, T_sample_F: float):
    """Fail if the candidate's max temperature rating is below the sample temperature."""
    if c.max_temp_F is None:
        return True, "Temperature rating not specified — assume OK (verify with vendor)"
    if T_sample_F > c.max_temp_F:
        return False, (f"Temperature {T_sample_F:.0f} °F exceeds rating "
                       f"{c.max_temp_F:.0f} °F")
    return True, ""


def check_service_type(c: CandidateCooler, is_steam_service: bool):
    """
    Fail if the service type is incompatible.
    Steam service requires 'steam' or 'both'.
    Liquid service is acceptable for any type.
    """
    if is_steam_service and c.service_type == "liquid":
        return False, (f"Steam/condensing service required; "
                       f"model rated for liquid service only")
    return True, ""


def check_sample_flow(c: CandidateCooler, flow_lbh: float):
    """Fail if the sample flow is outside the candidate's stated operating range."""
    if c.min_sample_flow_lbh is not None and flow_lbh < c.min_sample_flow_lbh:
        return False, (f"Sample flow {flow_lbh:.1f} lbm/hr below "
                       f"minimum {c.min_sample_flow_lbh:.0f} lbm/hr")
    if c.max_sample_flow_lbh is not None and flow_lbh > c.max_sample_flow_lbh:
        return False, (f"Sample flow {flow_lbh:.1f} lbm/hr exceeds "
                       f"maximum {c.max_sample_flow_lbh:.0f} lbm/hr")
    return True, ""


def check_thermal_area(c: CandidateCooler,
                       A_required_ft2: Optional[float],
                       margin_factor: float = 1.0):
    """
    Fail if the candidate's heat-transfer area is less than A_required * margin_factor.

    Returns (pass, reason, oversize_factor).
    oversize_factor = candidate_area / A_required  (>1 means candidate is larger than needed).
    None if area data is missing for the candidate or A_required is indeterminate.
    """
    if A_required_ft2 is None:
        return True, "Required area indeterminate (check LMTD / temperatures)", None
    if A_required_ft2 <= 0.0:
        return True, "No thermal duty for this stage", None
    if c.heat_transfer_area_ft2 is None:
        return True, "Candidate area not specified — verify with vendor", None
    oversize = c.heat_transfer_area_ft2 / A_required_ft2
    if oversize < margin_factor:
        return False, (
            f"Area {c.heat_transfer_area_ft2:.3f} ft² < required "
            f"{A_required_ft2:.3f} ft² (oversize factor {oversize:.2f} < {margin_factor:.2f})"
        ), oversize
    return True, "", oversize

# ---------------------------------------------------------------------------
# Stage screener — runs all checks and ranks passing candidates
# ---------------------------------------------------------------------------

def screen_stage(candidates: list,
                 is_steam_service: bool,
                 P_psig: float,
                 T_F: float,
                 flow_lbh: float,
                 A_required_ft2: Optional[float],
                 margin_factor: float,
                 # eNTU Pass-2 parameters
                 U_for_eNTU: Optional[float] = None,
                 m_sample_lbhr: Optional[float] = None,
                 T_sample_in_F: Optional[float] = None,
                 T_utility_in_F: Optional[float] = None,
                 is_condensing_eNTU: bool = False):
    """
    Screen all candidates for one exchanger stage and return passing/rejected lists.

    Checks applied in order (fail on first failure):
        1. Service type compatibility
        2. Pressure rating
        3. Temperature rating
        4. Sample flow range
        5. Thermal area adequacy (LMTD-based)

    Passing candidates are sorted ascending by oversize_factor (closest fit first).
    Candidates with unknown area sort last (can't rank them thermally).

    For passing candidates with known area, an epsilon-NTU check predicts the
    achievable sample outlet temperature and adds it to the CoolerMatch record.

    Returns
    -------
    (passing : list[CoolerMatch],  rejected : list[CoolerReject])
    """
    passing  = []
    rejected = []

    for c in candidates:
        # --- Check 1: service type ---
        ok, reason = check_service_type(c, is_steam_service)
        if not ok:
            rejected.append(CoolerReject(c, reason))
            continue

        # --- Check 2: pressure ---
        ok, reason = check_pressure(c, P_psig)
        if not ok:
            rejected.append(CoolerReject(c, reason))
            continue

        # --- Check 3: temperature ---
        ok, reason = check_temperature(c, T_F)
        if not ok:
            rejected.append(CoolerReject(c, reason))
            continue

        # --- Check 4: sample flow ---
        ok, reason = check_sample_flow(c, flow_lbh)
        if not ok:
            rejected.append(CoolerReject(c, reason))
            continue

        # --- Check 5: thermal area ---
        ok, reason, oversize = check_thermal_area(c, A_required_ft2, margin_factor)
        if not ok:
            rejected.append(CoolerReject(c, f"Thermal area — {reason}"))
            continue

        # --- Pass-2: eNTU outlet temperature prediction ---
        T_out_pred = None
        if (c.heat_transfer_area_ft2 is not None
                and U_for_eNTU is not None
                and m_sample_lbhr is not None
                and T_sample_in_F is not None
                and T_utility_in_F is not None):
            entu = predict_outlet_eNTU(
                A_candidate_ft2 = c.heat_transfer_area_ft2,
                U_assumed       = U_for_eNTU,
                m_sample_lbhr   = m_sample_lbhr,
                T_sample_in_F   = T_sample_in_F,
                T_utility_in_F  = T_utility_in_F,
                is_condensing   = is_condensing_eNTU,
            )
            T_out_pred = entu["T_sample_out_F"]

        match_note = reason if reason else ""  # captures area-unknown note
        passing.append(CoolerMatch(
            candidate         = c,
            A_required_ft2    = A_required_ft2 if A_required_ft2 is not None else 0.0,
            oversize_factor   = oversize,
            T_predicted_out_F = T_out_pred,
            notes             = match_note,
        ))

    # Sort passing by oversize_factor ascending; unknowns (None) sort last
    passing.sort(key=lambda m: m.oversize_factor if m.oversize_factor is not None else 1e9)
    return passing, rejected

# ---------------------------------------------------------------------------
# Per-point orchestration
# ---------------------------------------------------------------------------

def screen_single_point(r: dict,
                        candidates: list,
                        scr_cfg: dict,
                        cw_supply_F: float,
                        cw_max_gpm: float,
                        cw_dT_design: float) -> PointScreenResult:
    """
    Screen one sample point (result dict from calc_point()) against all
    candidate coolers.

    Returns a PointScreenResult with full per-stage analysis and ranked
    passing/rejected candidate lists.
    """
    chw_in_F       = float(scr_cfg.get("chiller_supply_F", 55.0))
    chw_out_F      = float(scr_cfg.get("chiller_return_F", 65.0))
    margin         = float(scr_cfg.get("ua_margin_factor",  1.0))
    F_corr         = float(scr_cfg.get("lmtd_correction_F", 1.0))
    cw_out_max_F   = cw_supply_F + cw_dT_design

    is_steam       = r["is_steam"]
    T_in_F         = r["T_in_F"]
    T_prim_F       = r["T_prim_F"]
    T_out_F        = r["T_out_F"]       # final target (77 F)
    P_psig         = r["P_psig"]
    Q_prim         = float(r["Q_prim"])
    Q_chill        = float(r["Q_chill"])
    Q_total        = float(r["Q_total"])
    flow_lbmin     = float(r["flow_slip"])      # flow through the cooler (lb/min)
    flow_lbhr      = flow_lbmin * LBM_PER_MIN_TO_LBM_HR

    # --- Primary utility limit ---
    Q_prim_max     = primary_max_duty_BTU_hr(cw_max_gpm, cw_supply_F, cw_out_max_F)
    prim_limited   = Q_prim > Q_prim_max

    # --- Primary thermal analysis ---
    steam_zones        = None
    ua_primary         = None
    A_primary          = None
    U_prim_for_eNTU    = None

    if is_steam:
        service_note = "Steam/condensing-service cooler required for primary stage"
        steam_zones  = decompose_steam_primary_zones(r, cw_supply_F, cw_max_gpm, F_corr)
        # Sum zone areas for total primary required area
        valid_areas  = [z.A_required_ft2 for z in steam_zones if z.A_required_ft2 is not None]
        valid_UAs    = [z.UA_required    for z in steam_zones if z.UA_required    is not None]
        A_primary    = sum(valid_areas) if valid_areas else None
        ua_primary   = sum(valid_UAs)   if valid_UAs   else None
        # For eNTU Pass-2 on steam: use condensation U (dominant zone) as representative
        U_prim_for_eNTU = U_CONDENSATION_BTU_HR_FT2_F

    else:
        service_note = "Liquid-service cooler acceptable for primary stage"
        # Single-phase counterflow primary
        # CW outlet temperature for this point (back-calculated from actual cw_dT)
        cw_actual_out_F = cw_supply_F + r["cw_dT"]
        lmtd_prim    = lmtd_counterflow(T_in_F, T_prim_F,
                                         cw_supply_F, cw_actual_out_F)
        U_prim       = select_U_assumed(is_steam_zone=False, T_sample_in_F=T_in_F)
        U_prim_for_eNTU = U_prim
        ua_primary   = ua_from_lmtd(Q_prim, lmtd_prim, F_corr)
        A_primary    = area_required(ua_primary, U_prim)

    # --- Secondary (chiller) thermal analysis — always single-phase liquid ---
    # Sample enters secondary at T_prim_F; exits at T_out_F (77 F target)
    lmtd_sec  = lmtd_counterflow(T_prim_F, T_out_F, chw_in_F, chw_out_F)
    U_sec     = select_U_assumed(is_steam_zone=False, T_sample_in_F=T_prim_F)
    ua_sec    = ua_from_lmtd(Q_chill, lmtd_sec, F_corr) if Q_chill > MIN_Q_BTU_HR else None
    A_sec     = area_required(ua_sec, U_sec)

    # --- Screen primary stage candidates ---
    passing_prim, rejected_prim = screen_stage(
        candidates      = candidates,
        is_steam_service= is_steam,
        P_psig          = P_psig,
        T_F             = T_in_F,
        flow_lbh        = flow_lbhr,
        A_required_ft2  = A_primary,
        margin_factor   = margin,
        U_for_eNTU      = U_prim_for_eNTU,
        m_sample_lbhr   = flow_lbhr,
        T_sample_in_F   = T_in_F,
        T_utility_in_F  = cw_supply_F,
        is_condensing_eNTU = is_steam,   # treat primary as condensing for eNTU when steam
    )

    # --- Screen secondary (chiller) stage candidates ---
    # Secondary is always liquid service; sample enters at T_prim_F
    passing_sec, rejected_sec = screen_stage(
        candidates      = candidates,
        is_steam_service= False,
        P_psig          = P_psig,
        T_F             = T_prim_F,
        flow_lbh        = flow_lbhr,
        A_required_ft2  = A_sec,
        margin_factor   = margin,
        U_for_eNTU      = U_sec,
        m_sample_lbhr   = flow_lbhr,
        T_sample_in_F   = T_prim_F,
        T_utility_in_F  = chw_in_F,
        is_condensing_eNTU = False,
    )

    return PointScreenResult(
        name                   = r["name"],
        phase                  = r["phase"],
        is_steam               = is_steam,
        T_in_F                 = T_in_F,
        P_psig                 = P_psig,
        T_prim_F               = T_prim_F,
        T_out_F                = T_out_F,
        Q_primary_required     = Q_prim,
        Q_primary_max_utility  = Q_prim_max,
        primary_utility_limited= prim_limited,
        Q_secondary_required   = Q_chill,
        ua_primary_required    = round(ua_primary, 1) if ua_primary else None,
        A_primary_required_ft2 = round(A_primary, 3)  if A_primary  else None,
        steam_zones            = steam_zones,
        ua_secondary_required  = round(ua_sec, 1) if ua_sec else None,
        A_secondary_required_ft2 = round(A_sec, 3) if A_sec else None,
        service_type_note      = service_note,
        passing_primary        = passing_prim,
        passing_secondary      = passing_sec,
        rejected_primary       = rejected_prim,
        rejected_secondary     = rejected_sec,
    )


def screen_all_points(results: list,
                      candidates: list,
                      scr_cfg: dict,
                      cw_supply_F: float,
                      cw_max_gpm: float,
                      cw_dT_design: float) -> list:
    """
    Screen every sample point result dict and return a list of PointScreenResult.
    Also prints a brief per-point progress line to stdout.
    """
    screen_results = []
    for r in results:
        sr = screen_single_point(r, candidates, scr_cfg, cw_supply_F,
                                 cw_max_gpm, cw_dT_design)
        screen_results.append(sr)
    return screen_results

# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------

def _fmt_f(val, decimals=1) -> str:
    """Format a float or return 'N/A' if None."""
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def _fmt_c(val, decimals=0) -> str:
    """Format a float with commas or return 'N/A' if None."""
    return f"{val:,.{decimals}f}" if val is not None else "N/A"


def print_screen_report(screen_results: list, scr_cfg: dict,
                        cw_supply_F: float, cw_dT_design: float) -> None:
    """
    Print a structured engineering screening report to stdout.

    Structure
    ---------
    1. System summary (totals, stats, candidates with no match)
    2. Per-point blocks:
       - Conditions and duties
       - Utility limit check
       - Required UA and area per stage
       - Steam zone table (steam points only)
       - Ranked passing candidates (primary and secondary)
       - Rejected candidates with failure reasons
    """
    chw_in   = float(scr_cfg.get("chiller_supply_F", 55.0))
    chw_out  = float(scr_cfg.get("chiller_return_F", 65.0))
    margin   = float(scr_cfg.get("ua_margin_factor",  1.0))

    W = 120   # report column width

    # ----------------------------------------------------------------
    # System summary
    # ----------------------------------------------------------------
    total_chiller  = sum(sr.Q_secondary_required for sr in screen_results)
    steam_count    = sum(1 for sr in screen_results if sr.is_steam)
    liquid_count   = len(screen_results) - steam_count
    no_match_prim  = sum(1 for sr in screen_results if not sr.passing_primary)
    no_match_sec   = sum(1 for sr in screen_results if not sr.passing_secondary)

    print()
    print("=" * W)
    print("  SWAS COOLER SCREENING REPORT")
    print(f"  U values: DSH={U_DESUPERHEAT_BTU_HR_FT2_F:.0f}  COND={U_CONDENSATION_BTU_HR_FT2_F:.0f}  "
          f"SUBCOOL={U_SUBCOOLING_BTU_HR_FT2_F:.0f}  "
          f"LIQ-HOT={U_LIQUID_HOT_BTU_HR_FT2_F:.0f}  "
          f"LIQ-WARM={U_LIQUID_WARM_BTU_HR_FT2_F:.0f}  "
          f"LIQ-COOL={U_LIQUID_COOL_BTU_HR_FT2_F:.0f}  "
          f"[Btu/hr·ft²·°F, conservative]")
    print(f"  CHW secondary : {chw_in:.0f} F supply / {chw_out:.0f} F return")
    print(f"  Area margin   : {margin:.2f}x  (candidate area >= required * {margin:.2f})")
    print("=" * W)
    print(f"  Sample points       : {len(screen_results):>6d}")
    print(f"  Steam (condensing)  : {steam_count:>6d}")
    print(f"  Liquid (single-ph.) : {liquid_count:>6d}")
    print(f"  Total chiller load  : {total_chiller:>12,.0f} Btu/hr  ({total_chiller/12000:.2f} tons)")
    print(f"  Points — no primary match   : {no_match_prim:>6d}")
    print(f"  Points — no secondary match : {no_match_sec:>6d}")
    print("=" * W)

    # ----------------------------------------------------------------
    # Per-point blocks
    # ----------------------------------------------------------------
    for sr in screen_results:
        print()
        print("-" * W)
        print(f"  SAMPLE POINT : {sr.name}   [{sr.phase}]")
        print(f"  Conditions   : T_in={sr.T_in_F:.0f} F   P={sr.P_psig:.0f} psig   "
              f"T_prim={sr.T_prim_F:.1f} F   T_out={sr.T_out_F:.0f} F")
        print(f"  {sr.service_type_note}")
        print()

        # Duties and utility limit
        lim_flag = " ** UTILITY-LIMITED **" if sr.primary_utility_limited else ""
        print(f"  PRIMARY   Q_required = {sr.Q_primary_required:>12,.0f} Btu/hr  "
              f"|  Q_max_CW = {sr.Q_primary_max_utility:>12,.0f} Btu/hr{lim_flag}")
        print(f"  SECONDARY Q_required = {sr.Q_secondary_required:>12,.0f} Btu/hr  "
              f"(sample {sr.T_prim_F:.1f} F → {sr.T_out_F:.0f} F  using {chw_in:.0f}/{chw_out:.0f} F CHW)")
        print()

        # Required area summary
        print(f"  Required heat-transfer area:")
        print(f"    Primary   : UA = {_fmt_c(sr.ua_primary_required)} Btu/hr·°F   "
              f"A = {_fmt_f(sr.A_primary_required_ft2, 3)} ft²")
        print(f"    Secondary : UA = {_fmt_c(sr.ua_secondary_required)} Btu/hr·°F   "
              f"A = {_fmt_f(sr.A_secondary_required_ft2, 3)} ft²")

        # Steam zone table
        if sr.steam_zones:
            print()
            print(f"  Steam zone breakdown (primary cooler):")
            hdr = f"    {'Zone':<14}  {'Q (Btu/hr)':>12}  {'LMTD (F)':>9}  "
            hdr += f"{'U':>6}  {'UA (Btu/hr°F)':>14}  {'A_req (ft²)':>11}"
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))
            for z in sr.steam_zones:
                row = (f"    {z.zone_name:<14}  {_fmt_c(z.Q_BTU_hr):>12}  "
                       f"{_fmt_f(z.lmtd_F, 1):>9}  "
                       f"{z.U_assumed:>6.0f}  {_fmt_c(z.UA_required, 0):>14}  "
                       f"{_fmt_f(z.A_required_ft2, 3):>11}")
                note = f"  [{z.note}]" if z.note else ""
                print(row + note)
            total_A = sum(z.A_required_ft2 for z in sr.steam_zones if z.A_required_ft2)
            total_UA = sum(z.UA_required for z in sr.steam_zones if z.UA_required)
            print(f"    {'TOTAL':<14}  {_fmt_c(sum(z.Q_BTU_hr for z in sr.steam_zones)):>12}  "
                  f"{'':>9}  {'':>6}  {_fmt_c(total_UA, 0):>14}  {_fmt_f(total_A, 3):>11}")

        # Passing primary candidates
        print()
        if sr.passing_primary:
            print(f"  PRIMARY STAGE — {len(sr.passing_primary)} passing candidate(s)  "
                  f"[sorted by area closest-fit first]:")
            print(f"    {'Model':<30}  {'Svc':>4}  {'MaxP':>5}  {'MaxT':>5}  "
                  f"{'CandA (ft²)':>11}  {'ReqA (ft²)':>10}  {'Oversize':>8}  "
                  f"{'eNTU T_out':>10}  Notes")
            print("    " + "-" * 112)
            for m in sr.passing_primary:
                c = m.candidate
                ov_str = f"{m.oversize_factor:.2f}x" if m.oversize_factor is not None else "  N/A"
                to_str = f"{m.T_predicted_out_F:.1f} F" if m.T_predicted_out_F is not None else "N/A"
                cand_a = _fmt_f(c.heat_transfer_area_ft2, 3) if c.heat_transfer_area_ft2 else "N/A"
                print(f"    {c.model_name:<30}  {c.service_type:>4}  "
                      f"{_fmt_f(c.max_pressure_psig, 0):>5}  "
                      f"{_fmt_f(c.max_temp_F, 0):>5}  "
                      f"{cand_a:>11}  {_fmt_f(m.A_required_ft2, 3):>10}  "
                      f"{ov_str:>8}  {to_str:>10}  {m.notes[:40]}")
        else:
            print(f"  PRIMARY STAGE — NO passing candidates.  Review pressure/temperature ratings or area.")

        # Rejected primary candidates
        if sr.rejected_primary:
            print()
            print(f"  PRIMARY STAGE — {len(sr.rejected_primary)} rejected candidate(s):")
            for rj in sr.rejected_primary:
                print(f"    FAIL  {rj.candidate.model_name:<30}  {rj.fail_reason}")

        # Passing secondary candidates
        print()
        if sr.passing_secondary:
            print(f"  SECONDARY (CHILLER) STAGE — {len(sr.passing_secondary)} passing candidate(s)  "
                  f"[sorted by area closest-fit first]:")
            print(f"    {'Model':<30}  {'Svc':>4}  {'MaxP':>5}  {'MaxT':>5}  "
                  f"{'CandA (ft²)':>11}  {'ReqA (ft²)':>10}  {'Oversize':>8}  "
                  f"{'eNTU T_out':>10}  Notes")
            print("    " + "-" * 112)
            for m in sr.passing_secondary:
                c = m.candidate
                ov_str = f"{m.oversize_factor:.2f}x" if m.oversize_factor is not None else "  N/A"
                to_str = f"{m.T_predicted_out_F:.1f} F" if m.T_predicted_out_F is not None else "N/A"
                cand_a = _fmt_f(c.heat_transfer_area_ft2, 3) if c.heat_transfer_area_ft2 else "N/A"
                print(f"    {c.model_name:<30}  {c.service_type:>4}  "
                      f"{_fmt_f(c.max_pressure_psig, 0):>5}  "
                      f"{_fmt_f(c.max_temp_F, 0):>5}  "
                      f"{cand_a:>11}  {_fmt_f(m.A_required_ft2, 3):>10}  "
                      f"{ov_str:>8}  {to_str:>10}  {m.notes[:40]}")
        else:
            print(f"  SECONDARY (CHILLER) STAGE — NO passing candidates.  "
                  f"Consider larger cooler or higher CHW flow.")

        # Rejected secondary candidates
        if sr.rejected_secondary:
            print()
            print(f"  SECONDARY (CHILLER) STAGE — {len(sr.rejected_secondary)} rejected candidate(s):")
            for rj in sr.rejected_secondary:
                print(f"    FAIL  {rj.candidate.model_name:<30}  {rj.fail_reason}")

    print()
    print("=" * W)
    print(f"  NOTE: All heat-transfer area values are FIRST-PASS SCREENING ESTIMATES only.")
    print(f"  U values are conservative; candidate areas are PLACEHOLDERS pending spec-sheet confirmation.")
    print(f"  Final cooler selection MUST be confirmed against current Sentry spec sheets and vendor sizing.")
    print("=" * W)
    print()

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_screen_csv(screen_results: list, output_path: str) -> None:
    """
    Export one row per (sample point × stage × candidate) with pass/fail and reasons.
    """
    fields = [
        "Sample Point", "Phase", "Is Steam", "Stage",
        "T_in (F)", "P (psig)", "T_prim (F)", "T_out (F)",
        "Q_required (Btu/hr)", "Q_utility_max (Btu/hr)", "Utility Limited",
        "UA_required (Btu/hr·F)", "A_required (ft2)",
        "Candidate Model", "Service Type",
        "Candidate Max P (psig)", "Candidate Max T (F)", "Candidate Area (ft2)",
        "Pass/Fail", "Fail Reason",
        "Oversize Factor", "eNTU Predicted T_out (F)", "Candidate Notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()

        for sr in screen_results:
            base = {
                "Sample Point":           sr.name,
                "Phase":                  sr.phase,
                "Is Steam":               str(sr.is_steam),
                "T_in (F)":               sr.T_in_F,
                "P (psig)":               sr.P_psig,
                "T_prim (F)":             sr.T_prim_F,
                "T_out (F)":              sr.T_out_F,
            }

            def write_stage(stage_name, Q_req, Q_util_max, util_lim,
                            ua_req, a_req, matches, rejects):
                for m in matches:
                    c = m.candidate
                    row = {**base,
                        "Stage":                    stage_name,
                        "Q_required (Btu/hr)":      round(Q_req),
                        "Q_utility_max (Btu/hr)":   round(Q_util_max),
                        "Utility Limited":          str(util_lim),
                        "UA_required (Btu/hr·F)":   _fmt_f(ua_req, 1),
                        "A_required (ft2)":         _fmt_f(a_req, 3),
                        "Candidate Model":          c.model_name,
                        "Service Type":             c.service_type,
                        "Candidate Max P (psig)":   _fmt_f(c.max_pressure_psig, 0),
                        "Candidate Max T (F)":      _fmt_f(c.max_temp_F, 0),
                        "Candidate Area (ft2)":     _fmt_f(c.heat_transfer_area_ft2, 3),
                        "Pass/Fail":                "PASS",
                        "Fail Reason":              "",
                        "Oversize Factor":          _fmt_f(m.oversize_factor, 2),
                        "eNTU Predicted T_out (F)": _fmt_f(m.T_predicted_out_F, 1),
                        "Candidate Notes":          c.notes[:80],
                    }
                    w.writerow(row)
                for rj in rejects:
                    c = rj.candidate
                    row = {**base,
                        "Stage":                    stage_name,
                        "Q_required (Btu/hr)":      round(Q_req),
                        "Q_utility_max (Btu/hr)":   round(Q_util_max),
                        "Utility Limited":          str(util_lim),
                        "UA_required (Btu/hr·F)":   _fmt_f(ua_req, 1),
                        "A_required (ft2)":         _fmt_f(a_req, 3),
                        "Candidate Model":          c.model_name,
                        "Service Type":             c.service_type,
                        "Candidate Max P (psig)":   _fmt_f(c.max_pressure_psig, 0),
                        "Candidate Max T (F)":      _fmt_f(c.max_temp_F, 0),
                        "Candidate Area (ft2)":     _fmt_f(c.heat_transfer_area_ft2, 3),
                        "Pass/Fail":                "FAIL",
                        "Fail Reason":              rj.fail_reason,
                        "Oversize Factor":          "",
                        "eNTU Predicted T_out (F)": "",
                        "Candidate Notes":          c.notes[:80],
                    }
                    w.writerow(row)

            write_stage("PRIMARY",   sr.Q_primary_required,   sr.Q_primary_max_utility,
                        sr.primary_utility_limited, sr.ua_primary_required,
                        sr.A_primary_required_ft2,  sr.passing_primary, sr.rejected_primary)
            write_stage("SECONDARY", sr.Q_secondary_required, sr.Q_primary_max_utility,
                        False, sr.ua_secondary_required,
                        sr.A_secondary_required_ft2, sr.passing_secondary, sr.rejected_secondary)

    print(f"  Screening results exported to: {Path(output_path).resolve()}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWAS Sample Cooler Screening Tool — screens candidate coolers "
                    "against calculated heat duties from swas_heat_duty.")
    parser.add_argument("--config",        default="swas_config.yaml",
                        help="Path to shared YAML config (default: swas_config.yaml)")
    parser.add_argument("--output",        default="swas_results.csv",
                        help="Heat duty CSV output path (default: swas_results.csv)")
    parser.add_argument("--screen-output", default="swas_screen_results.csv",
                        help="Screening results CSV (default: swas_screen_results.csv)")
    args = parser.parse_args()

    # ----- Load config -----
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"ERROR: config not found: {cfg_path.resolve()}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not cfg.get("sample_points"):
        sys.exit("ERROR: no sample_points in config.")

    # ----- Extract system parameters -----
    target_F  = float(cfg.get("target_temp_F",            77.0))
    cw_F      = float(cfg.get("cw_supply_temp_F",         90.0))
    cw_gpm    = float(cfg.get("cw_max_flow_gpm",          12.0))
    cw_dT     = float(cfg.get("cw_design_delta_T_F",      20.0))
    sc_marg   = float(cfg.get("subcool_margin_F",         10.0))
    liq_app   = float(cfg.get("liquid_approach_F",         5.0))
    an_psig   = float(cfg.get("analyzer_pressure_psig",   20.0))
    an_min    = float(cfg.get("analyzer_min_flow_cc_min", 150.0))
    an_max    = float(cfg.get("analyzer_max_flow_cc_min",1000.0))
    def_flow  = float(cfg.get("default_flow_lbm_per_min",  6.6))
    tube_od   = float(cfg.get("transport_tube_od_in",     0.375))
    tube_wall = float(cfg.get("transport_tube_wall_in",   0.065))

    scr_cfg   = cfg.get("screening", {})

    # ----- Locate candidate database -----
    cand_file = scr_cfg.get("candidates_file", "cooler_candidates.json")
    cand_path = Path(cand_file)
    if not cand_path.is_absolute():
        cand_path = cfg_path.parent / cand_path
    if not cand_path.exists():
        sys.exit(f"ERROR: candidates file not found: {cand_path.resolve()}")

    # ----- Run heat duty calculation for all sample points -----
    # Store resolved config back for downstream use
    cfg.update({"target_temp_F": target_F, "cw_supply_temp_F": cw_F,
                "cw_max_flow_gpm": cw_gpm, "cw_design_delta_T_F": cw_dT,
                "subcool_margin_F": sc_marg, "liquid_approach_F": liq_app,
                "analyzer_pressure_psig": an_psig})

    results = []
    for sp in cfg["sample_points"]:
        try:
            r = calc_point(
                str(sp["name"]),
                float(sp["temp_F"]),
                float(sp["pressure_psig"]),
                target_F,
                float(sp.get("flow_lbm_per_min", def_flow)),
                sp.get("phase_hint", "auto"),
                cw_F, cw_gpm, cw_dT, sc_marg, liq_app,
                an_psig, an_min, an_max,
                tube_od, tube_wall,
            )
            results.append(r)
        except Exception as exc:
            print(f"WARNING: Failed to calculate {sp['name']!r}: {exc}")

    if not results:
        sys.exit("ERROR: No sample points calculated.")

    # ----- Load candidates -----
    candidates = load_candidates(str(cand_path))
    print(f"  Loaded {len(candidates)} candidate coolers from {cand_path.name}")

    # ----- Screen all points -----
    screen_results = screen_all_points(results, candidates, scr_cfg,
                                       cw_F, cw_gpm, cw_dT)

    # ----- Output -----
    print_screen_report(screen_results, scr_cfg, cw_F, cw_dT)
    export_screen_csv(screen_results, args.screen_output)

    # ----- Feasibility & auto-adjustment -----
    if scr_cfg.get("feasibility_enabled", False):
        from swas_feasibility import (
            assess_all_points,
            print_feasibility_report,
            export_feasibility_csv,
            export_feasibility_excel,
        )
        feas_results = assess_all_points(results, candidates, scr_cfg,
                                         cw_F, cw_gpm, cw_dT)
        print_feasibility_report(feas_results)
        feas_output = scr_cfg.get("feasibility_output", "swas_feasibility_results.csv")
        export_feasibility_csv(feas_results, feas_output)
        # Excel report
        excel_output = feas_output.replace(".csv", ".xlsx") if feas_output.endswith(".csv") \
            else feas_output + ".xlsx"
        try:
            export_feasibility_excel(feas_results, excel_output, cfg=cfg)
        except PermissionError:
            print(f"\n  WARNING: Could not write {excel_output} — file may be open in another program.")
            print(f"  Close the file and re-run, or CSV results are available in {feas_output}.")


if __name__ == "__main__":
    main()
