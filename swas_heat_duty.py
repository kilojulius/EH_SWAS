#!/usr/bin/env python3
"""
SWAS Sample Cooler Heat Duty Calculator -- Two-Stage Cooling
=============================================================
Stage 1 - Primary cooler (CW):
  Steam points : desuperheat + condensation + subcool to
                 T_sat(analyzer_pressure) - subcool_margin_F.
                 Slip-stream flow is sized so CW duty <= cw_max_flow_gpm
                 * cw_design_delta_T_F limit.
  Liquid points: sensible cooling to cw_supply_temp_F + liquid_approach_F
                 at full transport flow.

Stage 2 - Chiller cooler: sensible subcooling from primary outlet to
          target_temp_F (77 F) on the cooler flow (slip or full).

Usage:
    python swas_heat_duty.py
    python swas_heat_duty.py --config my_config.yaml --output results.csv
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import yaml
from CoolProp.CoolProp import PropsSI
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Unit conversions  (CoolProp uses SI internally: K, Pa, J/kg, kg/s)
# ---------------------------------------------------------------------------

def F_to_K(T):
    return (T - 32.0) * 5.0 / 9.0 + 273.15

def K_to_F(T):
    return (T - 273.15) * 9.0 / 5.0 + 32.0

def psig_to_Pa(P):
    return (P + 14.696) * 6894.7572932

def lbm_min_to_kg_s(f):
    return f * 0.45359237 / 60.0

def kg_s_to_lbm_min(f):
    return f / 0.45359237 * 60.0

def W_to_BTU_hr(Q):
    return Q * 3.412141633

def lbm_min_to_cc_min(lbm_min, T_F=77.0, P_psig=20.0):
    """Convert lb/min to cc/min using CoolProp liquid water density."""
    rho = PropsSI("D", "T", F_to_K(T_F), "P", psig_to_Pa(P_psig), "Water")
    return (lbm_min * 0.45359237 / rho) * 1.0e6


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

SUPERHEATED   = "Superheated"
SATURATED     = "Saturated Vapor"
SUBCOOLED     = "Subcooled Liquid"
SUPERCRITICAL = "Supercritical"
STEAM_PHASES  = {SUPERHEATED, SATURATED, SUPERCRITICAL}

SAT_TOL_F = 5.0
P_CRIT_PA = 22_064_000.0    # water critical pressure ~3206 psia


def classify(T_in_F, P_psig, hint="auto"):
    """Return (phase, T_sat_F).  T_sat_F is None for supercritical."""
    P_Pa = psig_to_Pa(P_psig)
    if P_Pa >= P_CRIT_PA:
        return SUPERCRITICAL, None
    T_sat_F = K_to_F(PropsSI("T", "P", P_Pa, "Q", 0, "Water"))
    if hint == "liquid":
        return SUBCOOLED, T_sat_F
    if hint == "steam":
        return (SUPERHEATED if T_in_F > T_sat_F + SAT_TOL_F else SATURATED), T_sat_F
    if T_in_F > T_sat_F + SAT_TOL_F:
        return SUPERHEATED, T_sat_F
    if T_in_F >= T_sat_F - SAT_TOL_F:
        return SATURATED, T_sat_F
    return SUBCOOLED, T_sat_F


# ---------------------------------------------------------------------------
# Heat duty calculation for a single sample point
# ---------------------------------------------------------------------------

def calc_point(name, T_in_F, P_psig, T_out_F, flow_lbm_min, hint,
               cw_supply_F, cw_max_gpm, cw_dT_design,
               subcool_margin, liquid_approach,
               an_psig, an_min_cc, an_max_cc,
               tube_od_in=0.375, tube_wall_in=0.065):
    """
    Returns a result dict with all heat duty components in BTU/hr plus
    a list of engineering flags.
    """
    P_Pa            = psig_to_Pa(P_psig)
    T_in_K          = F_to_K(T_in_F)
    T_out_K         = F_to_K(T_out_F)
    m_dot_transport = lbm_min_to_kg_s(flow_lbm_min)

    phase, T_sat_F = classify(T_in_F, P_psig, hint)

    # Max heat the CW primary cooler can absorb given flow & dT limits (W)
    Q_prim_max_W = (cw_max_gpm * 500.0 * cw_dT_design) / 3.412141633

    h_out    = PropsSI("H", "T", T_out_K, "P", P_Pa, "Water")
    flags    = []
    Q_dsh_W  = 0.0
    Q_cond_W = 0.0

    # ----------------------------------------------------------------
    # STEAM PATH
    # Primary outlet = T_sat(analyzer panel pressure) - subcool_margin
    # Slip-stream m_dot sized to CW heat capacity limit
    # ----------------------------------------------------------------
    if phase in STEAM_PHASES:
        P_an_Pa  = psig_to_Pa(an_psig)
        T_prim_F = K_to_F(PropsSI("T", "P", P_an_Pa, "Q", 0, "Water")) - subcool_margin
        T_prim_K = F_to_K(T_prim_F)
        h_prim   = PropsSI("H", "T", T_prim_K, "P", P_Pa, "Water")

        if phase == SUPERHEATED:
            h_in = PropsSI("H", "T", T_in_K, "P", P_Pa, "Water")
            h_g  = PropsSI("H", "P", P_Pa, "Q", 1, "Water")
            h_f  = PropsSI("H", "P", P_Pa, "Q", 0, "Water")
            h_in_eff = h_in
            # Phase-component duties at full transport flow -- scaled to slip below
            Q_dsh_full  = m_dot_transport * (h_in - h_g)
            Q_cond_full = m_dot_transport * (h_g  - h_f)
        elif phase == SATURATED:
            h_g = PropsSI("H", "P", P_Pa, "Q", 1, "Water")
            h_f = PropsSI("H", "P", P_Pa, "Q", 0, "Water")
            h_in_eff    = h_g
            Q_dsh_full  = 0.0
            Q_cond_full = m_dot_transport * (h_g - h_f)
        else:   # SUPERCRITICAL
            h_in_eff    = PropsSI("H", "T", T_in_K, "P", P_Pa, "Water")
            Q_dsh_full  = 0.0
            Q_cond_full = 0.0

        h_drop      = h_in_eff - h_prim          # J/kg across primary cooler
        m_dot_slip  = min(Q_prim_max_W / h_drop, m_dot_transport)
        slip_frac   = m_dot_slip / m_dot_transport

        # Scale phase-component duties to actual slip flow
        Q_dsh_W  = Q_dsh_full  * slip_frac
        Q_cond_W = Q_cond_full * slip_frac

        Q_prim_W     = m_dot_slip * h_drop
        Q_chill_W    = m_dot_slip * (h_prim - h_out)
        actual_cw_dT = W_to_BTU_hr(Q_prim_W) / (cw_max_gpm * 500.0)

        # Subcooling path: sensible heat from sat liquid → 77 °F outlet (spans both stages)
        if phase in (SUPERHEATED, SATURATED):
            Q_subcool_path_W = m_dot_slip * (h_f - h_out)
        else:   # SUPERCRITICAL — no saturation state; all heat is sensible
            Q_subcool_path_W = Q_prim_W + Q_chill_W

        if slip_frac < 0.999:
            flags.append(
                f"Slip stream: {slip_frac * 100:.0f}% of transport flow "
                f"({kg_s_to_lbm_min(m_dot_slip):.2f} lb/min) to cooler")

    # ----------------------------------------------------------------
    # LIQUID PATH
    # Full transport flow; primary outlet = CW supply + approach temp
    # ----------------------------------------------------------------
    else:
        # Clamp to h_f if T_in is at or above T_sat (prevents CoolProp
        # returning vapor-side enthalpy for a physically liquid sample)
        if T_sat_F is not None and T_in_F >= T_sat_F:
            h_in_eff = PropsSI("H", "P", P_Pa, "Q", 0, "Water")
        else:
            h_in_eff = PropsSI("H", "T", T_in_K, "P", P_Pa, "Water")

        T_prim_F = cw_supply_F + liquid_approach
        if T_in_F <= T_prim_F:
            # Sample already below primary target; CW stage does nothing
            T_prim_F     = T_in_F
            h_prim       = h_in_eff
            Q_prim_W     = 0.0
            actual_cw_dT = 0.0
        else:
            h_prim       = PropsSI("H", "T", F_to_K(T_prim_F), "P", P_Pa, "Water")
            Q_prim_W     = m_dot_transport * (h_in_eff - h_prim)
            actual_cw_dT = W_to_BTU_hr(Q_prim_W) / (cw_max_gpm * 500.0)

        Q_chill_W  = m_dot_transport * (h_prim - h_out)
        m_dot_slip = m_dot_transport
        # All heat removal is subcooling for liquid points (no phase change)
        Q_subcool_path_W = m_dot_transport * (h_in_eff - h_out)

    Q_total_W = Q_prim_W + Q_chill_W

    # Analyzer flow (density at analyzer conditions)
    slip_lbm = kg_s_to_lbm_min(m_dot_slip)
    an_cc    = lbm_min_to_cc_min(slip_lbm)

    # Flag analyzer flow only for steam slip-stream paths
    if phase in STEAM_PHASES:
        if an_cc < an_min_cc:
            flags.append(
                f"Analyzer flow {an_cc:.0f} cc/min < min {an_min_cc:.0f} cc/min")
        if an_cc > an_max_cc:
            flags.append(
                f"Analyzer flow {an_cc:.0f} cc/min > max {an_max_cc:.0f} cc/min")

    if actual_cw_dT > cw_dT_design + 0.5:
        flags.append(
            f"CW rise {actual_cw_dT:.1f} F exceeds design {cw_dT_design:.0f} F")

    # Transport line velocity at inlet conditions (full transport flow, actual fluid)
    tube_id_ft    = (tube_od_in - 2.0 * tube_wall_in) / 12.0
    tube_area_ft2 = math.pi / 4.0 * tube_id_ft ** 2
    rho_in        = (PropsSI("D", "P", P_Pa, "Q", 1, "Water") if phase == SATURATED
                     else PropsSI("D", "T", T_in_K, "P", P_Pa, "Water"))
    velocity_fps  = (m_dot_transport / rho_in) * 35.3147 / tube_area_ft2

    return {
        "name":           name,
        "phase":          phase,
        "T_in_F":         T_in_F,
        "P_psig":         P_psig,
        "T_sat_F":        round(T_sat_F, 1) if T_sat_F is not None else None,
        "T_prim_F":       round(T_prim_F, 1),
        "T_out_F":        T_out_F,
        "flow_transport": flow_lbm_min,
        "flow_slip":      round(slip_lbm, 2),
        "an_cc":          round(an_cc),
        "is_steam":       phase in STEAM_PHASES,
        "Q_dsh":          round(W_to_BTU_hr(Q_dsh_W)),
        "Q_cond":         round(W_to_BTU_hr(Q_cond_W)),
        "Q_prim":         round(W_to_BTU_hr(Q_prim_W)),
        "cw_dT":          round(actual_cw_dT, 1),
        "Q_chill":        round(W_to_BTU_hr(Q_chill_W)),
        "Q_total":        round(W_to_BTU_hr(Q_total_W)),
        "velocity_fps":   round(velocity_fps, 1),
        "Q_subcool_path": round(W_to_BTU_hr(Q_subcool_path_W)),
        "flags":          flags,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_results(results, cfg):
    T_tgt   = cfg["target_temp_F"]
    cw_F    = cfg["cw_supply_temp_F"]
    cw_gpm  = cfg["cw_max_flow_gpm"]
    cw_dT   = cfg["cw_design_delta_T_F"]
    sc_marg = cfg["subcool_margin_F"]
    liq_app = cfg["liquid_approach_F"]
    an_psig = cfg["analyzer_pressure_psig"]

    T_sat_an  = K_to_F(PropsSI("T", "P", psig_to_Pa(an_psig), "Q", 0, "Water"))
    T_prim_sh = T_sat_an - sc_marg
    T_prim_lq = cw_F + liq_app

    hdrs = [
        "Sample Point", "Phase",
        "T_in\n(F)", "P\n(psig)", "T_sat\n(F)", "T_prim\nout (F)",
        "Cooler\nFlow\n(lb/min)", "Analyzer\nFlow\n(cc/min)",
        "Velocity\n(fps)",
        "Q_dsh\n(BTU/hr)", "Q_cond\n(BTU/hr)",
        "Q_primary\n[CW]\n(BTU/hr)", "CW\ndT (F)",
        "Q_chiller\n(BTU/hr)", "Q_total\n(BTU/hr)",
    ]

    rows = []
    for r in results:
        an_disp = (f"{r['an_cc']:,}" if r["is_steam"]
                   else f"{r['an_cc']:,} *")
        rows.append([
            r["name"], r["phase"],
            f"{r['T_in_F']:.0f}", f"{r['P_psig']:.0f}",
            f"{r['T_sat_F']:.1f}" if r["T_sat_F"] is not None else "N/A",
            f"{r['T_prim_F']:.1f}",
            f"{r['flow_slip']:.2f}", an_disp,
            f"{r['velocity_fps']:.1f}",
            f"{r['Q_dsh']:,.0f}", f"{r['Q_cond']:,.0f}",
            f"{r['Q_prim']:,.0f}", f"{r['cw_dT']:.1f}",
            f"{r['Q_chill']:,.0f}", f"{r['Q_total']:,.0f}",
        ])

    tot_dsh   = sum(r["Q_dsh"]   for r in results)
    tot_cond  = sum(r["Q_cond"]  for r in results)
    tot_prim  = sum(r["Q_prim"]  for r in results)
    tot_chill = sum(r["Q_chill"] for r in results)
    tot_all   = sum(r["Q_total"] for r in results)

    rows.append([
        "-- TOTAL --", "", "", "", "", "", "", "", "",
        f"{tot_dsh:,.0f}", f"{tot_cond:,.0f}",
        f"{tot_prim:,.0f}", "",
        f"{tot_chill:,.0f}", f"{tot_all:,.0f}",
    ])

    W = 140
    print()
    print("=" * W)
    print("  SWAS SAMPLE COOLER HEAT DUTY -- TWO-STAGE COOLING")
    print(f"  Stage 1  Primary CW Cooler : {cw_gpm:.0f} GPM max @ {cw_F:.0f} F supply,"
          f" design dT {cw_dT:.0f} F  =>  Q_prim_max = {cw_gpm*500*cw_dT:,.0f} BTU/hr per cooler")
    print(f"           Steam outlet      : T_sat({an_psig:.0f} psig) - {sc_marg:.0f} F"
          f" = {T_prim_sh:.1f} F  (slip-stream sized to CW cap)")
    print(f"           Liquid outlet     : {cw_F:.0f} F CW + {liq_app:.0f} F approach"
          f" = {T_prim_lq:.0f} F  (full transport flow)")
    print(f"  Stage 2  Chiller Cooler    : sensible subcooling -> {T_tgt:.0f} F")
    print("=" * W)
    print()
    print(tabulate(rows, headers=hdrs, tablefmt="simple",
                   stralign="right", numalign="right"))
    print()
    print("  * Liquid point: full transport flow through cooler;"
          " actual analyzer flow is PCV-controlled at panel (<=1000 cc/min).")
    print()

    # ------------------------------------------------------------------
    # Q Thermodynamic Pathway table
    # ------------------------------------------------------------------
    path_hdrs = [
        "Sample Point",
        "Q_desuperheat\n(BTU/hr)",
        "Q_latent (cond)\n(BTU/hr)",
        "Q_subcool (liq\u219277F)\n(BTU/hr)",
        "Q_total\n(BTU/hr)",
        "Check",
    ]
    path_rows = []
    for r in results:
        q_sum = r["Q_dsh"] + r["Q_cond"] + r["Q_subcool_path"]
        chk   = "OK" if abs(q_sum - r["Q_total"]) <= 5 else f"DIFF {q_sum - r['Q_total']:+,.0f}"
        path_rows.append([
            r["name"],
            f"{r['Q_dsh']:,.0f}",
            f"{r['Q_cond']:,.0f}",
            f"{r['Q_subcool_path']:,.0f}",
            f"{r['Q_total']:,.0f}",
            chk,
        ])
    tot_subco = sum(r["Q_subcool_path"] for r in results)
    path_rows.append([
        "-- TOTAL --",
        f"{tot_dsh:,.0f}",
        f"{tot_cond:,.0f}",
        f"{tot_subco:,.0f}",
        f"{tot_all:,.0f}",
        "",
    ])
    print("  Q THERMODYNAMIC PATHWAY  (desuperheat \u2192 condensation \u2192 subcooling, scaled to cooler flow)")
    print()
    print(tabulate(path_rows, headers=path_hdrs, tablefmt="simple",
                   stralign="right", numalign="right"))
    print()

    print("-" * W)
    print(f"  STAGE 1  Primary CW Cooler : {tot_prim:>12,.0f} BTU/hr")
    print(f"  STAGE 2  Chiller           : {tot_chill:>12,.0f} BTU/hr"
          f"  ({tot_chill / 12_000:.2f} tons)")
    print(f"  COMBINED                   : {tot_all:>12,.0f} BTU/hr"
          f"  ({tot_all / 12_000:.2f} tons)")
    print(f"  Sample Points              : {len(results):>12d}")
    print("-" * W)
    print()

    all_flags = [(r["name"], fl) for r in results for fl in r["flags"]]
    if all_flags:
        print("  NOTES / FLAGS:")
        for pt, fl in all_flags:
            print(f"    [{pt}]  {fl}")
        print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(results, output_path, cfg):
    fields = [
        "Sample Point", "Phase", "T_in (F)", "P (psig)", "T_sat (F)",
        "T_primary_out (F)", "T_out (F)",
        "Transport Flow (lb/min)", "Cooler Flow (lb/min)", "Analyzer Flow (cc/min)",
        "Velocity (fps)",
        "Q_desuperheat (BTU/hr)", "Q_condense (BTU/hr)",
        "Q_primary_CW (BTU/hr)", "CW_dT (F)",
        "Q_chiller (BTU/hr)", "Q_total (BTU/hr)",
        "Q_subcool_path (BTU/hr)", "Flags",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({
                "Sample Point":            r["name"],
                "Phase":                   r["phase"],
                "T_in (F)":                r["T_in_F"],
                "P (psig)":                r["P_psig"],
                "T_sat (F)":               r["T_sat_F"] if r["T_sat_F"] is not None else "N/A",
                "T_primary_out (F)":       r["T_prim_F"],
                "T_out (F)":               r["T_out_F"],
                "Transport Flow (lb/min)": r["flow_transport"],
                "Cooler Flow (lb/min)":    r["flow_slip"],
                "Analyzer Flow (cc/min)":  r["an_cc"],
                "Velocity (fps)":          r["velocity_fps"],
                "Q_desuperheat (BTU/hr)":  r["Q_dsh"],
                "Q_condense (BTU/hr)":     r["Q_cond"],
                "Q_primary_CW (BTU/hr)":   r["Q_prim"],
                "CW_dT (F)":               r["cw_dT"],
                "Q_chiller (BTU/hr)":      r["Q_chill"],
                "Q_total (BTU/hr)":        r["Q_total"],
                "Q_subcool_path (BTU/hr)": r["Q_subcool_path"],
                "Flags":                   " | ".join(r["flags"]),
            })
        # Totals row
        w.writerow({
            "Sample Point": "TOTAL", "Phase": "",
            "T_in (F)": "", "P (psig)": "", "T_sat (F)": "",
            "T_primary_out (F)": "", "T_out (F)": cfg["target_temp_F"],
            "Transport Flow (lb/min)": "", "Cooler Flow (lb/min)": "",
            "Analyzer Flow (cc/min)": "",
            "Q_desuperheat (BTU/hr)": sum(r["Q_dsh"]   for r in results),
            "Q_condense (BTU/hr)":    sum(r["Q_cond"]  for r in results),
            "Q_primary_CW (BTU/hr)":  sum(r["Q_prim"]  for r in results),
            "CW_dT (F)": "",
            "Q_chiller (BTU/hr)":     sum(r["Q_chill"] for r in results),
            "Q_total (BTU/hr)":       sum(r["Q_total"] for r in results),
            "Q_subcool_path (BTU/hr)": sum(r["Q_subcool_path"] for r in results),
            "Flags": "",
        })
    print(f"  Results exported to: {Path(output_path).resolve()}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWAS Sample Cooler Heat Duty Calculator (CoolProp, two-stage)")
    parser.add_argument("--config", default="swas_config.yaml",
                        help="Path to YAML config (default: swas_config.yaml)")
    parser.add_argument("--output", default="swas_results.csv",
                        help="Path for CSV output (default: swas_results.csv)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"ERROR: config not found: {cfg_path.resolve()}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not cfg.get("sample_points"):
        sys.exit("ERROR: no sample_points in config.")

    target_F   = float(cfg.get("target_temp_F",            77.0))
    cw_F       = float(cfg.get("cw_supply_temp_F",         90.0))
    cw_gpm     = float(cfg.get("cw_max_flow_gpm",          12.0))
    cw_dT      = float(cfg.get("cw_design_delta_T_F",      15.0))
    sc_marg    = float(cfg.get("subcool_margin_F",         20.0))
    liq_app    = float(cfg.get("liquid_approach_F",        20.0))
    an_psig    = float(cfg.get("analyzer_pressure_psig",   20.0))
    an_min     = float(cfg.get("analyzer_min_flow_cc_min", 150.0))
    an_max     = float(cfg.get("analyzer_max_flow_cc_min", 1000.0))
    def_flow   = float(cfg.get("default_flow_lbm_per_min", 6.6))
    tube_od    = float(cfg.get("transport_tube_od_in",      0.375))
    tube_wall  = float(cfg.get("transport_tube_wall_in",    0.065))

    # Store resolved scalars back so output functions can use them
    cfg.update({
        "target_temp_F": target_F, "cw_supply_temp_F": cw_F,
        "cw_max_flow_gpm": cw_gpm, "cw_design_delta_T_F": cw_dT,
        "subcool_margin_F": sc_marg, "liquid_approach_F": liq_app,
        "analyzer_pressure_psig": an_psig,
    })

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

    print_results(results, cfg)
    export_csv(results, args.output, cfg)


if __name__ == "__main__":
    main()
