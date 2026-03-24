"""
Vegetation module for SARAwater

Implements:
- Recruitment bands (Serlet et al. 2023)
- Bar elevation feedback (Zen et al. 2016 concept)
- Deterministic daily vegetation dynamics
  inspired by Camporeale et al. 2006

Works on single cross-section geometry.
"""

import numpy as np
import pandas as pd

# HYDROLOGICAL METRICS
class VegetationParameters:

    def __init__(self,
                 low_flow_percentile=20,
                 ag=0.04,
                 ad=0.08,
                 mg=1.2,
                 md=1.6):
        self.low_flow_percentile = low_flow_percentile
        self.ag = ag
        self.ad = ad
        self.mg = mg
        self.md = md

def compute_cv(Q):
    """Coefficient of variation of discharge."""
    Q = np.asarray(Q)
    return np.std(Q) / np.mean(Q)



# EBE1 — Recruitment elevation band

def compute_EBE1(stage, cv, cv_threshold=0.8):
    """
    Elevation band suitable for recruitment.
    """

    if cv > cv_threshold:
        return None

    base = np.min(stage)

    z_min = base + 0.6
    z_max = min(np.max(stage), base + 2.0)

    return z_min, z_max



# EBE2 — Window of Opportunity


def compute_EBE2(Q, Q_germination, min_days=30):

    low_flow = Q < Q_germination

    windows = []
    count = 0

    for i, val in enumerate(low_flow):

        if val:
            count += 1
        else:
            if count >= min_days:
                windows.append((i-count, i))
            count = 0

    return windows



# EBE3 — Flood mortality

def compute_EBE3(Q, Q_mortality):
    return Q > Q_mortality



# CROSS-SECTION ELEVATION GRID

def elevation_grid(reach, Q, n_points=40,
                   low_flow_percentile=20):
    """
    Generate candidate elevations outside the
    permanently wetted channel.

    Parameters
    ----------
    reach : Reach object
    Q : discharge series
    low_flow_percentile : defines active channel limit
    """

    cross_section = reach.get_cross_section()

    zmin = np.min(cross_section[:,1])
    zmax = np.max(cross_section[:,1])

    # hydraulic definition of active channel
    Q_low = np.percentile(Q, low_flow_percentile)
    z_active = reach.stage_from_discharge(Q_low)

    elevations = np.linspace(z_active, zmax, n_points)

    return elevations


# DAILY BIOMASS MODEL
# (Camporeale-inspired deterministic)


def daily_biomass_model(
        reach,
        Q,
        elevations,
        B0=0.05,
        ag=0.04,
        ad=0.08,
        mg=1.2,
        md=1.6):

    Q = np.asarray(Q)

    biomass = {}

    for z in elevations:

        Qcrit = reach.discharge_from_stage(z)

        B = np.zeros(len(Q))
        B[0] = B0

        for t in range(len(Q)-1):

            if Q[t] < Qcrit:
                growth = ag * (Qcrit - Q[t])**mg
                decay = 0.0
            else:
                growth = 0.0
                decay = ad * (Q[t] - Qcrit)**md

            B[t+1] = max(0, B[t] + growth - decay)

        biomass[z] = B

    return biomass


# BAR ELEVATION FEEDBACK
# (Zen et al. inspired)

def update_bar_elevation(z, biomass,
                         alpha_dep=0.002,
                         beta_ero=0.0015):
    """
    Vegetation increases deposition,
    low biomass allows erosion.
    """

    dz = alpha_dep * biomass - beta_ero * (1 - biomass)

    return z + dz


def simulate_bar_evolution(
        reach,
        Q,
        elevations,
        biomass_dict):

    results = {}

    for z in elevations:

        B = biomass_dict[z]

        Z = np.zeros(len(Q))
        Z[0] = z

        for t in range(len(Q)-1):
            Z[t+1] = update_bar_elevation(Z[t], B[t])

        results[z] = Z

    return results


# VEGETATION MODEL

def run_vegetation_model(reach, Q, dates):

    stage = reach.compute_stage_from_discharge(Q)

    cv = compute_cv(Q)

    ebe1 = compute_EBE1(stage, cv)

    if ebe1 is None:
        return {"recruitment": None}

    Q_germ = np.percentile(Q, 30)
    Q_mort = np.percentile(Q, 90)

    ebe2 = compute_EBE2(Q, Q_germ)
    ebe3 = compute_EBE3(Q, Q_mort)

    elevations = elevation_grid(
        reach.get_cross_section()
    )

    biomass = daily_biomass_model(
        reach,
        Q,
        elevations
    )

    bar_elevation = simulate_bar_evolution(
        reach,
        Q,
        elevations,
        biomass
    )

    return {
        "EBE1": ebe1,
        "EBE2": ebe2,
        "EBE3": ebe3,
        "biomass": biomass,
        "bar_elevation": bar_elevation,
        "cv": cv
    }