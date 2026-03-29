"""
Vegetation module for SARAwater

SERLET (2023)

Original: 2D hydro-morphodynamic ecohydraulics
SARAwater: 1D cross-section reduction

Assumption:
h(x,y,t) ≈ stage(t) − z_cross_section

ZEN et al. (2016)

Original: 2D morphodynamics
SARAWater: elevation feedback proxy

CAMPOREALE (2006)

Original: stochastic SDE vegetation model
SARAWater: deterministic reduced-order analogue
"""

import numpy as np
import pandas as pd

# PARAMETERS OBJECT


class VegetationParameters:

    def __init__(
        self,
        low_flow_percentile=20 #to be calibrated for each site (particularly if arid regions or snowmelt-rivers), but might be good to represent frequently exposed bars,
        flood_percentile=90,
        seasonal_windows=None, # Optional seasonal hydrological windows replacing percentiles like growth: [5,6,7], flood: [10,11,12]
        ag=0.04, #vegetation growth coefficient ->  Controls biomass increase when discharge is below the
        #optimal recrutiment treshold (Qcrit)
        ad=0.08, #Flood-induced decay coefficient -> Controls biomass loss during high-flow stress
        #(uprooting and inundation mortality)
        mg=1.2, #growth exponent -> Defines NONLINEAR sensitivity of growth to flow deficit
        md=1.6, #decay exponent -> Defines NONLINEAR sensitivity of decay to flow excess
    #ag, ad, mg, md TO BE CALIBRATED FOR EACH SITE
    ):
        """
        seasonal_windows example:
        {
            "growth":[4,5,6,7],
            "flood":[10,11,12]
        }
        """

        self.low_flow_percentile = low_flow_percentile
        self.flood_percentile = flood_percentile
        self.seasonal_windows = seasonal_windows

        # Camporeale analogue parameters
        self.ag = ag
        self.ad = ad
        self.mg = mg
        self.md = md



# HYDROLOGICAL METRICS

def compute_cv(Q): #cv = coefficient of variation of Stage (low CV = stable recruitment possible)
    Q = np.asarray(Q)
    return np.std(Q) / np.mean(Q)



# SERLET HYDROLOGICAL WINDOWS


def seasonal_thresholds(Q, dates, params):
    """
    Seasonal alternative to percentiles (Serlet-like).
    """

    if params.seasonal_windows is None:
        low = np.percentile(Q, params.low_flow_percentile)
        high = np.percentile(Q, params.flood_percentile)
        return low, high

    months = np.array([d.month for d in dates])

    growth_mask = np.isin(months, params.seasonal_windows["growth"])
    flood_mask = np.isin(months, params.seasonal_windows["flood"])

    low = np.percentile(Q[growth_mask], 50)
    high = np.percentile(Q[flood_mask], 70)

    return low, high



# RECRUITMENT BANDS (SERLET ADAPTATION)

def compute_EBE1(stage, cv, cv_threshold=0.8):

    if cv > cv_threshold: 
        return None

    base = np.min(stage)

    z_min = base + 0.6
    z_max = min(np.max(stage), base + 2.0)

    return z_min, z_max


def compute_EBE2(Q, Q_germination, min_days=30):

    mask = Q < Q_germination

    windows = []
    count = 0

    for i, val in enumerate(mask):
        if val:
            count += 1
        else:
            if count >= min_days:
                windows.append((i - count, i))
            count = 0

    return windows


def compute_EBE3(Q, Q_mortality):
    return Q > Q_mortality


# CROSS-SECTION REDUCTION (2D → 1D)

def elevation_grid(reach, Q, n_points=40, low_flow_percentile=20):

    cross_section = reach.get_cross_section()

    zmax = np.max(cross_section[:, 1])

    Q_low = np.percentile(Q, low_flow_percentile)
    z_active = reach.compute_stage_from_discharge(Q_low)

    return np.linspace(z_active, zmax, n_points)



# CAMPOREALE DETERMINISTIC ANALOGUE

def daily_biomass_model(reach, Q, elevations, params, B0=0.05):

    biomass = {}

    for z in elevations:

        Qcrit = reach.discharge_from_stage(z)

        B = np.zeros(len(Q))
        B[0] = B0

        for t in range(len(Q) - 1):

            # growth
            if Q[t] < Qcrit:
                growth = params.ag * (Qcrit - Q[t]) ** params.mg
                decay = 0

            # flood decay
            else:
                growth = 0
                decay = params.ad * (Q[t] - Qcrit) ** params.md

            B[t + 1] = max(0, B[t] + growth - decay)

        biomass[z] = B

    return biomass


# SERLET BIOMASS PROXY

def serlet_biomass_proxy(ebe2_windows, elevations):

    biomass = {}

    for z in elevations:

        B = np.zeros(sum([w[1] - w[0] for w in ebe2_windows]))

        for w in ebe2_windows:
            B[w[0]:w[1]] += 1.0

        biomass[z] = B

    return biomass



# ZEN FEEDBACK


def update_bar_elevation(z, biomass,
                         alpha_dep=0.002,
                         beta_ero=0.0015):

    dz = alpha_dep * biomass - beta_ero * (1 - biomass)
    return z + dz


def simulate_bar_evolution(reach, Q, elevations, biomass_dict):

    results = {}

    for z in elevations:

        B = biomass_dict[z]

        Z = np.zeros(len(B))
        Z[0] = z

        for t in range(len(B) - 1):
            Z[t + 1] = update_bar_elevation(Z[t], B[t])

        results[z] = Z

    return results



# MASTER MODEL

def run_vegetation_model(reach, Q, dates, params):

    stage = reach.compute_stage_from_discharge(Q)
    cv = compute_cv(Q)

    ebe1 = compute_EBE1(stage, cv)

    if ebe1 is None:
        return {"recruitment": None}

    Q_germ, Q_mort = seasonal_thresholds(Q, dates, params)

    ebe2 = compute_EBE2(Q, Q_germ)
    ebe3 = compute_EBE3(Q, Q_mort)

    elevations = elevation_grid(reach, Q,
                                params.low_flow_percentile)

    # MODEL 1 — deterministic
    biomass_det = daily_biomass_model(
        reach, Q, elevations, params
    )

    # MODEL 2 — Serlet proxy
    biomass_serlet = serlet_biomass_proxy(
        ebe2, elevations
    )

    bar_det = simulate_bar_evolution(
        reach, Q, elevations, biomass_det
    )

    bar_serlet = simulate_bar_evolution(
        reach, Q, elevations, biomass_serlet
    )

    return {
        "EBE1": ebe1,
        "EBE2": ebe2,
        "EBE3": ebe3,
        "biomass_deterministic": biomass_det,
        "biomass_serlet": biomass_serlet,
        "bar_det": bar_det,
        "bar_serlet": bar_serlet,
        "cv": cv,
    }