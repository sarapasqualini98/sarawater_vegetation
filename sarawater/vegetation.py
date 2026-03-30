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

from sarawater.hydraulics import (
    compute_stage_timeseries,
    compute_recession_rate
)  
 def __init__(self):

        # Hydrological thresholds
        self.low_flow_percentile = 20
        self.flood_percentile = 90

        # Logistic growth parameters
        self.ag = 0.04        # growth rate [day^-1]
        self.ad = 0.08        # decay rate
        self.Bmax = 1.0       # carrying capacity

        # Recruitment box
        self.Rcrit = 0.02     # max recession rate
        self.growing_months = (4,5,6,7)

        # Zen feedback
        self.alpha_serlet = 0.01
        self.beta_biomass = 0.02



# HYDROLOGICAL METRICS

def compute_cv(Q):
    """
    Coefficient of variation of discharge.
    Proxy for hydrological disturbance regime.
    """
    Q = np.asarray(Q)
    mu_Q = np.mean(Q)
    sigma_Q = np.std(Q)
    return sigma_Q / mu_Q


#Serlet 2023

def recruitment_box(stage, elevation, dates, params):
    """
    Recruitment probability following:
    Mahoney & Rood (1998) + Serlet ecohydraulic windows.
    """

    dates = pd.to_datetime(dates)
    recession = compute_recession_rate(stage)

    prob = np.zeros(len(elevation))

    for t in range(1, len(stage)):

        # growing season constraint
        if dates[t].month not in params.growing_months:
            continue

        # exposed surfaces
        exposed = elevation > stage[t]

        # recession constraint
        if recession[t] < params.Rcrit:
            prob += exposed.astype(float)

    return prob / len(stage)



# DETERMINISTIC BIOMASS MODEL
# (Camporeale 2006 inspired) 

def daily_biomass_model(Q, stage, elevation, params):

    Qcrit = np.percentile(Q, params.flood_percentile)

    B = np.zeros(len(elevation))
    biomass_history = []

    for t in range(len(Q)):

        # logistic growth
        growth = params.ag * B * (1 - B / params.Bmax)

        # flood mortality
        flood = (Q[t] > Qcrit).astype(float)
        decay = params.ad * flood * B

        dB = growth - decay
        B = B + dB

        B = np.clip(B, 0, params.Bmax)

        biomass_history.append(B.copy())

    return np.array(biomass_history)



# ZEN MORPHODYNAMIC FEEDBACK


def zen_from_recruitment(z, recruitment_prob, params):
    """
    Morphodynamic proxy driven by recruitment success.
    """
    return z + params.alpha_serlet * recruitment_prob


def zen_from_biomass(z, biomass_history, params):
    """
    Morphodynamic proxy driven by biomass stabilization.
    """
    Bmean = np.mean(biomass_history, axis=0)
    return z + params.beta_biomass * Bmean



#  VEGETATION MODEL

class VegetationModel:

    def __init__(self, Q, reach, dates, params=None):

        self.Q = np.asarray(Q)
        self.reach = reach
        self.dates = pd.to_datetime(dates)

        self.params = params or VegetationParameters()

    def run(self):

        # geometry
        elevation = self.reach.get_cross_section_elevation()

        # hydraulics → stage
        stage = compute_stage_timeseries(self.Q, self.reach)

        # hydrological variability
        cv = compute_cv(self.Q)

        # recruitment (Serlet)
        recruitment_prob = recruitment_box(
            stage,
            elevation,
            self.dates,
            self.params
        )

        # deterministic biomass
        biomass = daily_biomass_model(
            self.Q,
            stage,
            elevation,
            self.params
        )

        # Zen feedbacks
        z_serlet = zen_from_recruitment(
            elevation,
            recruitment_prob,
            self.params
        )

        z_biomass = zen_from_biomass(
            elevation,
            biomass,
            self.params
        )

        return {
            "stage": stage,
            "cv": cv,
            "recruitment_probability": recruitment_prob,
            "biomass": biomass,
            "zen_serlet": z_serlet,
            "zen_biomass": z_biomass,
            "elevation": elevation
        }
