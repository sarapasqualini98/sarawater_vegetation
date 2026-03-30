import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

from sarawater.vegetation import (
    compute_cv,
    recruitment_box,
    daily_biomass_model,
    zen_from_recruitment,
    zen_from_biomass,
    VegetationParameters,
)


# MOCK REACH OBJECT (minimal geometry)

class MockReach:
    def __init__(self):
        self.z = np.linspace(0, 2, 20)

    def get_cross_section_elevation(self):
        return self.z


# TEST DATA

np.random.seed(0)

Q_constant = np.ones(100) * 5.0
Q_variable = np.linspace(1, 10, 100)

stage_low = np.ones(100) * 0.5
stage_high = np.ones(100) * 2.5

dates = pd.date_range("2000-04-01", periods=100)

reach = MockReach()
elevation = reach.get_cross_section_elevation()

params = VegetationParameters()



# TESTS

def test_compute_cv_constant_flow():
    """CV should be zero for constant discharge."""
    cv = compute_cv(Q_constant)
    assert np.isclose(cv, 0.0), f"Expected CV=0, got {cv}"


def test_compute_cv_variability():
    """CV should increase with variability."""
    cv = compute_cv(Q_variable)
    assert cv > 0, "CV should be positive for variable flow"




def test_recruitment_only_when_exposed():
    """Recruitment occurs only where bed is exposed."""
    prob = recruitment_box(stage_low, elevation, dates, params)

    submerged = elevation < stage_low[0]

    assert np.all(prob[submerged] == 0), \
        "Recruitment should be zero underwater"


def test_recruitment_recession_constraint():
    """Fast rising water should prevent recruitment."""
    stage_rising = np.linspace(0.5, 1.5, 100)

    prob = recruitment_box(stage_rising, elevation, dates, params)

    assert np.all(prob < 0.2), \
        "Recruitment should be low during rising stage"




def test_biomass_logistic_growth_limit():
    """Biomass should not exceed carrying capacity."""
    biomass = daily_biomass_model(
        Q_constant,
        stage_low,
        elevation,
        params
    )

    assert np.max(biomass) <= params.Bmax + 1e-6, \
        "Biomass exceeded carrying capacity"


def test_biomass_positive():
    """Biomass must remain non-negative."""
    biomass = daily_biomass_model(
        Q_constant,
        stage_low,
        elevation,
        params
    )

    assert np.min(biomass) >= 0, "Biomass became negative"


def test_flood_mortality_effect():
    """High floods should reduce biomass."""
    Q_flood = np.ones(100) * 100.0

    biomass = daily_biomass_model(
        Q_flood,
        stage_low,
        elevation,
        params
    )

    mean_final = np.mean(biomass[-1])
    assert mean_final < 0.2, \
        "Flood mortality not reducing biomass"




def test_zen_recruitment_feedback():
    """Zen evolution should increase elevation where recruitment high."""
    recruitment = np.linspace(0, 1, len(elevation))

    z_new = zen_from_recruitment(elevation, recruitment, params)

    assert np.all(z_new >= elevation), \
        "Elevation should not decrease under positive recruitment"


def test_zen_biomass_feedback():
    """Higher biomass should increase elevation."""
    biomass = np.tile(np.linspace(0, 1, len(elevation)), (50, 1))

    z_new = zen_from_biomass(elevation, biomass, params)

    assert np.mean(z_new) > np.mean(elevation), \
        "Biomass feedback should increase elevation"




def test_zero_flow_stability():
    """Model should remain stable for zero discharge."""
    Q_zero = np.zeros(100)

    biomass = daily_biomass_model(
        Q_zero,
        stage_low,
        elevation,
        params
    )

    assert np.all(np.isfinite(biomass)), \
        "Model unstable for zero flow"


def test_model_determinism():
    """Same input must produce identical results."""
    b1 = daily_biomass_model(Q_constant, stage_low, elevation, params)
    b2 = daily_biomass_model(Q_constant, stage_low, elevation, params)

    assert np.allclose(b1, b2), \
        "Model is not deterministic"


# RUN ALL TESTS


if __name__ == "__main__":

    test_compute_cv_constant_flow()
    test_compute_cv_variability()
    test_recruitment_only_when_exposed()
    test_recruitment_recession_constraint()
    test_biomass_logistic_growth_limit()
    test_biomass_positive()
    test_flood_mortality_effect()
    test_zen_recruitment_feedback()
    test_zen_biomass_feedback()
    test_zero_flow_stability()
    test_model_determinism()

    print("All vegetation tests passed successfully.")