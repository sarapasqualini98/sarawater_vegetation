import numpy as np
import pandas as pd

from sarawater.vegetation import (
    AnnualRecruitmentBand,
    VegetationConfig,
    _recruitment_step,
    compute_serlet_bands,
)


def _state_one_candidate(*, elapsed_hours: float, effective_hours: float = 0.0):
    return {
        "presence": np.array([True]),
        "established": np.array([False]),
        "chronological_age_hours": np.array([elapsed_hours]),
        "effective_age_hours": np.array([effective_hours]),
        "recruitment_year": np.array([2023]),
        "habitat_mask": np.array([True]),
        "woo_hours": np.array([elapsed_hours]),
        "anoxia_hours": np.zeros(1),
        "shear_damage": np.zeros(1),
        "sediment_damage": np.zeros(1),
        "drought_damage": np.zeros(1),
        "drought_disconnection_hours": np.zeros(1),
        "shear_recovery_hours": np.zeros(1),
        "shear_recovery_reference": np.zeros(1),
        "sediment_recovery_hours": np.zeros(1),
        "sediment_recovery_reference": np.zeros(1),
        "drought_recovery_hours": np.zeros(1),
        "drought_recovery_reference": np.zeros(1),
        "resistance_hours": np.ones(1),
    }


def _band(year=2023):
    return AnnualRecruitmentBand(
        year=year,
        initial_lower_m=0.0,
        initial_upper_m=3.0,
        woo_stage_m=0.0,
        ebe1_lower_m=0.0,
        ebe1_upper_m=3.0,
        recession_mortality_coefficient=0.0,
        recession_class="favourable",
        ebe1_valid=True,
        woo_start=pd.Timestamp(f"{year}-04-01"),
        woo_end=pd.Timestamp(f"{year}-06-20"),
    )


def test_native_hourly_window_detects_a_one_hour_disturbance():
    # Exactly one 80-day window; a one-hour pulse must remain visible.
    dates = pd.date_range("2023-04-01", periods=80 * 24 + 1, freq="h")
    stage = np.ones(len(dates))
    stage[40 * 24] = 3.0
    band = compute_serlet_bands(
        dates,
        stage,
        woo_days=80,
    )[2023]
    assert np.isclose(band.woo_stage_m, 3.0)
    assert band.ebe1_lower_m >= 3.0


def test_candidate_establishment_uses_chronological_time_across_season_boundary():
    cfg = VegetationConfig(
        woo_days=80.0,
        enable_shear_mortality=False,
        enable_shields_mortality=False,
        enable_drought_mortality=False,
    )
    state = _state_one_candidate(elapsed_hours=80.0 * 24.0 - 1.0)
    effective_before = state["effective_age_hours"].copy()
    state, diagnostics, cause = _recruitment_step(
        state,
        z=np.array([2.0]),
        water_level=1.0,
        tau=np.zeros(1),
        theta=np.zeros(1),
        date=pd.Timestamp("2023-10-01 00:00"),
        dt_hours=1.0,
        config=cfg,
        annual_band=None,
        groundwater_stage=1.0,
        recession_rate_cm_day=0.0,
        previous_water_level=1.0,
    )
    assert state["established"][0]
    assert diagnostics["recruits"] == 1
    assert cause[0] == 0
    # Biological time advanced, trait age did not because Oct 1 is outside GS.
    assert np.isclose(state["chronological_age_hours"][0], 80.0 * 24.0)
    assert np.array_equal(state["effective_age_hours"], effective_before)


def test_nonlethal_submergence_does_not_reset_candidate_clock():
    cfg = VegetationConfig(
        woo_days=80.0,
        enable_shear_mortality=False,
        enable_shields_mortality=False,
        enable_drought_mortality=False,
    )
    # Effective age of 85 days gives ~7.7 h inundation resistance.
    state = _state_one_candidate(elapsed_hours=60 * 24.0, effective_hours=85 * 24.0)
    before = state["woo_hours"][0]
    state, diagnostics, cause = _recruitment_step(
        state,
        z=np.array([1.0]),
        water_level=1.1,
        tau=np.zeros(1),
        theta=np.zeros(1),
        date=pd.Timestamp("2023-07-01 12:00"),
        dt_hours=1.0,
        config=cfg,
        annual_band=_band(),
        groundwater_stage=1.1,
        recession_rate_cm_day=0.0,
        previous_water_level=1.1,
    )
    assert state["presence"][0]
    assert not state["established"][0]
    assert np.isclose(state["woo_hours"][0], before + 1.0)
    assert diagnostics["mortality"] == 0
    assert cause[0] == 0


def test_new_candidates_are_created_only_in_season_habitat_band_and_falling_stage():
    cfg = VegetationConfig(
        woo_days=80.0,
        enable_shear_mortality=False,
        enable_shields_mortality=False,
        enable_drought_mortality=False,
    )
    n = 3
    state = {
        "presence": np.zeros(n, dtype=bool),
        "established": np.zeros(n, dtype=bool),
        "chronological_age_hours": np.zeros(n),
        "effective_age_hours": np.zeros(n),
        "recruitment_year": np.full(n, -1),
        "habitat_mask": np.array([True, False, True]),
        "woo_hours": np.zeros(n),
        "anoxia_hours": np.zeros(n),
        "shear_damage": np.zeros(n),
        "sediment_damage": np.zeros(n),
        "drought_damage": np.zeros(n),
        "drought_disconnection_hours": np.zeros(n),
        "shear_recovery_hours": np.zeros(n),
        "shear_recovery_reference": np.zeros(n),
        "sediment_recovery_hours": np.zeros(n),
        "sediment_recovery_reference": np.zeros(n),
        "drought_recovery_hours": np.zeros(n),
        "drought_recovery_reference": np.zeros(n),
        "resistance_hours": np.ones(n),
    }
    state, diagnostics, _ = _recruitment_step(
        state,
        z=np.array([1.0, 1.2, 3.5]),
        water_level=0.8,
        tau=np.zeros(n),
        theta=np.zeros(n),
        date=pd.Timestamp("2023-05-15 10:00"),
        dt_hours=1.0,
        config=cfg,
        annual_band=_band(),
        groundwater_stage=0.8,
        recession_rate_cm_day=0.0,
        previous_water_level=1.3,
    )
    assert np.array_equal(state["presence"], np.array([True, False, False]))
    assert diagnostics["seedlings_created"] == 1


def test_mortality_cause_is_exclusive_when_thresholds_coincide():
    cfg = VegetationConfig(
        woo_days=80.0,
        drought_tolerance_hours_seedling=0.0,
        drought_tolerance_hours_adult=0.0,
        drought_damage_rate_multiplier=1000.0,
    )
    state = _state_one_candidate(elapsed_hours=10.0, effective_hours=0.0)
    state["shear_damage"][:] = 1.0
    state["sediment_damage"][:] = 1.0
    state["drought_damage"][:] = 1.0
    state["anoxia_hours"][:] = 10.0
    state, diagnostics, cause = _recruitment_step(
        state,
        z=np.array([2.0]),
        water_level=1.0,
        tau=np.array([100.0]),
        theta=np.array([1.0]),
        date=pd.Timestamp("2023-07-01"),
        dt_hours=1.0,
        config=cfg,
        annual_band=_band(),
        groundwater_stage=0.0,
        recession_rate_cm_day=12.0,
        previous_water_level=1.0,
    )
    assert not state["presence"][0]
    assert cause[0] in {1, 2, 3, 4}
    assert sum(
        diagnostics[name]
        for name in (
            "mortality_anoxia",
            "mortality_shear",
            "mortality_sediment",
            "mortality_drought",
        )
    ) == 1