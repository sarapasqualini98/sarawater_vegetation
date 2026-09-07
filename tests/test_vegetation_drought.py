import numpy as np
import pandas as pd

from sarawater.vegetation import (
    VegetationConfig,
    _apply_scheduled_recovery,
    _dynamic_groundwater_stage,
    update_drought_damage,
)


def _call_damage(
    *,
    damage=0.0,
    clock=0.0,
    bed=2.0,
    stage=1.0,
    root_age_h=0.0,
    dt=1.0,
    recession=0.0,
    tolerance_seedling=48.0,
    tolerance_adult=14.0 * 24.0,
):
    cfg = VegetationConfig(
        drought_tolerance_hours_seedling=tolerance_seedling,
        drought_tolerance_hours_adult=tolerance_adult,
    )
    return update_drought_damage(
        np.array([damage], dtype=float),
        np.array([clock], dtype=float),
        bed_elevation=np.array([bed]),
        groundwater_stage=stage,
        alive=np.array([True]),
        alive_emerged=np.array([True]),
        recession_rate_cm_day=recession,
        effective_age_hours=np.array([root_age_h]),
        dt_hours=dt,
        initial_root_depth_m=cfg.root_initial_depth_m,
        adult_root_depth_m=cfg.root_adult_depth_m,
        adult_age_days=cfg.adult_resistance_age_days,
        reference_hours_seedling=cfg.dose_reference_hours_seedling,
        reference_hours_adult=cfg.dose_reference_hours_adult,
        damage_exponent=cfg.dose_damage_exponent,
        tolerance_hours_seedling=tolerance_seedling,
        tolerance_hours_adult=tolerance_adult,
        enabled=True,
    )


def test_groundwater_proxy_is_current_river_level_at_every_timestep():
    dates = pd.date_range("2023-01-01", periods=4, freq="h")
    stage = np.array([1.0, 2.0, 0.5, 1.2])
    assert np.array_equal(_dynamic_groundwater_stage(stage, dates), stage)


def test_drought_requires_root_disconnection_only():
    damage, clock, fields = _call_damage(bed=2.0, stage=1.96)
    assert damage[0] == 0.0
    assert clock[0] == 0.0
    assert fields["water_accessible"][0]

    damage, clock, fields = _call_damage(
        bed=2.0, stage=0.0, tolerance_seedling=0.0, tolerance_adult=0.0
    )
    assert fields["drought_exposed"][0]
    assert clock[0] == 1.0
    assert damage[0] > 0.0


def test_drought_damage_starts_after_age_dependent_tolerance():
    cfg = VegetationConfig()
    damage = np.array([0.0])
    clock = np.array([0.0])
    for _ in range(48):
        damage, clock, fields = update_drought_damage(
            damage, clock,
            bed_elevation=np.array([2.0]), groundwater_stage=0.0,
            alive=np.array([True]), alive_emerged=np.array([True]),
            recession_rate_cm_day=0.0, effective_age_hours=np.array([0.0]),
            dt_hours=1.0, initial_root_depth_m=cfg.root_initial_depth_m,
            adult_root_depth_m=cfg.root_adult_depth_m,
            adult_age_days=cfg.adult_resistance_age_days,
            reference_hours_seedling=cfg.dose_reference_hours_seedling,
            reference_hours_adult=cfg.dose_reference_hours_adult,
            damage_exponent=cfg.dose_damage_exponent,
            tolerance_hours_seedling=cfg.drought_tolerance_hours_seedling,
            tolerance_hours_adult=cfg.drought_tolerance_hours_adult,
        )
    assert clock[0] == 48.0
    assert damage[0] == 0.0
    damage, clock, _ = update_drought_damage(
        damage, clock,
        bed_elevation=np.array([2.0]), groundwater_stage=0.0,
        alive=np.array([True]), alive_emerged=np.array([True]),
        recession_rate_cm_day=0.0, effective_age_hours=np.array([0.0]),
        dt_hours=1.0, initial_root_depth_m=cfg.root_initial_depth_m,
        adult_root_depth_m=cfg.root_adult_depth_m,
        adult_age_days=cfg.adult_resistance_age_days,
        reference_hours_seedling=cfg.dose_reference_hours_seedling,
        reference_hours_adult=cfg.dose_reference_hours_adult,
        damage_exponent=cfg.dose_damage_exponent,
        tolerance_hours_seedling=cfg.drought_tolerance_hours_seedling,
        tolerance_hours_adult=cfg.drought_tolerance_hours_adult,
    )
    assert clock[0] == 49.0
    assert damage[0] > 0.0

    adult_age = cfg.adult_resistance_age_days * 24.0
    _, _, adult_fields = _call_damage(root_age_h=adult_age, dt=1.0)
    assert np.isclose(adult_fields["drought_tolerance_hours"][0], 14.0 * 24.0)


def test_deeper_older_roots_prevent_stress_at_same_water_level():
    cfg = VegetationConfig()
    young_damage, young_clock, young = _call_damage(
        bed=2.0, stage=1.0, root_age_h=0.0,
        tolerance_seedling=0.0, tolerance_adult=0.0,
    )
    adult_age = cfg.adult_resistance_age_days * 24.0
    adult_damage, adult_clock, adult = _call_damage(
        bed=2.0, stage=1.0, root_age_h=adult_age,
        tolerance_seedling=0.0, tolerance_adult=0.0,
    )
    assert young["drought_exposed"][0]
    assert young_damage[0] > 0.0
    assert not adult["drought_exposed"][0]
    assert adult_damage[0] == 0.0
    assert adult_clock[0] == 0.0


def test_damage_recovery_is_linear_to_half_after_seven_days_and_zero_after_thirty():
    damage = np.array([1.0])
    clock = np.array([0.0])
    reference = np.array([0.0])
    alive = np.array([True])
    stress = np.array([False])
    damage, clock, reference = _apply_scheduled_recovery(
        damage, clock, reference, alive=alive, stress_active=stress,
        dt_hours=3.5 * 24.0, half_after_hours=7.0 * 24.0,
        full_after_hours=30.0 * 24.0,
    )
    assert np.isclose(damage[0], 0.75)
    damage, clock, reference = _apply_scheduled_recovery(
        damage, clock, reference, alive=alive, stress_active=stress,
        dt_hours=3.5 * 24.0, half_after_hours=7.0 * 24.0,
        full_after_hours=30.0 * 24.0,
    )
    assert np.isclose(damage[0], 0.5)
    damage, clock, reference = _apply_scheduled_recovery(
        damage, clock, reference, alive=alive, stress_active=stress,
        dt_hours=23.0 * 24.0, half_after_hours=7.0 * 24.0,
        full_after_hours=30.0 * 24.0,
    )
    assert damage[0] == 0.0
