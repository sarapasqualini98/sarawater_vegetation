import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sarawater.reach import Reach
from sarawater.vegetation import (
    HydraulicVegetationForcing,
    VegetationConfig,
    age_dependent_root_depth,
    caponi_resistance,
    initialize_vegetation,
    run_reach_vegetation,
    update_mechanical_damage,
)


def _hourly_reach(hours: int = 24 * 12, cells: int = 41) -> Reach:
    dates = pd.date_range("2023-04-01", periods=hours, freq="h")
    t = np.arange(hours, dtype=float)
    q = 18.0 + 4.0 * np.sin(2.0 * np.pi * t / (24.0 * 5.0))
    q += 25.0 * np.exp(-0.5 * ((t - 24.0 * 4.0) / 10.0) ** 2)
    reach = Reach("Hourly vegetation reach", dates.to_pydatetime().tolist(), q, 5.0)
    y = np.linspace(0.0, 40.0, cells)
    z = 2.0 + 0.004 * (y - 20.0) ** 2
    reach.add_cross_section_geometry(
        slope=0.002,
        ks=30.0,
        section=pd.DataFrame({"y [m]": y, "z [m]": z}),
    )
    reach.add_grain_size_distribution(30.0)
    return reach


def _dummy_forcing(z: np.ndarray) -> HydraulicVegetationForcing:
    n_t = 12
    water = np.linspace(float(np.min(z)) - 0.2, float(np.median(z)), n_t)
    zeros_t = np.zeros(n_t)
    zeros_tx = np.zeros((n_t, z.size))
    return HydraulicVegetationForcing(
        discharge=np.ones(n_t),
        water_level=water,
        velocity=zeros_t,
        wetted_area=zeros_t,
        wetted_perimeter=zeros_t,
        depth=zeros_tx,
        shear_stress=zeros_tx,
        shields_d50=zeros_tx,
        d50_m=0.03,
        groundwater_stage=water.copy(),
        recession_rate_cm_day=np.zeros(n_t, dtype=float),
    )


def test_defaults_match_latest_recruitment_and_mortality_alignment():
    """The public defaults encode the revised scientific design."""
    cfg = VegetationConfig()
    assert cfg.woo_days == 80.0
    assert cfg.recruitment_box_lower_offset_m == 0.6
    assert cfg.recruitment_box_upper_offset_m == 2.0
    assert cfg.resistance_initial_hours == 1.0
    assert cfg.resistance_max_hours == 60.0 * 24.0
    assert cfg.resistance_growth_rate == 0.001
    assert cfg.shear_seedling_critical_pa == 20.0
    assert cfg.shear_adult_critical_pa == 80.0
    assert cfg.shields_critical == 0.045
    assert cfg.shields_damage_rate_multiplier == 0.20
    assert cfg.drought_tolerance_hours_seedling == 48.0
    assert cfg.drought_tolerance_hours_adult == 14.0 * 24.0
    assert cfg.damage_recovery_half_after_hours == 7.0 * 24.0
    assert cfg.damage_recovery_full_after_hours == 30.0 * 24.0
    assert cfg.root_initial_depth_m == 0.05


def test_reach_runs_both_vegetation_modes():
    """Both models preserve the 41 cells explicitly supplied by this test."""
    reach = _hourly_reach(cells=41)
    cfg = VegetationConfig(
        initial_condition="young",
        woo_days=3.0,
        rating_curve_points=16,
        enable_shear_mortality=False,
        enable_shields_mortality=False,
        enable_drought_mortality=False,
    )
    recruitment = run_reach_vegetation(reach, "recruitment", cfg)
    camporeale = run_reach_vegetation(reach, "camporeale", cfg)
    assert recruitment.state.shape == (len(reach.dates), 41)
    assert camporeale.state.shape == recruitment.state.shape
    assert recruitment.presence.dtype == bool
    assert np.all((camporeale.state >= 0.0) & (camporeale.state <= 1.0))


def test_all_adult_traits_share_the_same_effective_age():
    """Optional root/mechanical traits reach adult values on one shared clock."""
    cfg = VegetationConfig()
    age_h = cfg.adult_resistance_age_days * 24.0
    root = age_dependent_root_depth(
        np.array([age_h]),
        initial_depth_m=cfg.root_initial_depth_m,
        adult_depth_m=cfg.root_adult_depth_m,
        adult_age_days=cfg.adult_resistance_age_days,
    )[0]
    _, _, fields = update_mechanical_damage(
        np.zeros(1),
        np.zeros(1),
        tau=np.zeros(1),
        theta=np.zeros(1),
        submerged_alive=np.array([False]),
        emerged_alive=np.array([True]),
        effective_age_hours=np.array([age_h]),
        dt_hours=1.0,
        shear_seedling_critical_pa=cfg.shear_seedling_critical_pa,
        shear_adult_critical_pa=cfg.shear_adult_critical_pa,
        shields_bed_critical=cfg.shields_critical,
        shields_adult_resistance_multiplier=cfg.shields_adult_multiplier,
        adult_age_days=cfg.adult_resistance_age_days,
        reference_hours_seedling=cfg.dose_reference_hours_seedling,
        reference_hours_adult=cfg.dose_reference_hours_adult,
        damage_exponent=cfg.dose_damage_exponent,
    )
    assert np.isclose(root, cfg.root_adult_depth_m)
    assert np.isclose(fields["tau_critical"][0], cfg.shear_adult_critical_pa)
    assert np.isclose(fields["shields_resistance_factor"][0], cfg.shields_adult_multiplier)
    assert np.isclose(fields["dose_reference_hours"][0], cfg.dose_reference_hours_adult)


def test_caponi_defaults_reproduce_about_seven_hours_after_85_days():
    """The exact Caponi defaults give ~7.7 h resistance at day 85."""
    cfg = VegetationConfig()
    resistance = caponi_resistance(np.array([85.0 * 24.0]), cfg)[0]
    assert 7.0 < resistance < 8.5
    # The logistic tends asymptotically to Rmax and is not forced by the
    # separate 15-year trait horizon.
    later = caponi_resistance(np.array([cfg.adult_resistance_age_days * 24.0]), cfg)[0]
    assert later <= cfg.resistance_max_hours
    assert later > resistance
