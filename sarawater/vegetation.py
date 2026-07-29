

"""
The module provides two complementary ecological representations:
``recruitment model``
    Binary plant presence with explicit candidate seedlings, chronological
    establishment time, growing-season resistance age and hourly mortality
    from prolonged inundation, direct shear, bed mobility and drought/root
    disconnection.

``Camporeale model``
    Continuous normalized biomass governed by hydrological carrying capacity,
    growth while exposed and flood-induced decay while submerged.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from sarawater.hydraulics import steady_flow_solver
from sarawater.sediment_load import DMI, PHI_RANGE, shields_parameter

HOURS_PER_DAY = 24.0


@dataclass(frozen=True)
class AnnualRecruitmentBand:
    """Annual Recruitment Box and EBE1 opportunity-band diagnostics.

    EBE1 is used only to delimit where a new candidate may be deposited.
    Subsequent survival is evaluated explicitly at the native input timestep.
    """

    year: int
    initial_lower_m: float
    initial_upper_m: float
    woo_stage_m: float
    ebe1_lower_m: float
    ebe1_upper_m: float
    recession_mortality_coefficient: float
    recession_class: str
    ebe1_valid: bool
    woo_start: pd.Timestamp | None = None
    woo_end: pd.Timestamp | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def in_season(date: pd.Timestamp, start: tuple[int, int], end: tuple[int, int]) -> bool:
    value = (int(date.month), int(date.day))
    return start <= value <= end


def shared_ontogenetic_fraction(
    effective_age_hours: np.ndarray | float,
    adult_age_days: float,
) -> np.ndarray:
    age_days = np.maximum(np.asarray(effective_age_hours, dtype=float), 0.0) / HOURS_PER_DAY
    return np.clip(age_days / max(float(adult_age_days), 1e-12), 0.0, 1.0)


def _caponi_resistance(
    effective_age_hours: np.ndarray | float,
    initial_hours: float,
    maximum_hours: float,
    growth_rate_per_hour: float,
) -> np.ndarray:
    """
    Caponi et al. logistic resistance to consecutive inundation [h], which 
    grows only with effective growing-season age (different than biological age).
    """
    age = np.maximum(np.asarray(effective_age_hours, dtype=float), 0.0)
    r0 = max(float(initial_hours), 1e-12)
    rmax = max(float(maximum_hours), r0)
    sigma = max(float(growth_rate_per_hour), 0.0)
    return rmax / (1.0 + (rmax / r0 - 1.0) * np.exp(-sigma * age))


def effective_age_from_chronological(
    chronological_age_hours: np.ndarray | float,
    growing_days_per_year: float = 183.0,
) -> np.ndarray:
    """
    Convert an initial chronological template to approximate growth age.
    This is only used to initialize pre-existing ``young`` and ``mature``
    vegetation.  
    """
    chronological = np.maximum(np.asarray(chronological_age_hours, float), 0.0)
    return chronological * np.clip(float(growing_days_per_year) / 365.25, 0.0, 1.0)


def _stage_series(dates: pd.DatetimeIndex, stage: np.ndarray) -> pd.Series:
    """Return the native-resolution water-level series without aggregation."""
    return pd.Series(np.asarray(stage, float), index=dates)


def _rolling_recession_rate_cm_day(
    water_level: np.ndarray | pd.Series,
    dates: Iterable | None = None,
) -> np.ndarray:
    """
    Return the preceding-72-hour mean positive stage decline [cm/day].
    Rising or constant stages have zero recession. 
    """
    if isinstance(water_level, pd.Series):
        stage = water_level.sort_index().astype(float)
    else:
        if dates is None:
            raise ValueError("dates are required when water_level is an array")
        stage = pd.Series(
            np.asarray(water_level, dtype=float),
            index=pd.DatetimeIndex(pd.to_datetime(list(dates))),
        ).sort_index()
    if stage.empty:
        return np.zeros(0, dtype=float)
    dt_h = stage.index.to_series().diff().dt.total_seconds().div(3600.0)
    rate = (-stage.diff()) / dt_h * HOURS_PER_DAY * 100.0
    rate = rate.clip(lower=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rolling = rate.rolling("72h", min_periods=1).mean().fillna(0.0)
    return rolling.to_numpy(dtype=float)


def recession_mortality_coefficient(stage: pd.Series) -> tuple[float, str, float, float]:
    """
    Serlet/Burke 72-hour recession diagnostic.
    The positive stage-decline rate is expressed in cm/day and averaged over
    the preceding 72 hours. Rates of 0--5 cm/day are favourable, 5--10 cm/day
    stressful and >10 cm/day lethal. The annual/window coefficient is
    ``M=(3*%lethal + %stressful)/3``. M remains a recruitment diagnostic; the
    dynamic drought routine uses the same classes only to modulate damage when
    roots are already disconnected from the instantaneous groundwater proxy.
    """
    stage = stage.sort_index().dropna()
    if stage.size < 2:
        return 0.0, "favourable", 0.0, 0.0
    rolling = _rolling_recession_rate_cm_day(stage)
    stressful = float(np.mean((rolling > 5.0) & (rolling <= 10.0)) * 100.0)
    lethal = float(np.mean(rolling > 10.0) * 100.0)
    mortality = (3.0 * lethal + stressful) / 3.0
    label = "favourable" if mortality < 20.0 else ("marginal" if mortality <= 30.0 else "unfavourable")
    return float(mortality), label, stressful, lethal


def _best_woo_window(
    stage: pd.Series, woo_days: float
) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None, pd.Series]:
    """
    Select the continuous chronological window with the lowest peak stage.
    The calculation uses the native time resolution (normally hourly). A full
    ``woo_days`` window is required; if the available season is shorter, the
    whole available period is returned and the band is marked invalid later.
    """
    stage = stage.sort_index().dropna()
    if stage.empty:
        return np.nan, None, None, stage
    duration = pd.Timedelta(days=float(woo_days))
    if stage.index[-1] - stage.index[0] < duration:
        return float(stage.max()), stage.index[0], stage.index[-1], stage
    rolling_max = stage.rolling(duration, closed="both", min_periods=1).max()
    valid = rolling_max[rolling_max.index >= stage.index[0] + duration]
    if valid.empty:
        return float(stage.max()), stage.index[0], stage.index[-1], stage
    end = pd.Timestamp(valid.idxmin())
    start = end - duration
    selected = stage.loc[(stage.index >= start) & (stage.index <= end)]
    return float(valid.loc[end]), start, end, selected


def compute_serlet_bands(
    dates: Iterable,
    water_level: np.ndarray,
    *,
    season_start: tuple[int, int] = (4, 1),
    season_end: tuple[int, int] = (9, 30),
    woo_days: float = 80.0,
    lower_offset_m: float = 0.6,
    upper_offset_m: float = 2.0,
    section_min_m: float | None = None,
    section_max_m: float | None = None,
) -> dict[int, AnnualRecruitmentBand]:
    """
    Compute the Mahoney--Rood Recruitment Box followed by EBE1.
    The default mode uses the original Recruitment Box geometry: a candidate
    elevation band 0.6--2.0 m above the late-summer/base stage. In this generic
    cross-section implementation the minimum stage within the configured
    recruitment season is used as the available late-summer base-stage proxy.
    The upper boundary is capped only by the user cross-section maximum.
    EBE1 is retained as an annual deposition/opportunity filter. Its lower
    boundary is raised to the maximum stage of the least-disturbed complete
    ``woo_days`` window. Subsequent plant survival is still evaluated hourly by
    anoxia, direct shear, Shields/bed mobility and root disconnection.
    """
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    h = np.asarray(water_level, float)
    if index.size != h.size or index.empty:
        raise ValueError("dates and water_level must be non-empty and aligned")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be strictly increasing")
    stage = _stage_series(index, h)

    bands: dict[int, AnnualRecruitmentBand] = {}
    for year in np.unique(index.year):
        start = pd.Timestamp(year=int(year), month=season_start[0], day=season_start[1])
        end = pd.Timestamp(year=int(year), month=season_end[0], day=season_end[1], hour=23, minute=59, second=59)
        growing = stage.loc[(stage.index >= start) & (stage.index <= end)]
        if growing.empty:
            continue

        base_stage = float(growing.min())
        initial_lower = base_stage + float(lower_offset_m)
        initial_upper = base_stage + float(upper_offset_m)
        if section_min_m is not None:
            initial_lower = max(initial_lower, float(section_min_m))
        if section_max_m is not None:
            initial_upper = min(initial_upper, float(section_max_m))

        woo_stage, woo_start, woo_end, woo_series = _best_woo_window(growing, woo_days)
        ebe1_lower = max(initial_lower, float(woo_stage))
        ebe1_upper = initial_upper
        M, recession_class, _, _ = recession_mortality_coefficient(woo_series)
        full_window = bool(
            woo_start is not None
            and woo_end is not None
            and (woo_end - woo_start) >= pd.Timedelta(days=float(woo_days))
        )
        valid = bool(full_window and np.isfinite(ebe1_lower) and ebe1_upper > ebe1_lower)
        bands[int(year)] = AnnualRecruitmentBand(
            year=int(year),
            initial_lower_m=float(initial_lower),
            initial_upper_m=float(initial_upper),
            woo_stage_m=float(woo_stage),
            ebe1_lower_m=float(ebe1_lower),
            ebe1_upper_m=float(ebe1_upper),
            recession_mortality_coefficient=float(M),
            recession_class=recession_class,
            ebe1_valid=valid,
            woo_start=woo_start,
            woo_end=woo_end,
        )
    return bands


def ontogenetic_value(
    effective_age_hours: np.ndarray,
    seedling_value: float,
    adult_value: float,
    *,
    adult_age_days: float,
) -> np.ndarray:
    """Interpolate a trait using the common effective-age clock."""
    fraction = shared_ontogenetic_fraction(effective_age_hours, adult_age_days)
    return float(seedling_value) + fraction * (
        float(adult_value) - float(seedling_value)
    )

def _apply_scheduled_recovery(
    damage: np.ndarray,
    stress_free_hours: np.ndarray,
    recovery_reference: np.ndarray,
    *,
    alive: np.ndarray,
    stress_active: np.ndarray,
    dt_hours: float,
    half_after_hours: float,
    full_after_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover cumulative damage linearly during a stress-free period.
    The damage starts decreasing as soon as the stress ceases. It declines
    linearly from 100% of the value present at the start of recovery to 50% at
    ``half_after_hours`` (7 days by default), then continues linearly from 50%
    to zero at ``full_after_hours`` (30 days by default). A renewed stress event
    resets the stress-free clock and the recovery reference.
    """
    damage = np.asarray(damage, dtype=float)
    clock = np.asarray(stress_free_hours, dtype=float)
    reference = np.asarray(recovery_reference, dtype=float)
    alive = np.asarray(alive, dtype=bool)
    active = alive & np.asarray(stress_active, dtype=bool)
    inactive = alive & ~active & (damage > 0.0)

    clock[active] = 0.0
    reference[active] = damage[active]

    starting = inactive & (clock <= 0.0)
    reference[starting] = damage[starting]
    clock[inactive] += float(dt_hours)

    half = max(float(half_after_hours), 0.0)
    full = max(float(full_after_hours), half + 1e-12)
    factor = np.ones_like(damage)
    if half > 0.0:
        before_or_at_half = inactive & (clock <= half)
        factor[before_or_at_half] = 1.0 - 0.5 * clock[before_or_at_half] / half
    else:
        before_or_at_half = inactive & (clock <= 0.0)
        factor[before_or_at_half] = 0.5
    after_half = inactive & (clock > half) & (clock < full)
    factor[after_half] = 0.5 * (full - clock[after_half]) / (full - half)
    completed = inactive & (clock >= full)
    factor[completed] = 0.0
    damage[inactive] = np.minimum(damage[inactive], reference[inactive] * factor[inactive])

    no_damage = alive & (damage <= 0.0)
    damage[no_damage] = 0.0
    clock[no_damage] = 0.0
    reference[no_damage] = 0.0
    damage[~alive] = 0.0
    clock[~alive] = 0.0
    reference[~alive] = 0.0
    return damage, clock, reference


def update_mechanical_damage(
    shear_damage: np.ndarray,
    shields_damage: np.ndarray,
    *,
    tau: np.ndarray,
    theta: np.ndarray,
    submerged_alive: np.ndarray,
    emerged_alive: np.ndarray,
    effective_age_hours: np.ndarray,
    dt_hours: float,
    shear_seedling_critical_pa: float,
    shear_adult_critical_pa: float,
    shields_bed_critical: float,
    shields_adult_resistance_multiplier: float,
    adult_age_days: float,
    reference_hours_seedling: float,
    reference_hours_adult: float,
    damage_exponent: float,
    enable_shear: bool = True,
    enable_shields: bool = True,
    shields_damage_rate_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    del emerged_alive
    fraction = shared_ontogenetic_fraction(effective_age_hours, adult_age_days)
    tau_crit = float(shear_seedling_critical_pa) + fraction * (
        float(shear_adult_critical_pa) - float(shear_seedling_critical_pa)
    )
    shields_resistance = 1.0 + fraction * (
        float(shields_adult_resistance_multiplier) - 1.0
    )
    reference_hours = float(reference_hours_seedling) + fraction * (
        float(reference_hours_adult) - float(reference_hours_seedling)
    )

    exponent = max(float(damage_exponent), 1.0)
    shear_excess = np.maximum(
        np.asarray(tau, float) / np.maximum(tau_crit, 1e-12) - 1.0, 0.0
    )
    bed_threshold = max(float(shields_bed_critical), 1e-12)
    shields_bed_excess = np.maximum(
        np.asarray(theta, float) / bed_threshold - 1.0, 0.0
    )
    shields_effective_excess = shields_bed_excess / np.maximum(
        shields_resistance, 1e-12
    )

    submerged_alive = np.asarray(submerged_alive, dtype=bool)
    if enable_shear:
        shear_damage[submerged_alive] += (
            shear_excess[submerged_alive] ** exponent
            * float(dt_hours)
            / np.maximum(reference_hours[submerged_alive], 1e-12)
        )
    if enable_shields:
        shields_damage[submerged_alive] += (
            float(shields_damage_rate_multiplier)
            * shields_effective_excess[submerged_alive] ** exponent
            * float(dt_hours)
            / np.maximum(reference_hours[submerged_alive], 1e-12)
        )

    return shear_damage, shields_damage, {
        "ontogenetic_fraction": fraction,
        "tau_critical": tau_crit,
        "theta_bed_critical": np.full_like(np.asarray(theta, float), bed_threshold),
        "shields_resistance_factor": shields_resistance,
        "shear_excess": shear_excess,
        "shields_bed_excess": shields_bed_excess,
        "shields_effective_excess": shields_effective_excess,
        "shields_excess": shields_effective_excess,
        "dose_reference_hours": reference_hours,
    }


def phreatic_surface(
    bed_elevation: np.ndarray,
    river_stage: float,
    reference_stage: float,
    coupling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Instantaneous river-linked phreatic surface and water-table depth.
    With the default coupling of one, the phreatic elevation equals current
    river stage, consistent with the no-delay assumption in the reduced
    Camporeale formulation. It is capped at the local ground surface.
    """
    bed = np.asarray(bed_elevation, float)
    coupled = float(reference_stage) + float(coupling) * (float(river_stage) - float(reference_stage))
    phreatic = np.minimum(bed, coupled)
    depth = np.maximum(bed - phreatic, 0.0)
    return phreatic, depth


def camporeale_dichotomous_step(
    biomass: np.ndarray,
    bed_elevation: np.ndarray,
    river_stage: float,
    reference_stage: float,
    dt_hours: float,
    *,
    growth_rate_per_day: float,
    flood_mortality_per_day: float,
    growth_exponent: float,
    capacity_exponent: float,
    mortality_exponent: float,
    depth_scale_m: float,
    optimum_depth_m: float,
    capacity_curvature: float,
    groundwater_coupling: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Camporeale-style mutually exclusive growth or flood decay.
    """
    old = np.clip(np.asarray(biomass, float), 0.0, 1.0)
    bed = np.asarray(bed_elevation, float)
    dt_days = max(float(dt_hours), 0.0) / HOURS_PER_DAY
    phreatic, groundwater_depth = phreatic_surface(
        bed, river_stage, reference_stage, groundwater_coupling
    )
    scale = max(float(depth_scale_m), 1e-12)
    dimensionless_depth = groundwater_depth / scale
    optimum = float(optimum_depth_m) / scale
    capacity = np.clip(
        1.0 - float(capacity_curvature) * (dimensionless_depth - optimum) ** 2,
        0.0,
        1.0,
    )
    submerged = float(river_stage) > bed
    exposed = ~submerged
    growth = np.zeros_like(old)
    decay = np.zeros_like(old)
    growth[exposed] = (
        float(growth_rate_per_day)
        * np.maximum(old[exposed], 1e-12) ** float(growth_exponent)
        * np.maximum(capacity[exposed] - old[exposed], 0.0) ** float(capacity_exponent)
    )
    flood_depth = np.maximum(float(river_stage) - bed, 0.0) / scale
    decay[submerged] = (
        float(flood_mortality_per_day)
        * flood_depth[submerged]
        * np.maximum(old[submerged], 1e-12) ** float(mortality_exponent)
    )
    new = np.clip(old + dt_days * (growth - decay), 0.0, 1.0)
    return new, {
        "growth": growth,
        "decay": decay,
        "capacity": capacity,
        "phreatic_surface": phreatic,
        "groundwater_depth": groundwater_depth,
        "flood_depth": flood_depth,
        "submerged": submerged,
        "exposed": exposed,
    }

def age_dependent_root_depth(
    effective_age_hours: np.ndarray,
    *,
    initial_depth_m: float,
    adult_depth_m: float,
    adult_age_days: float,
) -> np.ndarray:
    """Root access using the same effective-age clock as all mortality traits."""
    fraction = shared_ontogenetic_fraction(effective_age_hours, adult_age_days)
    depth = float(initial_depth_m) + fraction * (
        float(adult_depth_m) - float(initial_depth_m)
    )
    return np.clip(depth, 0.0, max(float(adult_depth_m), 0.0))

def update_drought_damage(
    drought_damage: np.ndarray,
    disconnection_hours: np.ndarray,
    *,
    bed_elevation: np.ndarray,
    groundwater_stage: float,
    alive: np.ndarray,
    alive_emerged: np.ndarray,
    recession_rate_cm_day: float,
    effective_age_hours: np.ndarray,
    dt_hours: float,
    initial_root_depth_m: float,
    adult_root_depth_m: float,
    adult_age_days: float,
    reference_hours_seedling: float,
    reference_hours_adult: float,
    damage_exponent: float,
    tolerance_hours_seedling: float = 48.0,
    tolerance_hours_adult: float = 14.0 * HOURS_PER_DAY,
    enabled: bool = True,
    damage_rate_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Update drought damage from actual root-zone disconnection.A live emerged plant is exposed when
    the instantaneous groundwater proxy is deeper than its age-dependent root
    depth. The consecutive tolerance increases linearly from 48 hours for a
    seedling to 14 days for an adult. Damage starts only after that tolerance.
    The preceding-72-hour recession class modulates severity: 0--5 cm/day = 1,
    5--10 cm/day = 2, and >10 cm/day = 3. This imports the empirically used
    Serlet/Burke recession classes without pretending that their annual M
    coefficient is itself a cell-level mortality equation.
    """
    damage = np.asarray(drought_damage, dtype=float)
    clock = np.asarray(disconnection_hours, dtype=float)
    bed = np.asarray(bed_elevation, dtype=float)
    alive = np.asarray(alive, dtype=bool)
    emerged = np.asarray(alive_emerged, dtype=bool)
    fraction = shared_ontogenetic_fraction(effective_age_hours, adult_age_days)
    root_depth = age_dependent_root_depth(
        effective_age_hours,
        initial_depth_m=initial_root_depth_m,
        adult_depth_m=adult_root_depth_m,
        adult_age_days=adult_age_days,
    )
    water_table_depth = np.maximum(bed - float(groundwater_stage), 0.0)
    deficit = np.maximum(water_table_depth - root_depth, 0.0)
    water_accessible = water_table_depth <= root_depth
    disconnected = alive & emerged & (deficit > 0.0)

    old_clock = clock.copy()
    clock[disconnected] += float(dt_hours)
    clock[~disconnected] = 0.0

    tolerance_hours = float(tolerance_hours_seedling) + fraction * (
        float(tolerance_hours_adult) - float(tolerance_hours_seedling)
    )
    reference_hours = float(reference_hours_seedling) + fraction * (
        float(reference_hours_adult) - float(reference_hours_seedling)
    )
    active_dt = np.maximum(
        np.minimum(clock, old_clock + float(dt_hours))
        - np.maximum(old_clock, tolerance_hours),
        0.0,
    )
    damaging = disconnected & (active_dt > 0.0)
    relative_deficit = deficit / np.maximum(root_depth, 0.05)
    recession = max(float(recession_rate_cm_day), 0.0)
    if recession <= 5.0:
        recession_multiplier = 1.0
    elif recession <= 10.0:
        recession_multiplier = 2.0
    else:
        recession_multiplier = 3.0

    if enabled and np.any(damaging):
        hydraulic_severity = (1.0 + relative_deficit[damaging]) ** max(
            float(damage_exponent), 1.0
        )
        damage[damaging] += (
            float(damage_rate_multiplier)
            * recession_multiplier
            * hydraulic_severity
            * active_dt[damaging]
            / np.maximum(reference_hours[damaging], 1e-12)
        )

    damage[~alive] = 0.0
    clock[~alive] = 0.0
    return damage, clock, {
        "ontogenetic_fraction": fraction,
        "root_depth_m": root_depth,
        "water_table_depth_m": water_table_depth,
        "drought_deficit_m": deficit,
        "water_accessible": water_accessible,
        "drought_exposed": disconnected,
        "drought_damaging": damaging,
        "drought_disconnection_hours": clock.copy(),
        "drought_tolerance_hours": tolerance_hours,
        "drought_reference_hours": reference_hours,
        "recession_rate_cm_day": np.full_like(damage, recession),
        "recession_damage_multiplier": np.full_like(damage, recession_multiplier),
    }


@dataclass(frozen=True)
class VegetationConfig:
    """Configuration for vegetation on a user-supplied Reach.
    Mortality is evaluated at the native timestamp resolution. Hourly discharge
    is therefore required for a genuinely hourly ecological analysis.
    """

    seed: int = 1
    initial_condition: str = "bare"
    minimum_emergence_fraction: float = 0.02
    biomass_presence_threshold: float = 0.05

    # Recruitment season and chronological establishment transition
    woo_days: float = 80.0
    seed_dispersal_stage_tolerance_m: float = 1e-6
    season_start_month: int = 4
    season_start_day: int = 1
    season_end_month: int = 9
    season_end_day: int = 30
    recruitment_box_lower_offset_m: float = 0.6
    recruitment_box_upper_offset_m: float = 2.0

    # Caponi inundation resistance: effective age grows only in the GS
    resistance_initial_hours: float = 1.0
    resistance_max_hours: float = 60.0 * 24.0
    resistance_growth_rate: float = 0.001

    # Other age-dependent traits use one configurable effective-age horizon
    adult_resistance_age_calendar_years: float = 15.0
    growing_season_days_per_year: float = 183.0
    resistance_transition_days: float | None = None

    # Shared cumulative-dose law for direct shear, bed mobility and drought
    dose_reference_hours_seedling: float = 24.0
    dose_reference_hours_adult: float = 144.0
    dose_damage_exponent: float = 1.5
    damage_recovery_half_after_hours: float = 7.0 * HOURS_PER_DAY
    damage_recovery_full_after_hours: float = 30.0 * HOURS_PER_DAY

    enable_shear_mortality: bool = True
    enable_shields_mortality: bool = True
    shear_seedling_critical_pa: float = 20.0
    shear_adult_critical_pa: float = 80.0
    shields_critical: float = 0.045
    shields_adult_multiplier: float = 8.0
    shields_damage_rate_multiplier: float = 0.20

    # Drought: actual root-zone disconnection; no discharge-percentile gate
    enable_drought_mortality: bool = True
    drought_tolerance_hours_seedling: float = 48.0
    drought_tolerance_hours_adult: float = 14.0 * HOURS_PER_DAY
    drought_damage_rate_multiplier: float = 1.0
    root_initial_depth_m: float = 0.05
    root_adult_depth_m: float = 4.0

    # Initial-state templates
    young_age_days: float = 180.0
    mature_age_days: float = 3.0 * 365.0
    young_presence_scale: float = 0.35
    mature_presence_scale: float = 0.75
    camporeale_background_biomass: float = 0.05
    camporeale_young_biomass: float = 0.20
    camporeale_mature_biomass: float = 0.75

    # Camporeale model
    camporeale_growth_rate: float = 0.005
    camporeale_flood_mortality: float = 5e-4
    camporeale_growth_exponent: float = 1.0
    camporeale_capacity_exponent: float = 1.0
    camporeale_mortality_exponent: float = 0.8
    groundwater_depth_scale_m: float = 1.0
    optimum_groundwater_depth_m: float = 0.5
    carrying_capacity_curvature: float = 0.055
    groundwater_coupling: float = 1.0

    # Hydraulic forcing generated from the user Reach
    hydraulic_method: str = "rating_curve"
    rating_curve_points: int = 120
    water_density: float = 1000.0
    sediment_density: float = 2650.0
    gravity: float = 9.81

    @property
    def adult_resistance_age_days(self) -> float:
        if self.resistance_transition_days is not None:
            return float(self.resistance_transition_days)
        return float(
            self.adult_resistance_age_calendar_years
            * self.growing_season_days_per_year
        )

    def validated(self) -> "VegetationConfig":
        condition = self.initial_condition.lower()
        if condition not in {"bare", "young", "mature", "custom"}:
            raise ValueError("initial_condition must be bare, young, mature, or custom")
        if not 0.0 <= self.minimum_emergence_fraction <= 1.0:
            raise ValueError("minimum_emergence_fraction must be in [0, 1]")
        if self.woo_days <= 0:
            raise ValueError("woo_days must be positive")
        if self.seed_dispersal_stage_tolerance_m < 0:
            raise ValueError("seed_dispersal_stage_tolerance_m must be non-negative")
        if self.recruitment_box_lower_offset_m < 0 or self.recruitment_box_upper_offset_m <= self.recruitment_box_lower_offset_m:
            raise ValueError("Recruitment Box offsets must satisfy 0 <= lower < upper")
        if self.resistance_initial_hours <= 0 or self.resistance_max_hours < self.resistance_initial_hours:
            raise ValueError("invalid Caponi resistance values")
        if self.resistance_growth_rate < 0:
            raise ValueError("resistance_growth_rate must be non-negative")
        if self.adult_resistance_age_days <= 0:
            raise ValueError("adult trait age must be positive")
        if self.hydraulic_method not in {"rating_curve", "exact"}:
            raise ValueError("hydraulic_method must be rating_curve or exact")
        if self.rating_curve_points < 12:
            raise ValueError("rating_curve_points must be at least 12")
        if self.shields_critical <= 0 or self.shear_seedling_critical_pa <= 0 or self.shear_adult_critical_pa <= 0:
            raise ValueError("mechanical critical values must be positive")
        if self.dose_reference_hours_seedling <= 0 or self.dose_reference_hours_adult <= 0:
            raise ValueError("dose reference durations must be positive")
        if self.damage_recovery_half_after_hours < 0 or self.damage_recovery_full_after_hours <= self.damage_recovery_half_after_hours:
            raise ValueError("damage recovery must satisfy 0 <= half-after < full-after")
        if self.drought_tolerance_hours_seedling < 0 or self.drought_tolerance_hours_adult < self.drought_tolerance_hours_seedling:
            raise ValueError("drought tolerance must increase from seedling to adult")
        if self.drought_damage_rate_multiplier < 0:
            raise ValueError("drought_damage_rate_multiplier must be non-negative")
        if self.root_initial_depth_m < 0 or self.root_adult_depth_m < self.root_initial_depth_m:
            raise ValueError("root depths must satisfy 0 <= initial <= adult")
        if not 0.0 <= self.groundwater_coupling <= 1.0:
            raise ValueError("groundwater_coupling must be in [0, 1]")
        return self


@dataclass
class HydraulicVegetationForcing:
    """Physical forcing passed to the vegetation models at user timestamps."""

    discharge: np.ndarray
    water_level: np.ndarray
    velocity: np.ndarray
    wetted_area: np.ndarray
    wetted_perimeter: np.ndarray
    depth: np.ndarray
    shear_stress: np.ndarray
    shields_d50: np.ndarray
    d50_m: float
    groundwater_stage: np.ndarray
    recession_rate_cm_day: np.ndarray


@dataclass
class VegetationResult:
    """Space-time output of one vegetation simulation."""

    mode: str
    flow_name: str
    initial_condition: str
    dates: pd.DatetimeIndex
    y: np.ndarray
    z: np.ndarray
    forcing: HydraulicVegetationForcing
    state: np.ndarray
    presence: np.ndarray
    age_days: np.ndarray
    effective_age_days: np.ndarray
    habitat_mask: np.ndarray
    mortality_cause: np.ndarray
    diagnostics: pd.DataFrame
    config: VegetationConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    established: np.ndarray | None = None
    candidate: np.ndarray | None = None

    @property
    def final_state(self) -> np.ndarray:
        return self.state[-1].copy()

    @property
    def final_presence(self) -> np.ndarray:
        return self.presence[-1].copy()

    @property
    def final_age_days(self) -> np.ndarray:
        return self.age_days[-1].copy()

    @property
    def final_effective_age_days(self) -> np.ndarray:
        return self.effective_age_days[-1].copy()

    @property
    def final_established(self) -> np.ndarray:
        if self.established is None:
            return self.final_presence.copy()
        return self.established[-1].copy()

    @property
    def final_candidate(self) -> np.ndarray:
        if self.candidate is None:
            return np.zeros_like(self.final_presence, dtype=bool)
        return self.candidate[-1].copy()

    def summary(self) -> dict[str, float | int | str]:
        """Return concise final and cumulative vegetation indicators."""
        out: dict[str, float | int | str] = {
            "mode": self.mode,
            "flow_name": self.flow_name,
            "initial_condition": self.initial_condition,
            "habitat_cells": int(self.habitat_mask.sum()),
            "final_present_cells": int(self.final_presence.sum()),
            "final_presence_fraction": float(
                self.final_presence[self.habitat_mask].mean()
                if self.habitat_mask.any()
                else 0.0
            ),
        }
        if self.mode == "camporeale":
            habitat = self.habitat_mask
            out["final_mean_biomass"] = float(
                np.mean(self.final_state[habitat]) if habitat.any() else 0.0
            )
            out["final_biomass_integral_m"] = float(
                trapezoid(self.final_state, self.y)
            )
        else:
            out["final_established_cells"] = int(self.final_established.sum())
            out["final_candidate_cells"] = int(self.final_candidate.sum())
            out["final_mean_age_days"] = float(
                np.mean(self.final_age_days[self.final_presence])
                if self.final_presence.any()
                else 0.0
            )
            out["final_mean_effective_age_days"] = float(
                np.mean(self.final_effective_age_days[self.final_presence])
                if self.final_presence.any()
                else 0.0
            )
            for column in (
                "recruits",
                "mortality",
                "mortality_anoxia",
                "mortality_shear",
                "mortality_sediment",
                "mortality_drought",
                "seedlings_created",
            ):
                out[f"total_{column}"] = int(self.diagnostics[column].sum()) if column in self.diagnostics else 0
        return out


@dataclass
class VegetationComparison:
    """Natural-versus-altered vegetation experiment."""

    natural: VegetationResult
    altered: VegetationResult
    scenario_name: str

    def summary(self) -> pd.DataFrame:
        """Return natural and altered summaries in tabular form."""
        return pd.DataFrame(
            [self.natural.summary(), self.altered.summary()],
            index=["natural", self.scenario_name],
        )


def _as_dates(dates: Any) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if index.empty:
        raise ValueError("dates cannot be empty")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be strictly increasing")
    return index


def _time_steps_hours(dates: pd.DatetimeIndex) -> np.ndarray:
    if dates.size == 1:
        return np.ones(1, dtype=float) * HOURS_PER_DAY
    seconds = np.diff(dates.asi8.astype(np.float64)) / 1e9
    hours = seconds / 3600.0
    if np.any(hours <= 0):
        raise ValueError("dates must be strictly increasing")
    return np.r_[hours[0], hours]


def _reach_geometry(reach: Any) -> tuple[np.ndarray, np.ndarray, float, float]:
    required = ("cross_section_coordinates", "slope", "ks")
    missing = [name for name in required if not hasattr(reach, name)]
    if missing:
        raise ValueError(
            "Reach geometry is incomplete. Call add_cross_section_geometry() first; "
            f"missing: {', '.join(missing)}"
        )
    section = reach.cross_section_coordinates
    y = np.asarray(section["y [m]"], dtype=float)
    z = np.asarray(section["z [m]"], dtype=float)
    if y.ndim != 1 or z.ndim != 1 or y.size != z.size or y.size < 2:
        raise ValueError("cross-section coordinates must be one-dimensional and aligned")
    return y, z, float(reach.slope), float(reach.ks)


def _reach_d50(reach: Any) -> float:
    """Return D50 [m] from the reach grain distribution."""
    if not hasattr(reach, "phi_percentages"):
        raise ValueError(
            "Reach grain-size distribution is missing. "
            "Call add_grain_size_distribution() before vegetation analysis."
        )
    fractions = np.asarray(reach.phi_percentages, dtype=float)
    if fractions.size != PHI_RANGE.size or np.any(fractions < 0):
        raise ValueError("reach.phi_percentages is invalid")
    total = fractions.sum()
    if total <= 0:
        raise ValueError("grain-size fractions sum to zero")
    fractions = fractions / total
    cumulative = np.cumsum(fractions)
    phi50 = float(np.interp(0.5, cumulative, PHI_RANGE))
    return float(2.0 ** (-phi50) / 1000.0)


def _solve_rating_curve(
    discharge: np.ndarray,
    slope: float,
    ks: float,
    y: np.ndarray,
    z: np.ndarray,
    config: VegetationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(discharge, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q < 0):
        raise ValueError("discharge must contain finite non-negative values")

    if config.hydraulic_method == "exact":
        nodes = np.unique(q)
    else:
        positive = q[q > 0]
        if positive.size == 0:
            nodes = np.array([0.0])
        else:
            probabilities = np.linspace(0.0, 1.0, config.rating_curve_points)
            nodes = np.unique(np.r_[0.0, np.quantile(positive, probabilities)])

    h_nodes = np.empty(nodes.size, dtype=float)
    area_nodes = np.empty(nodes.size, dtype=float)
    velocity_nodes = np.empty(nodes.size, dtype=float)
    perimeter_nodes = np.empty(nodes.size, dtype=float)

    for i, value in enumerate(nodes):
        if value <= 0.0:
            h_nodes[i] = float(np.min(z))
            area_nodes[i] = 0.0
            velocity_nodes[i] = 0.0
            perimeter_nodes[i] = 0.0
        else:
            solved = steady_flow_solver(float(value), slope, ks, y, z)
            # The original SARAwater solver normally returns four values.  One
            # legacy non-physical fallback returns three; handle it here so the
            # original hydraulics.py can remain byte-for-byte unchanged.
            if len(solved) == 4:
                h, area, velocity, perimeter = solved
            elif len(solved) == 3:
                h, area, velocity = solved
                perimeter = 0.0
            else:
                raise ValueError("Unexpected steady_flow_solver return length")
            h_nodes[i] = h
            area_nodes[i] = area
            velocity_nodes[i] = velocity
            perimeter_nodes[i] = perimeter

    if nodes.size == 1:
        shape = q.shape
        return (
            np.full(shape, h_nodes[0]),
            np.full(shape, area_nodes[0]),
            np.full(shape, velocity_nodes[0]),
            np.full(shape, perimeter_nodes[0]),
        )

    return (
        np.interp(q, nodes, h_nodes),
        np.interp(q, nodes, area_nodes),
        np.interp(q, nodes, velocity_nodes),
        np.interp(q, nodes, perimeter_nodes),
    )



def _dynamic_groundwater_stage(water_level, dates):
    """Return the current river level as the instantaneous groundwater proxy."""
    h = np.asarray(water_level, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if h.ndim != 1 or h.size != dates.size:
        raise ValueError("water_level and dates must be aligned 1D arrays")
    return h.copy()


def compute_vegetation_forcing(
    reach: Any,
    discharge: np.ndarray,
    config: VegetationConfig | None = None,
) -> HydraulicVegetationForcing:
    cfg = (config or VegetationConfig()).validated()
    q = np.asarray(discharge, dtype=float)
    y, z, slope, ks = _reach_geometry(reach)
    d50 = _reach_d50(reach)
    dates = _as_dates(reach.dates)

    digest = hashlib.blake2b(digest_size=16)
    for array in (q, y, z):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update(
        repr((
            slope, ks, d50, cfg.hydraulic_method, cfg.rating_curve_points,
            cfg.water_density, cfg.sediment_density, cfg.gravity,
        )).encode("utf-8")
    )
    cache_key = digest.hexdigest()
    cache = getattr(reach, "_vegetation_forcing_cache", None)
    if cache is None:
        cache = {}
        reach._vegetation_forcing_cache = cache
    if cache_key in cache:
        return cache[cache_key]

    h, area, velocity, perimeter = _solve_rating_curve(q, slope, ks, y, z, cfg)
    depth = np.maximum(h[:, None] - z[None, :], 0.0)
    tau = cfg.water_density * cfg.gravity * depth * slope
    theta = shields_parameter(
        tau, cfg.water_density, cfg.sediment_density, cfg.gravity, d50
    )
    groundwater_stage = _dynamic_groundwater_stage(h, dates)
    recession_rate = _rolling_recession_rate_cm_day(h, dates)
    forcing = HydraulicVegetationForcing(
        discharge=q.copy(),
        water_level=h,
        velocity=velocity,
        wetted_area=area,
        wetted_perimeter=perimeter,
        depth=depth,
        shear_stress=tau,
        shields_d50=theta,
        d50_m=d50,
        groundwater_stage=groundwater_stage,
        recession_rate_cm_day=recession_rate,
    )
    cache[cache_key] = forcing
    return forcing


def habitat_emergence_mask(
    z: np.ndarray,
    water_level: np.ndarray,
    minimum_emergence_fraction: float,
) -> np.ndarray:
    """Identify cells that emerge during a sufficient fraction of the record."""
    emergence_fraction = np.mean(
        np.asarray(water_level)[:, None] <= np.asarray(z)[None, :], axis=0
    )
    return emergence_fraction >= minimum_emergence_fraction


def _suitability(
    z: np.ndarray, reference_stage: float, habitat: np.ndarray
) -> np.ndarray:
    """Initial topographic suitability aligned with vegetation_init.py."""
    relative = np.maximum(np.asarray(z, float) - float(reference_stage), 0.0)
    if habitat.any():
        scale = max(float(np.nanpercentile(relative[habitat], 90.0)), 1e-12)
    else:
        scale = max(float(np.ptp(z)), 1e-12)
    return np.clip(relative / scale, 0.0, 1.0) * habitat


def initialize_vegetation(
    mode: str,
    z: np.ndarray,
    forcing: HydraulicVegetationForcing,
    config: VegetationConfig,
    initial_state: Mapping[str, Any] | None = None,
    habitat_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create a reproducible vegetation state for the selected condition.
    Recruitment stores chronological and growing-season effective age
    separately. The latter controls Caponi inundation resistance.
    """
    mode = mode.lower()
    if mode not in {"recruitment", "camporeale"}:
        raise ValueError("mode must be 'recruitment' or 'camporeale'")
    z = np.asarray(z, dtype=float)
    n = z.size
    habitat = (
        habitat_emergence_mask(z, forcing.water_level, config.minimum_emergence_fraction)
        if habitat_mask is None
        else np.asarray(habitat_mask, dtype=bool).copy()
    )
    if habitat.shape != z.shape:
        raise ValueError("habitat_mask must have the same shape as z")

    rng = np.random.default_rng(config.seed)
    reference_stage = float(np.nanmedian(forcing.water_level))
    suitability = _suitability(z, reference_stage, habitat)
    suitability = np.clip(suitability * rng.uniform(0.9, 1.1, n), 0.0, 1.0)

    presence = np.zeros(n, dtype=bool)
    chronological_age_hours = np.zeros(n, dtype=float)
    effective_age_hours = np.zeros(n, dtype=float)
    recruitment_year = np.full(n, -1, dtype=int)
    biomass = np.zeros(n, dtype=float)
    condition = config.initial_condition.lower()

    if condition == "custom":
        if not initial_state:
            raise ValueError("initial_state is required when initial_condition='custom'")
        if mode == "recruitment":
            if "presence" not in initial_state:
                raise ValueError("custom recruitment state requires 'presence'")
            presence = np.asarray(initial_state["presence"], dtype=bool).copy()
            if presence.shape != z.shape:
                raise ValueError("custom presence must match the cross-section")
            if "age_days" in initial_state:
                chronological_age_hours = np.asarray(initial_state["age_days"], dtype=float) * HOURS_PER_DAY
                if chronological_age_hours.shape != z.shape:
                    raise ValueError("custom age_days must match the cross-section")
            if "effective_age_days" in initial_state:
                effective_age_hours = np.asarray(initial_state["effective_age_days"], dtype=float) * HOURS_PER_DAY
                if effective_age_hours.shape != z.shape:
                    raise ValueError("custom effective_age_days must match the cross-section")
            else:
                effective_age_hours = effective_age_from_chronological(chronological_age_hours)
            biomass = presence.astype(float)
            recruitment_year[presence] = -2
        else:
            if "biomass" not in initial_state:
                raise ValueError("custom Camporeale state requires 'biomass'")
            biomass = np.clip(np.asarray(initial_state["biomass"], dtype=float), 0.0, 1.0)
            if biomass.shape != z.shape:
                raise ValueError("custom biomass must match the cross-section")
            presence = biomass >= config.biomass_presence_threshold
    elif mode == "recruitment":
        if condition == "young":
            presence = habitat & (rng.random(n) < config.young_presence_scale * suitability)
            chronological_age_hours[presence] = config.young_age_days * HOURS_PER_DAY
        elif condition == "mature":
            presence = habitat & (rng.random(n) < config.mature_presence_scale * suitability)
            chronological_age_hours[presence] = config.mature_age_days * HOURS_PER_DAY
        elif condition != "bare":
            raise ValueError("unsupported recruitment initial condition")
        effective_age_hours[:] = effective_age_from_chronological(chronological_age_hours)
        recruitment_year[presence] = -2
        biomass = presence.astype(float)
    else:
        if condition == "bare":
            biomass[habitat] = config.camporeale_background_biomass * np.maximum(suitability[habitat], 0.15)
        elif condition == "young":
            biomass[habitat] = config.camporeale_young_biomass * suitability[habitat]
        elif condition == "mature":
            biomass[habitat] = config.camporeale_mature_biomass * suitability[habitat]
        else:
            raise ValueError("unsupported Camporeale initial condition")
        presence = biomass >= config.biomass_presence_threshold

    presence &= habitat
    biomass[~habitat] = 0.0
    chronological_age_hours[~presence] = 0.0
    effective_age_hours[~presence] = 0.0
    return {
        "presence": presence,
        # Pre-existing vegetation is established. New seedling cohorts are
        # created explicitly with established=False during the simulation.
        "established": presence.copy(),
        "age_hours": chronological_age_hours,  # backward-compatible alias
        "chronological_age_hours": chronological_age_hours,
        "effective_age_hours": effective_age_hours,
        "recruitment_year": recruitment_year,
        "biomass": biomass,
        "habitat_mask": habitat,
        "woo_hours": np.zeros(n, dtype=float),
        "anoxia_hours": np.zeros(n, dtype=float),
        "shear_damage": np.zeros(n, dtype=float),
        "sediment_damage": np.zeros(n, dtype=float),
        "drought_damage": np.zeros(n, dtype=float),
        "drought_disconnection_hours": np.zeros(n, dtype=float),
        "shear_recovery_hours": np.zeros(n, dtype=float),
        "shear_recovery_reference": np.zeros(n, dtype=float),
        "sediment_recovery_hours": np.zeros(n, dtype=float),
        "sediment_recovery_reference": np.zeros(n, dtype=float),
        "drought_recovery_hours": np.zeros(n, dtype=float),
        "drought_recovery_reference": np.zeros(n, dtype=float),
        "resistance_hours": caponi_resistance(effective_age_hours, config),
    }

def caponi_resistance(effective_age_hours: np.ndarray, config: VegetationConfig) -> np.ndarray:
    """Caponi logistic resistance driven by growing-season age only."""
    return _caponi_resistance(
        effective_age_hours,
        config.resistance_initial_hours,
        config.resistance_max_hours,
        config.resistance_growth_rate,
    )


def _in_growing_season(date: pd.Timestamp, config: VegetationConfig) -> bool:
    value = (date.month, date.day)
    return (config.season_start_month, config.season_start_day) <= value <= (
        config.season_end_month,
        config.season_end_day,
    )

def _band_for_date(
    bands: dict[int, AnnualRecruitmentBand], date: pd.Timestamp
) -> AnnualRecruitmentBand | None:
    """Return the annual opportunity band for the calendar year."""
    return bands.get(int(date.year))


def _recruitment_step(
    state: dict[str, np.ndarray],
    z: np.ndarray,
    water_level: float,
    tau: np.ndarray,
    theta: np.ndarray,
    date: pd.Timestamp,
    dt_hours: float,
    config: VegetationConfig,
    annual_band: AnnualRecruitmentBand | None,
    groundwater_stage: float | None = None,
    recession_rate_cm_day: float = 0.0,
    previous_water_level: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    presence = state["presence"]
    established = state["established"]
    chronological = state["chronological_age_hours"]
    effective = state["effective_age_hours"]
    recruitment_year = state["recruitment_year"]
    habitat = state["habitat_mask"]
    woo = state["woo_hours"]
    anoxia = state["anoxia_hours"]
    shear_damage = state["shear_damage"]
    sediment_damage = state["sediment_damage"]
    drought_damage = state["drought_damage"]
    drought_clock = state["drought_disconnection_hours"]
    shear_recovery_hours = state["shear_recovery_hours"]
    shear_recovery_reference = state["shear_recovery_reference"]
    sediment_recovery_hours = state["sediment_recovery_hours"]
    sediment_recovery_reference = state["sediment_recovery_reference"]
    drought_recovery_hours = state["drought_recovery_hours"]
    drought_recovery_reference = state["drought_recovery_reference"]
    if groundwater_stage is None:
        groundwater_stage = float(water_level)

    current_stage = float(water_level)
    submerged = current_stage > z
    emerged = ~submerged
    growing = _in_growing_season(date, config)

    # Biological age is chronological. Effective trait age grows only in GS.
    pre_existing = presence.copy()
    chronological[pre_existing] += float(dt_hours)
    if growing:
        effective[pre_existing] += float(dt_hours)
    resistance = caponi_resistance(effective, config)

    ebe1_band = np.zeros_like(presence)
    if annual_band is not None:
        ebe1_band = (z >= annual_band.ebe1_lower_m) & (z <= annual_band.ebe1_upper_m)
    band_valid = bool(annual_band is not None and annual_band.ebe1_valid)

    # Unlimited propagule supply: falling stage exposes a thin elevation band.
    new_seedlings = np.zeros_like(presence)
    if growing and band_valid and previous_water_level is not None:
        previous_stage = float(previous_water_level)
        falling = current_stage < previous_stage - float(config.seed_dispersal_stage_tolerance_m)
        if falling:
            newly_exposed = emerged & (z <= previous_stage)
            new_seedlings = (~presence) & habitat & ebe1_band & newly_exposed
            presence[new_seedlings] = True
            established[new_seedlings] = False
            chronological[new_seedlings] = 0.0
            effective[new_seedlings] = 0.0
            recruitment_year[new_seedlings] = int(annual_band.year)
            woo[new_seedlings] = 0.0
            anoxia[new_seedlings] = 0.0
            shear_damage[new_seedlings] = 0.0
            sediment_damage[new_seedlings] = 0.0
            drought_damage[new_seedlings] = 0.0
            drought_clock[new_seedlings] = 0.0
            shear_recovery_hours[new_seedlings] = 0.0
            shear_recovery_reference[new_seedlings] = 0.0
            sediment_recovery_hours[new_seedlings] = 0.0
            sediment_recovery_reference[new_seedlings] = 0.0
            drought_recovery_hours[new_seedlings] = 0.0
            drought_recovery_reference[new_seedlings] = 0.0
            resistance[new_seedlings] = float(config.resistance_initial_hours)

    # The candidate clock is consecutive elapsed survival time, not effective age.
    candidate = presence & ~established
    woo[~candidate] = 0.0
    woo[candidate] += float(dt_hours)
    newly_established = candidate & (woo >= float(config.woo_days) * HOURS_PER_DAY)
    established[newly_established] = True
    woo[newly_established] = 0.0

    alive_submerged = presence & submerged
    alive_emerged = presence & emerged
    anoxia[alive_submerged] += float(dt_hours)
    anoxia[alive_emerged] = 0.0

    shear_damage, sediment_damage, mech = update_mechanical_damage(
        shear_damage, sediment_damage,
        tau=np.asarray(tau, float), theta=np.asarray(theta, float),
        submerged_alive=alive_submerged, emerged_alive=alive_emerged,
        effective_age_hours=effective, dt_hours=float(dt_hours),
        shear_seedling_critical_pa=config.shear_seedling_critical_pa,
        shear_adult_critical_pa=config.shear_adult_critical_pa,
        shields_bed_critical=config.shields_critical,
        shields_adult_resistance_multiplier=config.shields_adult_multiplier,
        adult_age_days=config.adult_resistance_age_days,
        reference_hours_seedling=config.dose_reference_hours_seedling,
        reference_hours_adult=config.dose_reference_hours_adult,
        damage_exponent=config.dose_damage_exponent,
        enable_shear=config.enable_shear_mortality,
        enable_shields=config.enable_shields_mortality,
        shields_damage_rate_multiplier=config.shields_damage_rate_multiplier,
    )

    shear_stress_active = alive_submerged & (mech["shear_excess"] > 0.0)
    sediment_stress_active = alive_submerged & (mech["shields_excess"] > 0.0)
    shear_damage, shear_recovery_hours, shear_recovery_reference = _apply_scheduled_recovery(
        shear_damage, shear_recovery_hours, shear_recovery_reference,
        alive=presence, stress_active=shear_stress_active, dt_hours=float(dt_hours),
        half_after_hours=config.damage_recovery_half_after_hours,
        full_after_hours=config.damage_recovery_full_after_hours,
    )
    sediment_damage, sediment_recovery_hours, sediment_recovery_reference = _apply_scheduled_recovery(
        sediment_damage, sediment_recovery_hours, sediment_recovery_reference,
        alive=presence, stress_active=sediment_stress_active, dt_hours=float(dt_hours),
        half_after_hours=config.damage_recovery_half_after_hours,
        full_after_hours=config.damage_recovery_full_after_hours,
    )

    drought_damage, drought_clock, drought_fields = update_drought_damage(
        drought_damage, drought_clock,
        bed_elevation=z, groundwater_stage=float(groundwater_stage),
        alive=presence, alive_emerged=alive_emerged,
        recession_rate_cm_day=float(recession_rate_cm_day),
        effective_age_hours=effective, dt_hours=float(dt_hours),
        initial_root_depth_m=config.root_initial_depth_m,
        adult_root_depth_m=config.root_adult_depth_m,
        adult_age_days=config.adult_resistance_age_days,
        reference_hours_seedling=config.dose_reference_hours_seedling,
        reference_hours_adult=config.dose_reference_hours_adult,
        damage_exponent=config.dose_damage_exponent,
        tolerance_hours_seedling=config.drought_tolerance_hours_seedling,
        tolerance_hours_adult=config.drought_tolerance_hours_adult,
        enabled=config.enable_drought_mortality,
        damage_rate_multiplier=config.drought_damage_rate_multiplier,
    )
    drought_damage, drought_recovery_hours, drought_recovery_reference = _apply_scheduled_recovery(
        drought_damage, drought_recovery_hours, drought_recovery_reference,
        alive=presence, stress_active=drought_fields["drought_exposed"],
        dt_hours=float(dt_hours),
        half_after_hours=config.damage_recovery_half_after_hours,
        full_after_hours=config.damage_recovery_full_after_hours,
    )

    fail_anoxia = presence & (anoxia > resistance)
    fail_shear = presence & (shear_damage >= 1.0) & config.enable_shear_mortality
    fail_sediment = presence & (sediment_damage >= 1.0) & config.enable_shields_mortality
    fail_drought = presence & (drought_damage >= 1.0) & config.enable_drought_mortality
    dead = fail_anoxia | fail_shear | fail_sediment | fail_drought
    candidate_before_death = presence & ~established
    established_before_death = presence & established

    # Exclusive attribution when multiple thresholds are crossed together.
    dead_drought = dead & fail_drought
    remaining = dead & ~dead_drought
    dead_shear = remaining & fail_shear
    remaining &= ~dead_shear
    dead_anoxia = remaining & fail_anoxia
    remaining &= ~dead_anoxia
    dead_sediment = remaining & fail_sediment

    cause = np.zeros(z.size, dtype=np.int8)
    cause[dead_anoxia] = 1
    cause[dead_shear] = 2
    cause[dead_sediment] = 3
    cause[dead_drought] = 4

    presence[dead] = False
    established[dead] = False
    chronological[dead] = 0.0
    effective[dead] = 0.0
    recruitment_year[dead] = -1
    woo[dead] = 0.0
    anoxia[dead] = 0.0
    shear_damage[dead] = 0.0
    sediment_damage[dead] = 0.0
    drought_damage[dead] = 0.0
    drought_clock[dead] = 0.0
    shear_recovery_hours[dead] = 0.0
    shear_recovery_reference[dead] = 0.0
    sediment_recovery_hours[dead] = 0.0
    sediment_recovery_reference[dead] = 0.0
    drought_recovery_hours[dead] = 0.0
    drought_recovery_reference[dead] = 0.0
    resistance[dead] = config.resistance_initial_hours

    outside = ~habitat
    presence[outside] = False
    established[outside] = False
    chronological[outside] = 0.0
    effective[outside] = 0.0
    recruitment_year[outside] = -1
    woo[outside] = 0.0
    anoxia[outside] = 0.0
    shear_damage[outside] = 0.0
    sediment_damage[outside] = 0.0
    drought_damage[outside] = 0.0
    drought_clock[outside] = 0.0
    shear_recovery_hours[outside] = 0.0
    shear_recovery_reference[outside] = 0.0
    sediment_recovery_hours[outside] = 0.0
    sediment_recovery_reference[outside] = 0.0
    drought_recovery_hours[outside] = 0.0
    drought_recovery_reference[outside] = 0.0
    resistance[outside] = config.resistance_initial_hours

    state["age_hours"] = chronological
    state["resistance_hours"] = resistance
    state["established"] = established
    state["drought_disconnection_hours"] = drought_clock
    state["shear_recovery_hours"] = shear_recovery_hours
    state["shear_recovery_reference"] = shear_recovery_reference
    state["sediment_recovery_hours"] = sediment_recovery_hours
    state["sediment_recovery_reference"] = sediment_recovery_reference
    state["drought_recovery_hours"] = drought_recovery_hours
    state["drought_recovery_reference"] = drought_recovery_reference
    diagnostics = {
        "seedlings_created": int(new_seedlings.sum()),
        "recruits": int(newly_established.sum()),
        "candidate_cells": int((presence & ~established).sum()),
        "established_cells": int(established.sum()),
        "mortality": int(dead.sum()),
        "mortality_candidate_seedlings": int((dead & candidate_before_death).sum()),
        "mortality_established": int((dead & established_before_death).sum()),
        "mortality_anoxia": int(dead_anoxia.sum()),
        "mortality_shear": int(dead_shear.sum()),
        "mortality_sediment": int(dead_sediment.sum()),
        "mortality_drought": int(dead_drought.sum()),
        "anoxia_failure_cells": int(fail_anoxia.sum()),
        "shear_failure_cells": int(fail_shear.sum()),
        "sediment_failure_cells": int(fail_sediment.sum()),
        "drought_failure_cells": int(fail_drought.sum()),
        "drought_exposed_cells": int(np.sum(drought_fields["drought_exposed"] & presence)),
        "drought_damaging_cells": int(np.sum(drought_fields["drought_damaging"] & presence)),
        "mean_drought_damage": float(np.mean(drought_damage[presence]) if presence.any() else 0.0),
        "mean_drought_disconnection_hours": float(np.mean(drought_clock[presence]) if presence.any() else 0.0),
        "max_drought_disconnection_hours": float(np.max(drought_clock[presence]) if presence.any() else 0.0),
        "mean_root_depth_m": float(np.mean(drought_fields["root_depth_m"][presence]) if presence.any() else 0.0),
        "mean_water_table_depth_m": float(np.mean(drought_fields["water_table_depth_m"][presence]) if presence.any() else 0.0),
        "current_recession_rate_cm_day": float(recession_rate_cm_day),
        "drought_recession_multiplier": float(drought_fields["recession_damage_multiplier"][0]) if drought_fields["recession_damage_multiplier"].size else 1.0,
        "mean_drought_tolerance_hours": float(np.mean(drought_fields["drought_tolerance_hours"][presence]) if presence.any() else 0.0),
        "submerged_alive_cells": int(alive_submerged.sum()),
        "present_cells": int(presence.sum()),
        "presence_fraction": float(presence[habitat].mean() if habitat.any() else 0.0),
        "mean_age_days": float(np.mean(chronological[presence]) / HOURS_PER_DAY if presence.any() else 0.0),
        "mean_effective_age_days": float(np.mean(effective[presence]) / HOURS_PER_DAY if presence.any() else 0.0),
        "mean_candidate_elapsed_days": float(np.mean(woo[presence & ~established]) / HOURS_PER_DAY if np.any(presence & ~established) else 0.0),
        "mean_resistance_hours": float(np.mean(resistance[presence]) if presence.any() else 0.0),
        "mean_anoxia_hours": float(np.mean(anoxia[presence]) if presence.any() else 0.0),
        "max_anoxia_hours": float(np.max(anoxia[presence]) if presence.any() else 0.0),
        "mean_shear_damage": float(np.mean(shear_damage[presence]) if presence.any() else 0.0),
        "mean_sediment_damage": float(np.mean(sediment_damage[presence]) if presence.any() else 0.0),
        "shear_exceedance_cells": int((alive_submerged & (mech["shear_excess"] > 0)).sum()),
        "sediment_exceedance_cells": int((alive_submerged & (mech["shields_excess"] > 0)).sum()),
        "ebe1_valid": bool(annual_band.ebe1_valid) if annual_band else False,
        "recession_mortality_coefficient_diagnostic": float(annual_band.recession_mortality_coefficient) if annual_band else np.nan,
    }
    return state, diagnostics, cause


def _camporeale_step(
    state: dict[str, np.ndarray],
    z: np.ndarray,
    water_level: float,
    reference_stage: float,
    dt_hours: float,
    config: VegetationConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    biomass = state["biomass"]
    habitat = state["habitat_mask"]
    old = np.clip(biomass.copy(), 0.0, 1.0)
    new, fields = camporeale_dichotomous_step(
        old,
        z,
        water_level,
        reference_stage,
        dt_hours,
        growth_rate_per_day=config.camporeale_growth_rate,
        flood_mortality_per_day=config.camporeale_flood_mortality,
        growth_exponent=config.camporeale_growth_exponent,
        capacity_exponent=config.camporeale_capacity_exponent,
        mortality_exponent=config.camporeale_mortality_exponent,
        depth_scale_m=config.groundwater_depth_scale_m,
        optimum_depth_m=config.optimum_groundwater_depth_m,
        capacity_curvature=config.carrying_capacity_curvature,
        groundwater_coupling=config.groundwater_coupling,
    )
    biomass[:] = new
    biomass[~habitat] = 0.0
    presence = biomass >= config.biomass_presence_threshold
    state["presence"] = presence
    state["established"] = presence.copy()
    state["chronological_age_hours"][presence] += dt_hours
    state["chronological_age_hours"][~presence] = 0.0
    state["age_hours"] = state["chronological_age_hours"]

    crossed = (old >= config.biomass_presence_threshold) & ~presence
    diagnostics = {
        "recruits": 0,
        "mortality": int(crossed.sum()),
        "mortality_anoxia": int(crossed.sum()),
        "mortality_shear": 0,
        "mortality_sediment": 0,
        "mortality_drought": 0,
        "present_cells": int(presence.sum()),
        "presence_fraction": float(presence[habitat].mean() if habitat.any() else 0.0),
        "mean_age_days": float(np.mean(state["chronological_age_hours"][presence]) / HOURS_PER_DAY if presence.any() else 0.0),
        "mean_effective_age_days": 0.0,
        "mean_biomass": float(np.mean(biomass[habitat]) if habitat.any() else 0.0),
        "biomass_integral_m": 0.0,
        "mean_carrying_capacity": float(np.mean(fields["capacity"][habitat]) if habitat.any() else 0.0),
        "mean_groundwater_depth_m": float(np.mean(fields["groundwater_depth"][habitat]) if habitat.any() else 0.0),
        "mean_growth_rate": float(np.mean(fields["growth"][habitat]) if habitat.any() else 0.0),
        "mean_flood_decay_rate": float(np.mean(fields["decay"][habitat]) if habitat.any() else 0.0),
        "growth_cells": int(np.sum(fields["growth"] > 0.0)),
        "decay_cells": int(np.sum(fields["decay"] > 0.0)),
    }
    cause = np.zeros(z.size, dtype=np.int8)
    cause[crossed] = 1
    return state, diagnostics, cause

def run_vegetation(
    reach: Any,
    discharge: np.ndarray,
    mode: str = "recruitment",
    config: VegetationConfig | None = None,
    *,
    flow_name: str = "natural",
    initial_condition: str | None = None,
    initial_state: Mapping[str, Any] | None = None,
    forcing: HydraulicVegetationForcing | None = None,
    habitat_mask: np.ndarray | None = None,
) -> VegetationResult:
    """Run one vegetation model using a reach and a discharge series."""
    cfg = config or VegetationConfig()
    if initial_condition is not None:
        cfg = replace(cfg, initial_condition=initial_condition)
    cfg = cfg.validated()
    mode = mode.lower()
    if mode not in {"recruitment", "camporeale"}:
        raise ValueError("mode must be 'recruitment' or 'camporeale'")

    dates = _as_dates(reach.dates)
    q = np.asarray(discharge, dtype=float)
    if q.shape != (dates.size,):
        raise ValueError("discharge must have the same length as reach.dates")
    y, z, _, _ = _reach_geometry(reach)
    physical = forcing or compute_vegetation_forcing(reach, q, cfg)
    state = initialize_vegetation(
        mode,
        z,
        physical,
        cfg,
        initial_state=initial_state,
        habitat_mask=habitat_mask,
    )
    fixed_initial = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in state.items()
    }
    dt_hours = _time_steps_hours(dates)
    recruitment_bands = compute_serlet_bands(
        dates,
        physical.water_level,
        season_start=(cfg.season_start_month, cfg.season_start_day),
        season_end=(cfg.season_end_month, cfg.season_end_day),
        woo_days=float(cfg.woo_days),
        lower_offset_m=cfg.recruitment_box_lower_offset_m,
        upper_offset_m=cfg.recruitment_box_upper_offset_m,
        section_min_m=float(np.nanmin(z)),
        section_max_m=float(np.nanmax(z)),
    )
    groundwater_reference_stage = float(np.nanmin(physical.water_level))

    n_time = dates.size
    n_cells = z.size
    state_history = np.zeros((n_time, n_cells), dtype=float)
    presence_history = np.zeros((n_time, n_cells), dtype=bool)
    age_history = np.zeros((n_time, n_cells), dtype=float)
    effective_age_history = np.zeros((n_time, n_cells), dtype=float)
    mortality_history = np.zeros((n_time, n_cells), dtype=np.int8)
    established_history = np.zeros((n_time, n_cells), dtype=bool)
    candidate_history = np.zeros((n_time, n_cells), dtype=bool)
    records: list[dict[str, Any]] = []

    for t, date in enumerate(dates):
        if mode == "recruitment":
            state, diagnostics, cause = _recruitment_step(
                state,
                z,
                float(physical.water_level[t]),
                physical.shear_stress[t],
                physical.shields_d50[t],
                pd.Timestamp(date),
                float(dt_hours[t]),
                cfg,
                _band_for_date(recruitment_bands, pd.Timestamp(date)),
                float(physical.groundwater_stage[t]),
                float(physical.recession_rate_cm_day[t]),
                previous_water_level=(
                    float(physical.water_level[t - 1]) if t > 0 else float(physical.water_level[t])
                ),
            )
            state_value = state["presence"].astype(float)
        else:
            state, diagnostics, cause = _camporeale_step(
                state,
                z,
                float(physical.water_level[t]),
                groundwater_reference_stage,
                float(dt_hours[t]),
                cfg,
            )
            diagnostics["biomass_integral_m"] = float(
                trapezoid(state["biomass"], y)
            )
            state_value = state["biomass"]

        state_history[t] = state_value
        presence_history[t] = state["presence"]
        age_history[t] = state["chronological_age_hours"] / HOURS_PER_DAY
        effective_age_history[t] = state["effective_age_hours"] / HOURS_PER_DAY
        mortality_history[t] = cause
        established_history[t] = state.get("established", state["presence"])
        candidate_history[t] = state["presence"] & ~state.get("established", state["presence"])
        records.append(
            {
                "date": pd.Timestamp(date),
                "discharge": float(q[t]),
                "water_level": float(physical.water_level[t]),
                **diagnostics,
            }
        )

    result = VegetationResult(
        mode=mode,
        flow_name=flow_name,
        initial_condition=cfg.initial_condition,
        dates=dates,
        y=y.copy(),
        z=z.copy(),
        forcing=physical,
        state=state_history,
        presence=presence_history,
        age_days=age_history,
        effective_age_days=effective_age_history,
        habitat_mask=state["habitat_mask"].copy(),
        mortality_cause=mortality_history,
        diagnostics=pd.DataFrame.from_records(records).set_index("date"),
        config=cfg,
        metadata={
            "initial_state": fixed_initial,
            "recruitment_bands": {year: band.to_dict() for year, band in recruitment_bands.items()},
            "groundwater_reference_stage": groundwater_reference_stage,
            "scientific_modes": {
                "caponi_effective_age": True,
                "caponi_sigma_per_hour": cfg.resistance_growth_rate,
                "caponi_rmax_hours": cfg.resistance_max_hours,
                "mahoney_rood_recruitment_box": True,
                "serlet_woo_ebe1_adjustment": True,
                "winter_survival_is_process_based": True,
                "candidate_establishment_clock": "chronological consecutive hours",
                "candidate_establishment_days": cfg.woo_days,
                "drought_mortality": True,
                "drought_definition": "root-zone disconnection with age-dependent 48 h to 14 d tolerance",
                "drought_tolerance_hours_seedling": cfg.drought_tolerance_hours_seedling,
                "drought_tolerance_hours_adult": cfg.drought_tolerance_hours_adult,
                "drought_recession_modifier": "72 h mean recession classes: 1x <=5, 2x 5-10, 3x >10 cm/day",
                "damage_recovery_schedule": "linear from stress cessation: 50% at 7 stress-free days; zero at 30 days",
                "camporeale_dichotomous": True,
                "mechanical_extension": True,
                "explicit_candidate_seedlings": True,
                "mortality_during_candidate_phase": True,
                "instant_groundwater_tracking": True,
                "input_architecture": "user Reach cross-section and user discharge only",
                "native_timestep_hours_max": float(np.max(dt_hours)),
                "hourly_input_recommended": True,
                "recruitment_box_model": "Mahoney and Rood 1998 metric offsets",
                "fixed_shields_bed_threshold": cfg.shields_critical,
                "shared_adult_effective_age_days": cfg.adult_resistance_age_days,
            },
        },
        established=established_history,
        candidate=candidate_history,
    )
    return result


def run_reach_vegetation(
    reach: Any,
    mode: str = "recruitment",
    config: VegetationConfig | None = None,
    *,
    initial_condition: str | None = None,
    initial_state: Mapping[str, Any] | None = None,
) -> VegetationResult:
    """Run vegetation under the natural discharge of a Reach."""
    result = run_vegetation(
        reach,
        reach.Qnat,
        mode,
        config,
        flow_name="natural",
        initial_condition=initial_condition,
        initial_state=initial_state,
    )
    if not hasattr(reach, "vegetation_results"):
        reach.vegetation_results = {}
    reach.vegetation_results[("natural", mode, result.initial_condition)] = result
    return result


def run_scenario_vegetation(
    scenario: Any,
    mode: str = "recruitment",
    config: VegetationConfig | None = None,
    *,
    initial_condition: str | None = None,
    initial_state: Mapping[str, Any] | None = None,
) -> VegetationResult:
    """Run vegetation under a Scenario's released discharge."""
    if scenario.Qrel is None:
        scenario.compute_Qrel()
    result = run_vegetation(
        scenario.reach,
        scenario.Qrel,
        mode,
        config,
        flow_name=scenario.name,
        initial_condition=initial_condition,
        initial_state=initial_state,
    )
    if not hasattr(scenario, "vegetation_results"):
        scenario.vegetation_results = {}
    scenario.vegetation_results[(mode, result.initial_condition)] = result
    return result


def compare_vegetation(
    reach: Any,
    scenario: Any,
    mode: str = "recruitment",
    config: VegetationConfig | None = None,
    *,
    initial_condition: str | None = None,
) -> VegetationComparison:
    """Run a fair natural-versus-altered comparison.

    Both simulations receive exactly the same initial vegetation state and the
    habitat domain derived from the natural flow record.  This isolates the
    effect of altered discharge from differences in random initialization.
    """
    cfg = config or VegetationConfig()
    if initial_condition is not None:
        cfg = replace(cfg, initial_condition=initial_condition)
    cfg = cfg.validated()
    if scenario.reach is not reach:
        raise ValueError("scenario must belong to the supplied reach")
    if scenario.Qrel is None:
        scenario.compute_Qrel()

    natural_forcing = compute_vegetation_forcing(reach, reach.Qnat, cfg)
    y, z, _, _ = _reach_geometry(reach)
    shared_habitat = habitat_emergence_mask(
        z,
        natural_forcing.water_level,
        cfg.minimum_emergence_fraction,
    )
    initial = initialize_vegetation(
        mode,
        z,
        natural_forcing,
        cfg,
        habitat_mask=shared_habitat,
    )
    if mode == "recruitment":
        initial_public: dict[str, Any] = {
            "presence": initial["presence"].copy(),
            "age_days": initial["chronological_age_hours"].copy() / 24.0,
            "effective_age_days": initial["effective_age_hours"].copy() / 24.0,
        }
    else:
        initial_public = {"biomass": initial["biomass"].copy()}
    custom_cfg = replace(cfg, initial_condition="custom")

    natural = run_vegetation(
        reach,
        reach.Qnat,
        mode,
        custom_cfg,
        flow_name="natural",
        initial_state=initial_public,
        forcing=natural_forcing,
        habitat_mask=shared_habitat,
    )
    altered = run_vegetation(
        reach,
        scenario.Qrel,
        mode,
        custom_cfg,
        flow_name=scenario.name,
        initial_state=initial_public,
        habitat_mask=shared_habitat,
    )
    # Preserve the user-facing selected template in labels and summaries.
    natural.initial_condition = cfg.initial_condition
    altered.initial_condition = cfg.initial_condition
    comparison = VegetationComparison(natural, altered, scenario.name)
    if not hasattr(reach, "vegetation_comparisons"):
        reach.vegetation_comparisons = {}
    reach.vegetation_comparisons[(scenario.name, mode, cfg.initial_condition)] = comparison
    return comparison


@dataclass
class VegetationAnalysis:
    """Collection returned by :func:`run_full_vegetation_analysis`."""

    natural: dict[str, dict[str, VegetationResult]]
    comparisons: dict[str, dict[str, dict[str, VegetationComparison]]]
    initial_conditions: tuple[str, ...]
    scenario_names: tuple[str, ...]

    def summary(self) -> pd.DataFrame:
        """Return one row for every natural or altered model run."""
        rows: list[dict[str, Any]] = []
        for initial, mode_results in self.natural.items():
            for mode, result in mode_results.items():
                row = result.summary()
                row.update({"scenario": "natural", "selected_initial_state": initial})
                rows.append(row)
        for scenario_name, by_initial in self.comparisons.items():
            for initial, by_mode in by_initial.items():
                for mode, comparison in by_mode.items():
                    row = comparison.altered.summary()
                    row.update(
                        {
                            "scenario": scenario_name,
                            "selected_initial_state": initial,
                            "mode": mode,
                        }
                    )
                    rows.append(row)
        return pd.DataFrame(rows)


def run_full_vegetation_analysis(
    reach: Any,
    scenarios: list[Any] | tuple[Any, ...] | None = None,
    *,
    initial_conditions: tuple[str, ...] = ("bare",),
    modes: tuple[str, ...] = ("recruitment", "camporeale"),
    config: VegetationConfig | None = None,
) -> VegetationAnalysis:
    """
    Runs the complete vegetation workflow for a Reach.
    The function runs both ecological models under natural flow and, when
    scenarios are supplied, performs paired natural-versus-released-flow
    comparisons.  ``initial_conditions`` can contain any selection of
    ``"bare"``, ``"young"`` and ``"mature"``.
    """
    base = (config or VegetationConfig()).validated()
    selected_scenarios = tuple(reach.scenarios if scenarios is None else scenarios)
    selected_modes = tuple(mode.lower() for mode in modes)
    if any(mode not in {"recruitment", "camporeale"} for mode in selected_modes):
        raise ValueError("modes can only contain 'recruitment' and 'camporeale'")
    selected_initial = tuple(value.lower() for value in initial_conditions)
    if any(value not in {"bare", "young", "mature"} for value in selected_initial):
        raise ValueError("initial_conditions can only contain bare, young and mature")

    natural: dict[str, dict[str, VegetationResult]] = {}
    comparisons: dict[str, dict[str, dict[str, VegetationComparison]]] = {
        scenario.name: {} for scenario in selected_scenarios
    }

    for initial in selected_initial:
        cfg = replace(base, initial_condition=initial)
        natural[initial] = {}
        for mode in selected_modes:
            natural[initial][mode] = run_reach_vegetation(
                reach, mode=mode, config=cfg
            )
        for scenario in selected_scenarios:
            comparisons[scenario.name][initial] = {}
            for mode in selected_modes:
                comparisons[scenario.name][initial][mode] = compare_vegetation(
                    reach,
                    scenario,
                    mode=mode,
                    config=cfg,
                )

    analysis = VegetationAnalysis(
        natural=natural,
        comparisons=comparisons,
        initial_conditions=selected_initial,
        scenario_names=tuple(scenario.name for scenario in selected_scenarios),
    )
    reach.vegetation_analysis = analysis
    return analysis
