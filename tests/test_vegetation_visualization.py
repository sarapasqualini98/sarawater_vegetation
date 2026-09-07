import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sarawater.reach import Reach
from sarawater.scenarios import ConstScenario
from sarawater.vegetation import (
    VegetationConfig,
    compare_vegetation,
    run_full_vegetation_analysis,
    run_reach_vegetation,
)
from sarawater.visualization import VegetationPlotter, save_vegetation_analysis


def _reach(hours=24 * 35):
    dates = pd.date_range("2023-04-01", periods=hours, freq="h")
    t = np.arange(hours, dtype=float)
    q = 12.0 + 4.0 * np.sin(2 * np.pi * t / (24 * 8))
    q += 18.0 * np.exp(-0.5 * ((t - 24 * 10) / 14.0) ** 2)
    reach = Reach("Plot reach", dates.to_pydatetime().tolist(), q, 4.0)
    y = np.linspace(0, 30, 31)
    z = 1.5 + 0.006 * (y - 15) ** 2
    reach.add_cross_section_geometry(
        0.002, 30.0, section=pd.DataFrame({"y [m]": y, "z [m]": z})
    )
    reach.add_grain_size_distribution(30.0)
    scenario = ConstScenario("Qrel", "Reduced release", reach, [7.0] * 12)
    reach.add_scenario(scenario)
    scenario.compute_Qrel()
    return reach, scenario


def _config():
    return VegetationConfig(
        initial_condition="young",
        woo_days=5.0,
        rating_curve_points=14,
        enable_shear_mortality=False,
        enable_shields_mortality=False,
        enable_drought_mortality=False,
    )


def test_candidate_establishment_plot_renders_all_three_panels():
    reach, _ = _reach()
    result = run_reach_vegetation(reach, "recruitment", _config())
    fig = VegetationPlotter().plot_candidate_establishment_dynamics(result)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_qnat_qrel_final_cross_section_plots_render_for_both_models():
    reach, scenario = _reach()
    cfg = _config()
    recruitment = compare_vegetation(reach, scenario, "recruitment", cfg)
    camporeale = compare_vegetation(reach, scenario, "camporeale", cfg)
    plotter = VegetationPlotter()
    fig_r = plotter.plot_final_cross_section_comparison(recruitment)
    fig_c = plotter.plot_final_cross_section_comparison(camporeale)
    assert len(fig_r.axes) >= 3
    assert len(fig_c.axes) >= 3
    plt.close(fig_r)
    plt.close(fig_c)


def test_automatic_export_includes_candidate_and_qnat_qrel_figures(tmp_path):
    reach, scenario = _reach()
    analysis = run_full_vegetation_analysis(
        reach,
        scenarios=[scenario],
        initial_conditions=("young",),
        modes=("recruitment", "camporeale"),
        config=_config(),
    )
    paths = save_vegetation_analysis(analysis, tmp_path)
    names = {path.name for path in paths}
    assert any(name.endswith("_candidate_dynamics.png") for name in names)
    assert any("recruitment_final_cross_section_comparison" in name for name in names)
    assert any("camporeale_final_cross_section_comparison" in name for name in names)
    assert all(path.exists() for path in paths)
