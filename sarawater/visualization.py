"""
This module provides plotting functionality for comparing scenarios in a reach.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Optional, Union
from datetime import datetime
from pathlib import Path
from typing import Any
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

from sarawater.vegetation import VegetationComparison, VegetationResult
from sarawater.reach import Reach
from sarawater.scenarios import Scenario


class ReachPlotter:
    """A class for plotting and comparing scenarios in a reach."""

    def __init__(
        self,
        reach: Reach,
        output_dir: Optional[str] = "outputs",
        scenario_colors: List[str] = [
            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ],
    ):
        """
        Initialize a ReachPlotter instance.

        Parameters
        ----------
        reach : Reach
            The reach object containing scenarios to plot
        output_dir : str or None, optional
            Directory where to save the plots. By default "outputs". Set to None to prevent the directory from being created. Note that plotting methods need to be called with save=True to save the plots.
        scenario_colors : list of str, optional
            List of colors to use for each scenario in the plots. Default is a set of distinct tab colors.
        """
        self.reach = reach
        self.scenario_colors = scenario_colors
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def _ensure_iha_dir(self) -> str:
        """Create IHA subfolder if it doesn't exist (for multi-file methods)."""
        iha_dir = os.path.join(self.output_dir, "IHA_plots")
        os.makedirs(iha_dir, exist_ok=True)
        return iha_dir

    def plot_scenarios_discharge(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        log_scale: bool = True,
        save: bool = False,
        plot_Qnat: bool = True,
    ) -> None:
        """
        Plot discharge comparison between scenarios.

        Parameters
        ----------
        start_date : str or datetime, optional
            Start date for the plot range
        end_date : str or datetime, optional
            End date for the plot range
        log_scale : bool, default=True
            Whether to use log scale for y-axis
        save : bool, default=True
            Whether to save the plot to file
        plot_Qnat : bool, default=True
            Whether to plot the natural flow (Qnat)
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if start_date and end_date:
            mask = [(dt >= start_date) and (dt <= end_date) for dt in self.reach.dates]
        else:
            start_date = self.reach.dates[0]
            end_date = self.reach.dates[-1]
            mask = [True] * len(self.reach.dates)

        plt.figure()
        for color, scenario in zip(self.scenario_colors, self.reach.scenarios):
            plt.plot(
                np.array(self.reach.dates)[mask],
                scenario.Qrel[mask],
                color=color,
                label=scenario.name,
            )

        if plot_Qnat:
            plt.plot(
                np.array(self.reach.dates)[mask],
                self.reach.Qnat[mask],
                color="tab:blue",
                label="Natural",
            )

        if log_scale:
            plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - Discharge Comparison")
        plt.xlabel("Date")
        plt.ylabel("Discharge (m³/s)")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "discharge_comparison.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_iari_groups(
        self, save: bool = False, ylims: list = [None, None, None, None, None]
    ) -> None:
        """
        Plot IARI values comparison for each group.

        Parameters
        ----------
        ylims : list, default=None for all groups
            Y-axis limit for each group. If None, the limit is set automatically.
        save : bool, default=True
            Whether to save the plots to files
        """
        min_year = self.reach.dates[0].year
        years = range(
            min_year, min_year + len(self.reach.IHA_nat["Group1"]["mean_january"])
        )
        groups = self.reach.scenarios[0].IARI["groups"].keys()

        for g_idx, group in enumerate(groups):
            plt.figure()

            for i, scenario in enumerate(self.reach.scenarios):
                plt.plot(
                    years,
                    scenario.IARI["groups"][group],
                    color=self.scenario_colors[i],
                    label=scenario.name,
                )

            plt.title(f"{self.reach.name} - IARI Values Comparison - {group}")
            plt.xlabel("Year")
            plt.ylabel("IARI Value")
            plt.xlim(min(years), max(years))
            plt.xticks(years, years, rotation=45)
            plt.grid(True)
            plt.legend()

            # Set y-axis limits if provided
            if ylims[g_idx] is not None:
                plt.ylim(ylims[g_idx])
            else:
                plt.ylim(bottom=0)

            if save:
                plt.savefig(
                    os.path.join(self.output_dir, f"{group}_IARI_Comparison.png"),
                    bbox_inches="tight",
                )
        return plt.gca()

    def plot_iha_parameters(self, save: bool = False) -> None:
        """
        Plot IHA parameter comparisons for all parameters.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plots to files
        """
        min_year = self.reach.dates[0].year
        num_years = len(self.reach.IHA_nat["Group1"]["mean_january"])
        years_nat = range(min_year, min_year + num_years)
        years_alt = range(years_nat[-1] + 1, years_nat[-1] + 1 + num_years)
        years = list(years_nat) + list(years_alt)

        for IHA_group_name, IHA_group in self.reach.scenarios[0].IHA.items():
            for indicator in IHA_group:
                natural_values = self.reach.IHA_nat[IHA_group_name][indicator]

                plt.figure()
                plt.plot(
                    years_nat,
                    natural_values,
                    color="tab:blue",
                    label="Natural",
                )

                # Calculate and plot percentiles for natural flow.
                # The 25th and 75th percentiles define a band that represents the
                # natural inter-annual variability of this indicator, which is used in the computation of the IARI index and serves as reference range when visually comparing scenario results.
                p25 = np.percentile(natural_values, 25)
                p75 = np.percentile(natural_values, 75)

                # Plot percentile lines across the entire time range
                plt.axhline(
                    y=p25,
                    color="tab:blue",
                    linestyle="--",
                )
                plt.axhline(y=p75, color="tab:blue", linestyle="--")

                for j, scenario in enumerate(self.reach.scenarios):
                    plt.plot(
                        years_alt,
                        scenario.IHA[IHA_group_name][indicator],
                        label=scenario.name,
                        color=self.scenario_colors[j],
                    )

                plt.title(f"{self.reach.name} - IHA - {indicator}")
                plt.xlabel("Year")
                plt.ylabel("IHA Value")
                plt.xlim(min(years), max(years))
                plt.xticks(years, years, rotation=45)
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(
                        os.path.join(
                            self._ensure_iha_dir(), f"{indicator}_IHA_Comparison.png"
                        ),
                        bbox_inches="tight",
                    )
        return plt.gca()

    def plot_iari_summary(self, save: bool = False) -> None:
        """
        Create a summary bar plot of IARI indices for all scenarios.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        groups = ["Group1", "Group2", "Group3", "Group4", "Group5"]
        n_scenarios = len(self.reach.scenarios)

        # Calculate mean IARI values for each group and scenario
        means = {
            scenario.name: [np.mean(scenario.IARI["groups"][group]) for group in groups]
            for scenario in self.reach.scenarios
        }
        for scenario in self.reach.scenarios:
            means[scenario.name].append(np.mean(scenario.IARI["aggregated"]))

        # Create grouped bar plot
        width = 0.8 / n_scenarios
        fig, ax = plt.subplots()

        bar_labels = groups + ["Aggregated"]
        for i, (scenario_name, values) in enumerate(means.items()):
            x = np.arange(len(bar_labels)) + (i - n_scenarios / 2 + 0.5) * width
            ax.bar(x, values, width, label=scenario_name, color=self.scenario_colors[i])

        ax.set_ylabel("Mean IARI Value")
        ax.set_title(f"{self.reach.name} IARI Summary")
        ax.set_xticks(np.arange(len(bar_labels)))
        ax.set_xticklabels(bar_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "IARI_Summary.png"), bbox_inches="tight"
            )
        return plt.gca()

    def plot_nIHA_summary(self, save: bool = False) -> None:
        """
        Create a summary bar plot of normalized IHA indices for all scenarios.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        groups = ["Group1", "Group2", "Group3", "Group4", "Group5"]
        n_scenarios = len(self.reach.scenarios)

        # Calculate mean nIHA values for each group and scenario
        means = {
            scenario.name: [
                np.mean(scenario.normalized_IHA["groups"][group]) for group in groups
            ]
            for scenario in self.reach.scenarios
        }
        for scenario in self.reach.scenarios:
            means[scenario.name].append(np.mean(scenario.normalized_IHA["aggregated"]))

        # Create grouped bar plot
        width = 0.8 / n_scenarios
        fig, ax = plt.subplots()

        bar_labels = groups + ["Aggregated"]
        for i, (scenario_name, values) in enumerate(means.items()):
            x = np.arange(len(bar_labels)) + (i - n_scenarios / 2 + 0.5) * width
            ax.bar(x, values, width, label=scenario_name, color=self.scenario_colors[i])

        ax.set_ylabel("Mean nIHA Value")
        ax.set_title(f"{self.reach.name} nIHA Summary")
        ax.set_xticks(np.arange(len(bar_labels)))
        ax.set_xticklabels(bar_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "nIHA_Summary.png"), bbox_inches="tight"
            )
        return plt.gca()

    def plot_iha_boxplots(self, save: bool = False) -> None:
        """
        Create boxplot comparisons for IHA parameters across scenarios.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plots to files
        """
        for IHA_group_name, IHA_group in self.reach.scenarios[0].IHA.items():
            for indicator in IHA_group:
                plt.figure()

                data = [self.reach.IHA_nat[IHA_group_name][indicator]]
                labels = ["Natural"]

                for scenario in self.reach.scenarios:
                    data.append(scenario.IHA[IHA_group_name][indicator])
                    labels.append(scenario.name)

                plt.boxplot(data, labels=labels)
                plt.title(f"{self.reach.name} - IHA Distribution - {indicator}")
                plt.ylabel("Value")
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

                if save:
                    plt.savefig(
                        os.path.join(
                            self._ensure_iha_dir(), f"{indicator}_boxplot.png"
                        ),
                        bbox_inches="tight",
                    )
        return plt.gca()

    def plot_relative_deviations(self, save: bool = False) -> None:
        """
        Plot relative deviations of IHA parameters from natural flow.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plots to files
        """
        for IHA_group_name, IHA_group in self.reach.scenarios[0].IHA.items():
            for indicator in IHA_group:
                plt.figure()
                natural_values = self.reach.IHA_nat[IHA_group_name][indicator]

                min_year = self.reach.dates[0].year
                years = range(min_year, min_year + len(natural_values))

                for scenario in self.reach.scenarios:
                    scenario_values = scenario.IHA[IHA_group_name][indicator]
                    relative_dev = (
                        (scenario_values - natural_values) / natural_values * 100
                    )

                    plt.plot(
                        years,
                        relative_dev,
                        label=scenario.name,
                        color=self.scenario_colors[
                            self.reach.scenarios.index(scenario)
                        ],
                    )

                plt.title(
                    f"{self.reach.name} - Relative Deviation from Natural - {indicator}"
                )
                plt.xlabel("Year")
                plt.ylabel("Relative Deviation (%)")
                plt.xlim(min(years), max(years))
                plt.xticks(years, years, rotation=45)
                plt.grid(True)
                plt.legend()

                if save:
                    plt.savefig(
                        os.path.join(
                            self._ensure_iha_dir(),
                            f"{indicator}_relative_deviation.png",
                        ),
                        bbox_inches="tight",
                    )
        return plt.gca()

    def plot_cases_duration(self, save: bool = False) -> None:
        """
        Create a bar plot showing the duration percentage of each flow case for all scenarios.

        Flow cases are:
        - Case 1: Q ≤ Qreq (Natural flow is less than or equal to minimum release)
        - Case 2: Qreq < Q < Qreq + Qabs_max (Natural flow is between minimum release and maximum abstraction)
        - Case 3: Q ≥ Qreq + Qabs_max (Natural flow exceeds maximum abstraction capacity)

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        # Ensure all scenarios have computed their cases_duration
        for scenario in self.reach.scenarios:
            if (
                not hasattr(scenario, "cases_duration")
                or scenario.cases_duration is None
            ):
                scenario.compute_Qrel()  # This will compute cases_duration as well

        n_scenarios = len(self.reach.scenarios)
        case_labels = [
            "Case 1\n(Qnat ≤ Qreq)",
            "Case 2\n(Qreq < Qnat < Qreq+Qabs_max)",
            "Case 3\n(Qnat ≥ Qreq+Qabs_max)",
        ]

        # Create a figure with appropriate size
        plt.figure()

        # Set up bar positions
        x = np.arange(len(case_labels))
        width = 0.8 / n_scenarios  # Width of bars

        # Plot bars for each scenario
        for i, scenario in enumerate(self.reach.scenarios):
            pos = x + (i - n_scenarios / 2 + 0.5) * width
            plt.bar(
                pos,
                [d * 100 for d in scenario.cases_duration],
                width,
                label=scenario.name,
                color=self.scenario_colors[i],
            )

        plt.ylabel("Duration (%)")
        plt.title(f"{self.reach.name} - Flow Cases Duration by Scenario")
        plt.xticks(x, case_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add value labels on top of each bar
        for i, scenario in enumerate(self.reach.scenarios):
            pos = x + (i - n_scenarios / 2 + 0.5) * width
            for j, value in enumerate(scenario.cases_duration):
                plt.text(
                    pos[j],
                    value * 100,
                    f"{value*100:.0f}%",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "cases_duration.png"), bbox_inches="tight"
            )
        return plt.gca()

    def plot_cases_duration_month(self, month, save: bool = False) -> None:
        """
        Create a bar plot showing the duration percentage of each flow case for all scenarios for a specific month.

        Parameters
        ----------
        month : int or str
            Month to plot (1-12 or month name, e.g., 'Jan', 'January')
        save : bool, default=False
            Whether to save the plot to file
        """
        # Convert month input to integer (1-12)
        if isinstance(month, str):
            month_strs = [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
            month_lower = month.strip().lower()[:3]
            if month_lower in month_strs:
                month_num = month_strs.index(month_lower) + 1
            else:
                raise ValueError(f"Invalid month string: {month}")
        elif isinstance(month, int) and 1 <= month <= 12:
            month_num = month
        else:
            raise ValueError(
                "month must be an integer (1-12) or a valid month name string"
            )

        n_scenarios = len(self.reach.scenarios)
        case_labels = [
            "Case 1\n(Q ≤ Qreq)",
            "Case 2\n(Qreq < Q < Qreq+Qabs_max)",
            "Case 3\n(Q ≥ Qreq+Qabs_max)",
        ]

        plt.figure()
        x = np.arange(len(case_labels))
        width = 0.8 / n_scenarios

        for i, scenario in enumerate(self.reach.scenarios):
            # Use the new method from Scenario for robust calculation
            durations = scenario.cases_duration_for_month(month_num)
            pos = x + (i - n_scenarios / 2 + 0.5) * width
            plt.bar(
                pos,
                [d * 100 for d in durations],
                width,
                label=scenario.name,
                color=self.scenario_colors[i],
            )
            # Add value labels
            for j, value in enumerate(durations):
                plt.text(
                    pos[j],
                    value * 100,
                    f"{value*100:.0f}%",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

        plt.ylabel("Duration (%)")
        plt.title(
            f"{self.reach.name} - Flow Cases Duration by Scenario - {month if isinstance(month, str) else month_num}"
        )
        plt.xticks(x, case_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(
                    self.output_dir,
                    f"cases_duration_month_{month if isinstance(month, str) else month_num}.png",
                ),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_monthly_abstraction(self, save: bool = False) -> None:
        """
        Create a bar plot showing the average monthly abstracted volumes for each scenario.

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        # Ensure all scenarios have computed their abstracted volumes
        for scenario in self.reach.scenarios:
            if (
                not hasattr(scenario, "monthly_abs_volumes")
                or scenario.monthly_abs_volumes is None
            ):
                scenario.compute_natural_abstracted_volumes()

        # Set up plot
        plt.figure()
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        x = np.arange(len(months))
        width = 0.8 / (len(self.reach.scenarios) + 1)  # +1 for natural flow

        # Plot natural flow volumes as reference
        plt.bar(
            x,
            self.reach.scenarios[0].monthly_nat_volumes / 1e6,
            width,
            label="Natural",
            color="tab:blue",
            alpha=0.3,
        )

        # Plot abstracted volumes for each scenario
        for i, scenario in enumerate(self.reach.scenarios):
            pos = x + (i + 1) * width - 0.4
            plt.bar(
                pos,
                scenario.monthly_abs_volumes / 1e6,
                width,
                label=f"{scenario.name}",
                color=self.scenario_colors[i],
            )

        # Customize plot
        plt.xlabel("Month")
        plt.ylabel("Volume (million m³)")
        plt.title(f"{self.reach.name} - Monthly-averaged Abstracted Water Volumes")
        plt.xticks(x, months)
        plt.legend(title="Scenario")
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "monthly_abstraction.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_iari_vs_volume(self, save: bool = False) -> None:
        """
        Create a scatter plot showing the relationship between abstracted volumes and IARI indices.
        Each scenario is shown with error bars representing standard deviations.

        X-axis shows ecohydrological quality (1 - IARI)
        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        plt.figure()

        # For each scenario except natural flow
        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate IARI statistics
            iari_values = scenario.IARI["aggregated"]
            iari_median = np.median(iari_values)
            iari_std = np.std(iari_values)

            # Calculate volume statistics
            vol_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
            vol_median = np.median(vol_values)
            vol_std = np.std(vol_values)

            # Plot error bars and point
            plt.errorbar(
                1 - iari_median,
                vol_median,
                xerr=iari_std,
                yerr=vol_std,
                fmt="^",  # marker style
                color=self.scenario_colors[i],
                label=scenario.name,
                capsize=5,
                elinewidth=1,
                markersize=10,
            )

        plt.xlabel(r"Ecohydrological quality $(1 - IARI)$ [-]")
        plt.ylabel(r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - IARI vs Abstracted Volume")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "iari_vs_volume.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_hq_curves(
        self,
        save: bool = False,
        xlim: float = None,
        rule_min: float = None,
        rule_max: float = None,
        rule_name: str = "DMV",
    ) -> None:
        """
        Plot all HQ curves of the reach.

        Parameters


        ----------
        save : bool, default=True
            Whether to save the plot to file
        """

        plt.figure(figsize=(10, 6))

        for species in self.reach.get_list_available_HQ_curves():
            curve = self.reach.get_HQ_curve(curve_name=species)
            plt.plot(curve["DIS"], curve[species], label=f"{species}")

        # LABELS AND TITLE
        plt.xlabel(r"Q $[\mathrm{m}^3/\mathrm{s}]$")
        plt.ylabel(r"Available area $[\mathrm{m}^2]$")
        plt.title("Habitat-Discharge (HQ) curves")

        if xlim:
            plt.xlim(0, xlim)

        if rule_min and rule_max:
            plt.axvline(
                x=rule_min,
                color="tab:gray",
                linestyle="--",
                label=f"{rule_name} range: {rule_min}-{rule_max} m³/s",
            )
            plt.axvline(x=rule_max, color="tab:gray", linestyle="--")

        # Show the plot
        plt.grid(True)
        plt.legend()

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "hq_curves.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_habitat_timeseries(
        self,
        species: str,
        save: bool = False,
        start_year: int = None,
        end_year: int = None,
    ) -> None:
        """
        Plot habitat time series for specific species and scenario

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        species : str
            Species to plot
        scenario : Scenario
            Scenario to plot
        start_year : int
            Start year for the plot range
        end_year : int
            End year for the plot range
        """

        plt.figure()
        for scenario in self.reach.scenarios:
            plt.plot(
                self.reach.dates,
                scenario.IH[species]["H_alt"],
                label=f"{scenario.name} - {species}",
                # color="tab:orange",
            )

        plt.plot(
            self.reach.dates,
            scenario.IH[species]["H_ref"],
            label=f"Reference Q",
            # color="tab:blue",
        )

        plt.xlim(
            datetime(start_year if start_year else self.reach.dates[0].year, 1, 1),
            datetime(end_year if end_year else self.reach.dates[-1].year, 12, 31),
        )

        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Available Habitat Area [%]")
        plt.title(f"{self.reach.name} - Habitat Time Series - {species}")
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(
                os.path.join(self.output_dir, f"habitat_timeseries_{species}.png"),
                bbox_inches="tight",
            )

    def plot_ucut_curves(
        self,
        species: str,
        save: bool = False,
    ) -> None:
        """
        Plot ucut curves for specific species and all scenarios

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        species : str
            Species to plot
        scenario : Scenario
            Scenario to plot
        """

        plt.figure()
        for scenario in self.reach.scenarios:
            plt.plot(
                scenario.IH[species]["UCUT_cum_alt"],
                scenario.IH[species]["UCUT_events_alt"],
                label=f"{scenario.name} - {species}",
                # color="tab:orange",
            )

        plt.plot(
            scenario.IH[species]["UCUT_cum_ref"],
            scenario.IH[species]["UCUT_events_ref"],
            label=f"Reference Q",
            # color="tab:blue",
        )

        plt.xticks(rotation=45)
        plt.xlabel("Cumulative continuous duration [%]")
        plt.ylabel("Continuous days below threshold [days]")
        plt.title(f"{self.reach.name} - UCUT - {species}")
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(
                os.path.join(
                    self.output_dir, f"habitat_timeseries_{species}_{scenario.name}.png"
                ),
                bbox_inches="tight",
            )

    def plot_ih_vs_volume(self, save: bool = False) -> None:
        """
        Create a scatter plot showing the relationship between abstracted volumes and IH index for a selected species.
        Each scenario is shown with error bars representing standard deviations of volumes.

        X-axis shows Habitat Index (IH)
        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=False
            Whether to save the plot to file
        """
        plt.figure()

        # Create color mapping for species
        all_species = set()
        for scenario in self.reach.scenarios:
            all_species.update(scenario.IH.keys())

        species_colors = plt.cm.tab10(np.linspace(0, 1, len(all_species)))
        species_color_map = {
            species: species_colors[i] for i, species in enumerate(sorted(all_species))
        }
        species_plotted = set()  # Track which species have been added to legend

        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate IH statistics
            ih_values = []
            for species in scenario.IH.keys():
                ih_values.append(scenario.IH[species]["IH"])

            # Calculate volume statistics
            vol_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
            vol_median = np.median(vol_values)
            vol_std = np.std(vol_values)

            # Plot error bars and point
            plt.errorbar(
                min(ih_values),
                vol_median,
                yerr=vol_std,
                xerr=np.array([[0], [max(ih_values)] - min(ih_values)]),
                fmt="^",  # marker style
                color=self.scenario_colors[i],
                label=scenario.name,
                capsize=5,
                elinewidth=1,
                markersize=10,
            )

            # plot a point for each species IH
            for species in scenario.IH.keys():
                plt.plot(
                    scenario.IH[species]["IH"],
                    vol_median,
                    "o",
                    color=species_color_map[species],
                    alpha=0.7,
                    markersize=8,
                    label=species if species not in species_plotted else None,
                )
                species_plotted.add(species)

        plt.xlabel(r"Habitat Index $(IH)$ [-]")
        plt.ylabel(r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - IH vs Abstracted Volume")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "ih_vs_volume.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_nIHA_vs_volume(self, save: bool = False) -> None:
        """
        Create a scatter plot showing the relationship between abstracted volumes and nIHA indexes.
        Each scenario is shown with error bars representing standard deviations.

        X-axis shows ecohydrological quality (-nIHA)

        Y-axis shows normalized abstracted volume (Vturb/Vnat)

        Parameters
        ----------
        save : bool, default=True
            Whether to save the plot to file
        """
        plt.figure()

        # For each scenario except natural flow
        for i, scenario in enumerate(self.reach.scenarios):
            # Calculate nIHA statistics
            nIHA_values = scenario.normalized_IHA["aggregated"]
            nIHA_median = np.median(nIHA_values)
            nIHA_std = np.std(nIHA_values)

            # Calculate volume statistics
            vol_values = scenario.yearly_abs_volumes / scenario.yearly_nat_volumes
            vol_median = np.median(vol_values)
            vol_std = np.std(vol_values)

            # Plot error bars and point
            plt.errorbar(
                -nIHA_median,
                vol_median,
                xerr=nIHA_std,
                yerr=vol_std,
                fmt="^",  # marker style
                color=self.scenario_colors[i],
                label=scenario.name,
                capsize=5,
                elinewidth=1,
                markersize=10,
            )

        plt.xlabel(r"Ecohydrological quality $(-nIHA)$ [-]")
        plt.ylabel(r"Normalized abstracted volume $V_{der}/V_{nat}$ [-]")
        plt.grid(True)
        plt.legend()
        plt.title(f"{self.reach.name} - nIHA vs Abstracted Volume")

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "niha_vs_volume.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_sediment_load_total(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        log_scale: bool = True,
        save: bool = False,
    ) -> None:
        """
        Plot total sediment load (Qs_total) over time for all scenarios.

        Parameters
        ----------
        start_date : str or datetime, optional
            Start date for the plot
        end_date : str or datetime, optional
            End date for the plot
        log_scale : bool, default=True
            Use log scale for y-axis
        save : bool, default=False
            Whether to save the plot
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if start_date and end_date:
            mask = [(dt >= start_date) and (dt <= end_date) for dt in self.reach.dates]
        else:
            mask = [True] * len(self.reach.dates)

        plt.figure()
        for i, scenario in enumerate(self.reach.scenarios):
            if not hasattr(scenario, "sediment_load"):
                continue
            plt.plot(
                np.array(self.reach.dates)[mask],
                scenario.sediment_load["Qs_total"].values[mask],
                label=scenario.name,
                color=self.scenario_colors[i],
            )

        if log_scale:
            plt.yscale("log")
        plt.xlabel("Date")
        plt.ylabel("Total Sediment Load (kg/s)")
        plt.title(f"{self.reach.name} - Total Sediment Load")
        plt.grid(True)
        plt.legend()

        if save:
            plt.savefig(
                os.path.join(self.output_dir, "sediment_load_total.png"),
                bbox_inches="tight",
            )
        return plt.gca()

    def plot_sediment_load_fractions(
        self,
        scenario_index: int = 0,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        save: bool = False,
    ) -> None:
        """
        Plot sediment load fractions per phi class as stacked area for a scenario.

        Parameters
        ----------
        scenario_index : int, default=0
            Index of the scenario to plot
        start_date : str or datetime, optional
            Start date for the plot
        end_date : str or datetime, optional
            End date for the plot
        save : bool, default=False
            Whether to save the plot
        """
        scenario = self.reach.scenarios[scenario_index]
        if not hasattr(scenario, "sediment_load"):
            raise ValueError(f"Scenario {scenario.name} has no sediment_load data.")

        df = scenario.sediment_load.copy()
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        if start_date and end_date:
            mask = (df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)
            df = df.loc[mask]

        phi_cols = [c for c in df.columns if c.startswith("Qs_phi_")]
        plt.figure(figsize=(12, 6))
        plt.stackplot(df["Datetime"], df[phi_cols].T, labels=phi_cols, alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("Sediment Load per Phi Class (kg/s)")
        plt.title(f"{self.reach.name} - Sediment Load Fractions - {scenario.name}")
        plt.legend(loc="upper right", ncol=2)
        plt.grid(True)

        if save:
            plt.savefig(
                os.path.join(
                    self.output_dir, f"sediment_load_fractions_{scenario.name}.png"
                ),
                bbox_inches="tight",
            )
        return plt.gca()

    #Riparian Vegetation Analysis visualization
    NAVY = "#0B2454"
BLUE = "#2E86AB"
WATER_LIGHT = "#DCEEF8"
GREEN = "#2F855A"
GREEN_LIGHT = "#DDEEDC"
AMBER = "#D38B2C"
RED = "#C94C4C"
PURPLE = "#7655A6"
CHARCOAL = "#263238"
MUTED = "#657180"
GRID = "#D9DEE5"
BED = "#4A4A4A"


def _axes_style(ax: plt.Axes, *, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def _format_dates(ax: plt.Axes, dates: pd.DatetimeIndex) -> None:
    span = dates[-1] - dates[0]
    if span.days > 730:
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter("%Y")
    elif span.days > 180:
        locator = mdates.MonthLocator(interval=2)
        formatter = mdates.DateFormatter("%b %Y")
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _resolve_time(result: VegetationResult, when: Any = None) -> tuple[int, pd.Timestamp]:
    if when is None:
        index = result.dates.size - 1
    elif isinstance(when, (int, np.integer)):
        index = int(when)
        if index < 0:
            index += result.dates.size
        if index < 0 or index >= result.dates.size:
            raise IndexError("time index is outside the simulation")
    else:
        timestamp = pd.Timestamp(when)
        index = int(np.argmin(np.abs(result.dates - timestamp)))
    return index, pd.Timestamp(result.dates[index])


def _bed(ax: plt.Axes, result: VegetationResult) -> None:
    ax.plot(result.y, result.z, color=BED, linewidth=1.8, zorder=4)
    bottom = float(np.nanmin(result.z) - 0.08 * max(np.ptp(result.z), 1.0))
    ax.fill_between(result.y, bottom, result.z, color="#EAE7E1", alpha=0.85)


def _save(fig: plt.Figure, save: str | Path | None) -> None:
    if save is not None:
        destination = Path(save)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")


class VegetationPlotter:
    """Plot SARAwater vegetation results with a consistent scientific style."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def _path(self, filename: str, save: bool | str | Path) -> Path | None:
        if isinstance(save, (str, Path)):
            return Path(save)
        if save and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            return self.output_dir / filename
        return None

    def plot_recruitment_state(
        self,
        result: VegetationResult,
        when: Any = None,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Plot binary plant presence directly on the reach cross-section."""
        if result.mode != "recruitment":
            raise ValueError("plot_recruitment_state requires a recruitment result")
        index, timestamp = _resolve_time(result, when)
        alive = result.presence[index]

        fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
        _bed(ax, result)
        ax.scatter(
            result.y[result.habitat_mask],
            result.z[result.habitat_mask],
            s=11,
            color="#B8C0C8",
            label="Potential riparian habitat",
            zorder=5,
        )
        ax.scatter(
            result.y[alive],
            result.z[alive],
            s=34,
            color=GREEN,
            edgecolor="white",
            linewidth=0.45,
            label="Vegetation present",
            zorder=7,
        )
        water = result.forcing.water_level[index]
        ax.axhline(water, color=BLUE, linewidth=1.3, linestyle="--", label="Water level")
        ax.fill_between(
            result.y,
            result.z,
            water,
            where=water > result.z,
            color=WATER_LIGHT,
            alpha=0.58,
            zorder=2,
        )
        ax.set_title(f"Recruitment model: vegetation presence — {timestamp:%Y-%m-%d}")
        ax.set_xlabel("Cross-section coordinate [m]")
        ax.set_ylabel("Elevation [m]")
        ax.legend(loc="upper left", frameon=False, ncol=3)
        _axes_style(ax, grid="y")
        _save(fig, self._path("recruitment_state.png", save))
        return fig

    def plot_recruitment_age(
        self,
        result: VegetationResult,
        when: Any = None,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Plot age only for cells where recruitment vegetation is present."""
        if result.mode != "recruitment":
            raise ValueError("plot_recruitment_age requires a recruitment result")
        index, timestamp = _resolve_time(result, when)
        alive = result.presence[index]
        ages = result.age_days[index]

        fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
        _bed(ax, result)
        if alive.any():
            vmax = max(float(np.nanmax(ages[alive])), 1.0)
            points = ax.scatter(
                result.y[alive],
                result.z[alive],
                c=ages[alive],
                cmap="YlGn",
                norm=Normalize(0.0, vmax),
                s=42,
                edgecolor=CHARCOAL,
                linewidth=0.35,
                zorder=7,
            )
            cbar = fig.colorbar(points, ax=ax, pad=0.015)
            cbar.set_label("Plant age [days]")
        else:
            ax.text(
                0.5,
                0.88,
                "No living recruitment vegetation",
                transform=ax.transAxes,
                ha="center",
                color=MUTED,
            )
        ax.set_title(f"Recruitment model: plant age — {timestamp:%Y-%m-%d}")
        ax.set_xlabel("Cross-section coordinate [m]")
        ax.set_ylabel("Elevation [m]")
        _axes_style(ax, grid="y")
        _save(fig, self._path("recruitment_age.png", save))
        return fig

    def plot_annual_recruitment_opportunity(
        self,
        result: VegetationResult,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Plot annual EBE1 opportunity bands and end-of-season vegetation.

        EBE1 is a hydrological establishment opportunity band.  It indicates
        where a cell could complete the Recruitment Box plus Window of
        Opportunity criteria during that year; it is not a map of occupied
        vegetation at the September snapshot.
        """
        if result.mode != "recruitment":
            raise ValueError("annual recruitment opportunity requires recruitment mode")
        bands = result.metadata.get("recruitment_bands", {})
        years = sorted(int(year) for year in bands)
        if not years:
            raise ValueError("No annual recruitment bands are stored in the result")

        fig, axes = plt.subplots(
            len(years), 1, figsize=(12.2, 2.8 * len(years)),
            sharex=True, sharey=True, constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        relief = max(float(np.ptp(result.z)), 1.0)
        for ax, year in zip(axes, years):
            candidates = np.flatnonzero(
                (result.dates.year == year)
                & (
                    (result.dates.month < 9)
                    | ((result.dates.month == 9) & (result.dates.day <= 30))
                )
            )
            if not candidates.size:
                candidates = np.flatnonzero(result.dates.year == year)
            index = int(candidates[-1])
            alive = result.presence[index]
            band = bands.get(year, bands.get(str(year), {}))
            lo = float(band.get("ebe1_lower_m", np.nan))
            hi = float(band.get("ebe1_upper_m", np.nan))

            _bed(ax, result)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ax.axhspan(
                    lo, hi, color=GREEN_LIGHT, alpha=0.42,
                    label=f"{year} EBE1 hydrological opportunity band",
                )
            if alive.any():
                ax.vlines(
                    result.y[alive], result.z[alive],
                    result.z[alive] + 0.04 * relief,
                    color=GREEN, linewidth=0.9, alpha=0.85,
                )
                ax.scatter(
                    result.y[alive], result.z[alive] + 0.042 * relief,
                    s=25, color=GREEN, edgecolor="white", linewidth=0.35,
                    label=f"Living recruits: {int(alive.sum())}", zorder=7,
                )
            else:
                ax.text(
                    0.5, 0.58, "No living recruits at this snapshot",
                    transform=ax.transAxes, ha="center", color=MUTED,
                )
            ax.text(
                0.0, 1.035,
                f"Snapshot: {pd.Timestamp(result.dates[index]):%d %b %Y}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontweight="semibold", color=CHARCOAL, clip_on=False,
            )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    handles, labels, ncol=2, loc="lower right",
                    bbox_to_anchor=(1.0, 1.015), borderaxespad=0.0,
                    frameon=False, fontsize=8,
                )
            ax.set_ylabel("Elevation [m]")
            _axes_style(ax, grid="y")
        axes[-1].set_xlabel("Cross-section coordinate [m]")
        fig.suptitle("Annual recruitment opportunity and observed vegetation")
        _save(fig, self._path("annual_recruitment_opportunity.png", save))
        return fig

    def plot_candidate_establishment_dynamics(
        self,
        result: VegetationResult,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Show where candidate recruitment starts and whether it establishes."""
        if result.mode != "recruitment":
            raise ValueError("candidate dynamics require recruitment mode")
        if result.candidate is None or result.established is None:
            raise ValueError("candidate and established histories are unavailable")

        candidates = result.candidate.sum(axis=1)
        established = result.established.sum(axis=1)
        daily = result.diagnostics[["seedlings_created", "recruits"]].resample("1D").sum()
        ever_candidate = result.candidate.any(axis=0)
        ever_established = result.established.any(axis=0)
        failed_sites = ever_candidate & ~ever_established
        final_candidate = result.final_candidate

        fig, axes = plt.subplots(
            3, 1, figsize=(12.0, 9.0),
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.2]},
            constrained_layout=True,
        )
        counts, events, spatial = axes
        counts.plot(result.dates, candidates, color=AMBER, linewidth=1.5, label="Candidate seedlings")
        counts.plot(result.dates, established, color=GREEN, linewidth=1.6, label="Established vegetation")
        counts.set_ylabel("Living cells [count]")
        counts.set_title("Candidate-to-established transition")
        counts.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        _format_dates(counts, result.dates); _axes_style(counts, grid="y")

        events.bar(daily.index, daily["seedlings_created"], width=0.9, color=AMBER, alpha=0.75, label="New candidates")
        events.bar(daily.index, daily["recruits"], width=0.55, color=GREEN, alpha=0.9, label="Newly established")
        events.set_ylabel("Cell events per day")
        events.set_title(f"Recruitment events; establishment after {result.config.woo_days:g} consecutive chronological days")
        events.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        _format_dates(events, result.dates); _axes_style(events, grid="y")

        _bed(spatial, result)
        spatial.scatter(result.y[result.habitat_mask], result.z[result.habitat_mask], s=10, color="#C9D0D7", alpha=0.7, label="Potential habitat", zorder=4)
        if failed_sites.any():
            spatial.scatter(result.y[failed_sites], result.z[failed_sites], s=38, color=RED, edgecolor="white", linewidth=0.4, label="Candidate site never established", zorder=7)
        if ever_established.any():
            spatial.scatter(result.y[ever_established], result.z[ever_established], s=38, color=GREEN, edgecolor="white", linewidth=0.4, label="Established at least once", zorder=8)
        if final_candidate.any():
            spatial.scatter(result.y[final_candidate], result.z[final_candidate], s=58, facecolor="none", edgecolor=AMBER, linewidth=1.4, label="Still candidate at final time", zorder=9)
        spatial.set_xlabel("Cross-section coordinate [m]")
        spatial.set_ylabel("Elevation [m]")
        spatial.set_title("Spatial fate of candidate recruitment attempts")
        spatial.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        _axes_style(spatial, grid="y")
        fig.suptitle("Recruitment candidates and establishment outcomes")
        _save(fig, self._path("candidate_establishment_dynamics.png", save))
        return fig

    def plot_camporeale_biomass(
        self,
        result: VegetationResult,
        when: Any = None,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Plot continuous biomass without mixing biomass and elevation units."""
        if result.mode != "camporeale":
            raise ValueError("plot_camporeale_biomass requires a Camporeale result")
        index, timestamp = _resolve_time(result, when)
        biomass = result.state[index]

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11.5, 6.8),
            sharex=True,
            gridspec_kw={"height_ratios": [1.5, 1.0]},
            constrained_layout=True,
        )
        ax, profile = axes
        _bed(ax, result)
        points = ax.scatter(
            result.y[result.habitat_mask],
            result.z[result.habitat_mask],
            c=biomass[result.habitat_mask],
            cmap="YlGn",
            norm=Normalize(0.0, 1.0),
            s=36,
            edgecolor=CHARCOAL,
            linewidth=0.25,
            zorder=7,
        )
        water = result.forcing.water_level[index]
        ax.axhline(water, color=BLUE, linewidth=1.2, linestyle="--")
        ax.fill_between(
            result.y,
            result.z,
            water,
            where=water > result.z,
            color=WATER_LIGHT,
            alpha=0.55,
            zorder=2,
        )
        cbar = fig.colorbar(points, ax=ax, pad=0.015)
        cbar.set_label("Normalized biomass")
        ax.set_title(f"Camporeale model: biomass — {timestamp:%Y-%m-%d}")
        ax.set_ylabel("Elevation [m]")
        _axes_style(ax, grid="y")

        profile.fill_between(result.y, 0.0, biomass, color=GREEN_LIGHT, alpha=0.9)
        profile.plot(result.y, biomass, color=GREEN, linewidth=1.8)
        profile.set_ylim(0.0, 1.02)
        profile.set_xlabel("Cross-section coordinate [m]")
        profile.set_ylabel("Biomass [0–1]")
        _axes_style(profile, grid="y")
        _save(fig, self._path("camporeale_biomass.png", save))
        return fig

    def plot_mortality(
        self,
        result: VegetationResult,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Plot submergence exposure, cause-specific deaths and mortality locations."""
        if result.mode != "recruitment":
            raise ValueError("mortality causes are specific to recruitment mode")
        diagnostic = result.diagnostics
        dates = result.dates
        causes = result.mortality_cause
        # Four explicit process-based mortality pathways.
        totals = [
            (causes == 1).sum(axis=0),
            (causes == 2).sum(axis=0),
            (causes == 3).sum(axis=0),
            (causes == 4).sum(axis=0),
        ]
        fig, axes = plt.subplots(
            3, 1, figsize=(12.0, 9.0),
            gridspec_kw={"height_ratios": [0.9, 1.1, 1.0]},
            constrained_layout=True,
        )
        exposure, temporal, spatial = axes
        submerged = diagnostic.get("submerged_alive_cells", pd.Series(0, index=diagnostic.index))
        exposure.step(dates, submerged, where="post", color=BLUE, linewidth=1.35,
                      label="Living plants submerged")
        exposure.set_ylim(bottom=0)
        exposure.set_ylabel("Submerged cells")
        exposure.set_title("Recruitment exposure and mortality")
        exposure.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        _format_dates(exposure, dates); _axes_style(exposure, grid="y")

        mortality_daily = pd.DataFrame(
            {
                "Prolonged inundation": diagnostic["mortality_anoxia"],
                "Shear stress": diagnostic["mortality_shear"],
                "Shields / bed mobility": diagnostic["mortality_sediment"],
                "Drought": diagnostic.get("mortality_drought", pd.Series(0, index=diagnostic.index)),
            },
            index=diagnostic.index,
        ).resample("1D").sum()
        colors_temporal = [BLUE, AMBER, RED, PURPLE]
        bottom = np.zeros(len(mortality_daily), dtype=float)
        for label, color in zip(mortality_daily.columns, colors_temporal):
            values = mortality_daily[label].to_numpy(float)
            temporal.bar(mortality_daily.index, values, bottom=bottom, width=0.9,
                         color=color, alpha=0.86,
                         label=f"{label}: {int(values.sum())}")
            bottom += values
        nonzero = bottom > 0
        if np.any(nonzero):
            offset = max(float(np.max(bottom))*0.035, 0.12)
            temporal.scatter(mortality_daily.index[nonzero], bottom[nonzero]+offset,
                             marker="x", s=32, color=RED, linewidth=1.3,
                             label="Observed mortality date", zorder=7)
            for date_value, total_value in zip(mortality_daily.index[nonzero], bottom[nonzero]):
                temporal.annotate(str(int(total_value)), (date_value, total_value),
                                  xytext=(0, 4), textcoords="offset points",
                                  ha="center", fontsize=7, color=RED)
        temporal.set_ylim(0, max(1.0, float(np.max(bottom))*1.22 if bottom.size else 1.0))
        process_deaths = int(mortality_daily.to_numpy(float).sum())
        temporal.text(
            0.01, 0.90,
            f"Process-based deaths: {process_deaths}",
            transform=temporal.transAxes, color=RED, fontweight="semibold"
        )
        if process_deaths == 0:
            temporal.text(0.5, 0.52, "No mortality threshold crossed",
                          transform=temporal.transAxes, ha="center", color=MUTED)
        temporal.set_ylabel("Dead cells per timestep")
        temporal.set_title("Observed mortality by cause")
        temporal.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        _format_dates(temporal, dates); _axes_style(temporal, grid="y")

        _bed(spatial, result)
        offsets = np.linspace(0.035, 0.16, 4) * max(np.ptp(result.z), 1.0)
        colors = [BLUE, AMBER, RED, PURPLE]
        labels = ["Inundation", "Direct shear", "Shields / bed mobility", "Drought"]
        for values, offset, color, label in zip(totals, offsets, colors, labels):
            mask = values > 0
            if mask.any():
                spatial.scatter(result.y[mask], result.z[mask] + offset,
                                s=22 + 10*np.sqrt(values[mask]), color=color, alpha=0.82,
                                label=label, zorder=6)
        if any(np.any(values > 0) for values in totals):
            spatial.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        spatial.set_xlabel("Cross-section coordinate [m]")
        spatial.set_ylabel("Elevation [m]")
        spatial.set_title("Cumulative mortality locations")
        _axes_style(spatial, grid="y")
        _save(fig, self._path("recruitment_mortality.png", save))
        return fig

    def plot_hydroperiod_response(
        self,
        result: VegetationResult,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Relate inundation exposure to final recruitment or biomass."""
        inundated_fraction = np.mean(result.forcing.depth > 0.0, axis=0)
        maximum_consecutive = np.zeros(result.y.size, dtype=float)
        dt_days = np.median(np.diff(result.dates.asi8)) / 1e9 / 86400.0
        for j in range(result.y.size):
            wet = result.forcing.depth[:, j] > 0.0
            padded = np.r_[False, wet, False].astype(int)
            changes = np.diff(padded)
            starts = np.where(changes == 1)[0]
            stops = np.where(changes == -1)[0]
            maximum_consecutive[j] = (
                np.max(stops - starts) * dt_days if starts.size else 0.0
            )

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
        x_metric = result.final_age_days if result.mode == "recruitment" else result.final_state
        ylabel = "Final plant age [days]" if result.mode == "recruitment" else "Final biomass [0–1]"
        color = result.z
        points = axes[0].scatter(
            inundated_fraction * 100.0,
            x_metric,
            c=color,
            cmap="viridis",
            s=28,
            alpha=0.82,
        )
        axes[0].set_xlabel("Time inundated [%]")
        axes[0].set_ylabel(ylabel)
        axes[0].set_title("Long-term hydroperiod")
        _axes_style(axes[0], grid="both")
        cbar = fig.colorbar(points, ax=axes[0], pad=0.015)
        cbar.set_label("Bed elevation [m]")

        axes[1].scatter(
            maximum_consecutive,
            x_metric,
            c=color,
            cmap="viridis",
            s=28,
            alpha=0.82,
        )
        axes[1].set_xlabel("Maximum consecutive inundation [days]")
        axes[1].set_ylabel(ylabel)
        axes[1].set_title("Event-duration exposure")
        _axes_style(axes[1], grid="both")
        fig.suptitle(f"Hydrological exposure and {result.mode} response")
        _save(fig, self._path(f"{result.mode}_hydroperiod_response.png", save))
        return fig

    def plot_flow_vegetation_comparison(
        self,
        comparison: VegetationComparison,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Compare natural and post-abstraction flow and vegetation response."""
        natural = comparison.natural
        altered = comparison.altered
        if natural.mode != altered.mode:
            raise ValueError("comparison results must use the same vegetation mode")

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12.5, 9.0),
            sharex=False,
            gridspec_kw={"height_ratios": [1.1, 1.0, 1.35]},
            constrained_layout=True,
        )
        flow, response, section = axes
        flow.plot(natural.dates, natural.forcing.discharge, color=NAVY, linewidth=1.35, label="Natural")
        flow.plot(altered.dates, altered.forcing.discharge, color=AMBER, linewidth=1.15, label=comparison.scenario_name)
        flow.fill_between(
            natural.dates,
            altered.forcing.discharge,
            natural.forcing.discharge,
            where=natural.forcing.discharge >= altered.forcing.discharge,
            color="#F5DDBD",
            alpha=0.55,
            label="Abstracted discharge",
        )
        flow.set_ylabel("Discharge [m³ s⁻¹]")
        flow.set_title("Natural and post-abstraction discharge")
        flow.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, frameon=False)
        _format_dates(flow, natural.dates)
        _axes_style(flow, grid="y")

        if natural.mode == "recruitment":
            nat_metric = natural.presence.sum(axis=1)
            alt_metric = altered.presence.sum(axis=1)
            ylabel = "Living cells [count]"
            title = "Recruitment vegetation through time"
        else:
            habitat = natural.habitat_mask
            nat_metric = np.mean(natural.state[:, habitat], axis=1)
            alt_metric = np.mean(altered.state[:, habitat], axis=1)
            ylabel = "Mean biomass [0–1]"
            title = "Camporeale biomass through time"
        response.plot(natural.dates, nat_metric, color=GREEN, linewidth=1.7, label="Natural")
        response.plot(altered.dates, alt_metric, color=PURPLE, linewidth=1.5, label=comparison.scenario_name)
        response.set_ylim(bottom=0.0)
        response.set_ylabel(ylabel)
        response.set_title(title)
        response.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, frameon=False)
        _format_dates(response, natural.dates)
        _axes_style(response, grid="y")

        _bed(section, natural)
        visual_offset = 0.035 * max(np.ptp(natural.z), 1.0)
        if natural.mode == "recruitment":
            section.scatter(
                natural.y[natural.final_presence],
                natural.z[natural.final_presence] + visual_offset,
                color=GREEN,
                s=28,
                label="Natural final presence",
                zorder=7,
            )
            section.scatter(
                altered.y[altered.final_presence],
                altered.z[altered.final_presence] + 2.2 * visual_offset,
                facecolor="none",
                edgecolor=PURPLE,
                linewidth=1.2,
                s=35,
                label=f"{comparison.scenario_name} final presence",
                zorder=8,
            )
        else:
            habitat = natural.habitat_mask
            section.scatter(
                natural.y[habitat],
                natural.z[habitat] + visual_offset,
                s=12.0 + 55.0 * natural.final_state[habitat],
                color=GREEN,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                label="Natural final biomass",
                zorder=7,
            )
            section.scatter(
                altered.y[habitat],
                altered.z[habitat] + 2.2 * visual_offset,
                s=12.0 + 55.0 * altered.final_state[habitat],
                facecolor="none",
                edgecolor=PURPLE,
                linewidth=1.1,
                label=f"{comparison.scenario_name} final biomass",
                zorder=8,
            )
            section.text(
                0.01,
                0.04,
                "Marker size is proportional to normalized biomass",
                transform=section.transAxes,
                color=MUTED,
                fontsize=9,
            )
        section.set_xlabel("Cross-section coordinate [m]")
        section.set_ylabel("Elevation [m]")
        section.set_title("Final cross-section response")
        section.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, frameon=False)
        _axes_style(section, grid="y")
        fig.suptitle(f"Vegetation response to flow alteration — {natural.mode}")
        _save(fig, self._path(f"{natural.mode}_natural_altered_comparison.png", save))
        return fig

    def plot_mortality_comparison(
        self,
        comparison: VegetationComparison,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Compare recruitment mortality under natural and altered flow."""
        natural = comparison.natural
        altered = comparison.altered
        if natural.mode != "recruitment" or altered.mode != "recruitment":
            raise ValueError("mortality comparison requires recruitment results")

        columns = [
            "mortality_anoxia",
            "mortality_shear",
            "mortality_sediment",
            "mortality_drought",
        ]
        labels = ["Prolonged inundation", "Direct shear", "Shields / bed mobility", "Drought"]
        colors = [BLUE, AMBER, RED, PURPLE]

        fig = plt.figure(figsize=(12.5, 7.5), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 0.9])
        natural_ax = fig.add_subplot(grid[0, 0])
        altered_ax = fig.add_subplot(grid[0, 1], sharey=natural_ax)
        totals_ax = fig.add_subplot(grid[1, :])

        for ax, result, title in [
            (natural_ax, natural, "Natural flow"),
            (altered_ax, altered, comparison.scenario_name),
        ]:
            ax.stackplot(
                result.dates,
                *[result.diagnostics[name] for name in columns],
                labels=labels,
                colors=colors,
                alpha=0.82,
            )
            ax.set_title(title)
            ax.set_ylabel("Cells dying per timestep")
            ax.set_ylim(bottom=0.0)
            _format_dates(ax, result.dates)
            _axes_style(ax, grid="y")
        altered_ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )

        x = np.arange(len(labels))
        natural_totals = [natural.diagnostics[name].sum() for name in columns]
        altered_totals = [altered.diagnostics[name].sum() for name in columns]
        width = 0.36
        totals_ax.bar(
            x - width / 2,
            natural_totals,
            width,
            color=GREEN,
            label="Natural",
        )
        totals_ax.bar(
            x + width / 2,
            altered_totals,
            width,
            color=PURPLE,
            label=comparison.scenario_name,
        )
        totals_ax.set_xticks(x)
        totals_ax.set_xticklabels(labels)
        totals_ax.set_ylabel("Cumulative deaths [cells]")
        totals_ax.set_ylim(bottom=0.0)
        totals_ax.set_title("Cumulative process-based mortality by cause")
        totals_ax.legend(frameon=False, ncol=2)
        _axes_style(totals_ax, grid="y")
        fig.suptitle("Recruitment mortality: natural versus post-abstraction")
        _save(fig, self._path("recruitment_mortality_comparison.png", save))
        return fig

    def plot_final_cross_section_comparison(
        self,
        comparison: VegetationComparison,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Compare natural and altered final states directly on the cross-section.

        The figure is designed to make the Qnat versus Qrel contrast explicit at
        the end of the simulation.  For recruitment it shows final living plants
        under natural flow, final living plants under altered flow, and the
        natural-to-altered transition classes.  For Camporeale it shows final
        biomass under natural flow, final biomass under altered flow, and the
        altered-minus-natural biomass difference.
        """
        natural = comparison.natural
        altered = comparison.altered
        if natural.mode != altered.mode:
            raise ValueError("comparison results must use the same vegetation mode")

        fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
        relief = max(float(np.ptp(natural.z)), 1.0)
        offset = 0.03 * relief
        final_time = pd.Timestamp(natural.dates[-1])

        for ax in axes:
            _bed(ax, natural)
            ax.set_xlabel("Cross-section coordinate [m]")
            ax.set_ylabel("Elevation [m]")
            _axes_style(ax, grid="y")

        if natural.mode == "recruitment":
            # Natural final state
            nat_alive = natural.final_presence
            alt_alive = altered.final_presence
            for ax, mask, title, color in [
                (axes[0], nat_alive, "Natural flow (Qnat)", GREEN),
                (axes[1], alt_alive, f"Altered flow ({comparison.scenario_name})", PURPLE),
            ]:
                ax.scatter(
                    natural.y[natural.habitat_mask],
                    natural.z[natural.habitat_mask],
                    s=11,
                    color="#B8C0C8",
                    alpha=0.8,
                    label="Potential habitat",
                    zorder=5,
                )
                if mask.any():
                    ax.scatter(
                        natural.y[mask],
                        natural.z[mask] + offset,
                        s=35,
                        color=color,
                        edgecolor="white",
                        linewidth=0.45,
                        label=f"Living recruits: {int(mask.sum())}",
                        zorder=8,
                    )
                else:
                    ax.text(0.5, 0.08, "No living recruits", transform=ax.transAxes, ha="center", color=MUTED)
                ax.set_title(title)
                ax.legend(loc="upper left", frameon=False)

            lost = nat_alive & ~alt_alive
            persistent = nat_alive & alt_alive
            gained = ~nat_alive & alt_alive
            groups = [
                (lost, RED, "Lost with Qrel"),
                (persistent, GREEN, "Persistent"),
                (gained, BLUE, "Gained with Qrel"),
            ]
            axes[2].scatter(
                natural.y[natural.habitat_mask],
                natural.z[natural.habitat_mask],
                s=11,
                color="#D2D8DE",
                alpha=0.7,
                label="Potential habitat",
                zorder=4,
            )
            for mask, color, label in groups:
                if mask.any():
                    axes[2].scatter(
                        natural.y[mask],
                        natural.z[mask] + offset,
                        s=38,
                        color=color,
                        edgecolor="white",
                        linewidth=0.45,
                        label=label,
                        zorder=8,
                    )
            axes[2].set_title("Final transition: Qnat vs Qrel")
            axes[2].legend(loc="upper left", frameon=False)
            fig.suptitle(
                f"Recruitment comparison at the end of the simulation — {final_time:%Y-%m-%d}",
                fontsize=12,
            )
        else:
            habitat = natural.habitat_mask
            vmax = max(
                float(np.nanmax(natural.final_state[habitat])) if habitat.any() else 0.0,
                float(np.nanmax(altered.final_state[habitat])) if habitat.any() else 0.0,
                1e-6,
            )
            cmap = plt.get_cmap("YlGn")
            diff = altered.final_state - natural.final_state
            dmax = max(float(np.nanmax(np.abs(diff[habitat]))) if habitat.any() else 0.0, 1e-6)

            for ax, values, title in [
                (axes[0], natural.final_state, "Natural flow (Qnat)"),
                (axes[1], altered.final_state, f"Altered flow ({comparison.scenario_name})"),
            ]:
                pts = ax.scatter(
                    natural.y[habitat],
                    natural.z[habitat] + offset,
                    c=values[habitat],
                    cmap=cmap,
                    norm=Normalize(0.0, vmax),
                    s=48,
                    edgecolor=CHARCOAL,
                    linewidth=0.25,
                    zorder=8,
                )
                ax.set_title(title)
                cbar = fig.colorbar(pts, ax=ax, pad=0.01)
                cbar.set_label("Final biomass [0–1]")

            pts = axes[2].scatter(
                natural.y[habitat],
                natural.z[habitat] + offset,
                c=diff[habitat],
                cmap="BrBG",
                norm=Normalize(-dmax, dmax),
                s=48,
                edgecolor=CHARCOAL,
                linewidth=0.25,
                zorder=8,
            )
            axes[2].set_title("Biomass difference (Qrel − Qnat)")
            cbar = fig.colorbar(pts, ax=axes[2], pad=0.01)
            cbar.set_label("Δ biomass [0–1]")
            fig.suptitle(
                f"Camporeale comparison at the end of the simulation — {final_time:%Y-%m-%d}",
                fontsize=12,
            )

        _save(fig, self._path(f"{natural.mode}_final_cross_section_comparison.png", save))
        return fig

    def plot_final_transition(
        self,
        comparison: VegetationComparison,
        *,
        save: bool | str | Path = False,
    ) -> plt.Figure:
        """Map final vegetation loss, persistence and gain after abstraction."""
        natural = comparison.natural
        altered = comparison.altered
        fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
        _bed(ax, natural)

        if natural.mode == "recruitment":
            lost = natural.final_presence & ~altered.final_presence
            persistent = natural.final_presence & altered.final_presence
            gained = ~natural.final_presence & altered.final_presence
            groups = [
                (lost, RED, "Lost after abstraction"),
                (persistent, GREEN, "Persistent"),
                (gained, BLUE, "Gained after abstraction"),
            ]
            for mask, color, label in groups:
                if mask.any():
                    ax.scatter(
                        natural.y[mask],
                        natural.z[mask],
                        color=color,
                        s=38,
                        edgecolor="white",
                        linewidth=0.45,
                        label=label,
                        zorder=7,
                    )
        else:
            difference = altered.final_state - natural.final_state
            points = ax.scatter(
                natural.y[natural.habitat_mask],
                natural.z[natural.habitat_mask],
                c=difference[natural.habitat_mask],
                cmap="BrBG",
                norm=Normalize(-max(np.max(np.abs(difference)), 1e-6), max(np.max(np.abs(difference)), 1e-6)),
                s=38,
                edgecolor=CHARCOAL,
                linewidth=0.25,
                zorder=7,
            )
            cbar = fig.colorbar(points, ax=ax, pad=0.015)
            cbar.set_label("Altered − natural biomass")
        ax.set_xlabel("Cross-section coordinate [m]")
        ax.set_ylabel("Elevation [m]")
        ax.set_title(f"Final vegetation transition — {comparison.scenario_name}")
        if natural.mode == "recruitment" and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", ncol=3, frameon=False)
        _axes_style(ax, grid="y")
        _save(fig, self._path(f"{natural.mode}_final_transition.png", save))
        return fig


def save_vegetation_analysis(analysis: Any, output_dir: str | Path) -> list[Path]:
    """Render the standard vegetation figure set for a full analysis.

    The output contains natural-flow state, age/biomass, mortality and
    hydroperiod figures for every selected initial condition, plus paired
    discharge-response and final-transition figures for every scenario.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plotter = VegetationPlotter(destination)
    written: list[Path] = []

    for initial, by_mode in analysis.natural.items():
        for mode, result in by_mode.items():
            prefix = f"natural_{initial}_{mode}"
            if mode == "recruitment":
                figures = [
                    (plotter.plot_recruitment_state(result), f"{prefix}_state.png"),
                    (plotter.plot_recruitment_age(result), f"{prefix}_age.png"),
                    (plotter.plot_annual_recruitment_opportunity(result), f"{prefix}_annual_opportunity.png"),
                    (plotter.plot_candidate_establishment_dynamics(result), f"{prefix}_candidate_dynamics.png"),
                    (plotter.plot_mortality(result), f"{prefix}_mortality.png"),
                ]
            else:
                figures = [
                    (plotter.plot_camporeale_biomass(result), f"{prefix}_biomass.png")
                ]
            figures.append(
                (
                    plotter.plot_hydroperiod_response(result),
                    f"{prefix}_hydroperiod.png",
                )
            )
            for fig, filename in figures:
                path = destination / filename
                fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                written.append(path)

    for scenario_name, by_initial in analysis.comparisons.items():
        safe_scenario = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in scenario_name
        )
        for initial, by_mode in by_initial.items():
            for mode, comparison in by_mode.items():
                prefix = f"{safe_scenario}_{initial}_{mode}"
                figures = [
                    (
                        plotter.plot_flow_vegetation_comparison(comparison),
                        f"{prefix}_flow_response.png",
                    ),
                    (
                        plotter.plot_final_cross_section_comparison(comparison),
                        f"{prefix}_final_cross_section_comparison.png",
                    ),
                    (
                        plotter.plot_final_transition(comparison),
                        f"{prefix}_final_transition.png",
                    ),
                ]
                if mode == "recruitment":
                    figures.append(
                        (
                            plotter.plot_mortality_comparison(comparison),
                            f"{prefix}_mortality_comparison.png",
                        )
                    )
                for fig, filename in figures:
                    path = destination / filename
                    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
                    plt.close(fig)
                    written.append(path)
    return written
