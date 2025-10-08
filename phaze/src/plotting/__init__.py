"""PHAZE Plotting Module.

This module provides comprehensive plotting capabilities for PHAZE benchmarking results,
with support for statistical analysis and publication-ready visualizations.
"""

from .base_plotter import BasePlotter
from .plot_config import PlotConfig, PlotStyle
from .plot_registry import PlotRegistry
from .plot_suite import PHAZEPlotSuite
from .statistical_utils import StatisticalUtils

__all__ = [
    "BasePlotter",
    "PlotConfig",
    "PlotStyle",
    "StatisticalUtils",
    "PlotRegistry",
    "PHAZEPlotSuite",
]
