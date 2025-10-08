"""Configuration for PHAZE plotting system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


class PlotStyle(Enum):
    """Available plot styles."""
    PUBLICATION = "publication"
    PRESENTATION = "presentation"
    WEB = "web"
    NEURIPS = "neurips"


@dataclass
class PlotConfig:
    """Configuration for PHAZE plotting system."""

    # Style settings
    style: PlotStyle = PlotStyle.NEURIPS
    dpi: int = 300
    figure_size: Tuple[float, float] = (10, 6)
    font_size: int = 12
    title_size: int = 14
    legend_size: int = 10

    # Colors (colorblind-friendly palette)
    primary_colors: List[str] = None
    secondary_colors: List[str] = None

    # Export settings
    export_formats: List[str] = None

    # Statistical settings
    confidence_level: float = 0.95
    num_trials: int = 10

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.primary_colors is None:
            # Colorblind-friendly palette based on ColorBrewer
            self.primary_colors = [
                '#1f77b4',  # Blue
                '#ff7f0e',  # Orange
                '#2ca02c',  # Green
                '#d62728',  # Red
                '#9467bd',  # Purple
                '#8c564b',  # Brown
                '#e377c2',  # Pink
                '#7f7f7f',  # Gray
                '#bcbd22',  # Olive
                '#17becf'   # Cyan
            ]

        if self.secondary_colors is None:
            # Lighter versions for error bars, fill areas
            self.secondary_colors = [
                '#aec7e8',  # Light blue
                '#ffbb78',  # Light orange
                '#98df8a',  # Light green
                '#ff9896',  # Light red
                '#c5b0d5',  # Light purple
                '#c49c94',  # Light brown
                '#f7b6d3',  # Light pink
                '#c7c7c7',  # Light gray
                '#dbdb8d',  # Light olive
                '#9edae5'   # Light cyan
            ]

        if self.export_formats is None:
            self.export_formats = ['png', 'pdf']

    def apply_style(self) -> None:
        """Apply the configured style to matplotlib."""
        # Set style based on configuration
        if self.style == PlotStyle.PUBLICATION:
            self._apply_publication_style()
        elif self.style == PlotStyle.PRESENTATION:
            self._apply_presentation_style()
        elif self.style == PlotStyle.WEB:
            self._apply_web_style()
        elif self.style == PlotStyle.NEURIPS:
            self._apply_neurips_style()

    def _apply_publication_style(self) -> None:
        """Apply publication-ready style settings."""
        plt.style.use('default')
        mpl.rcParams.update({
            'figure.figsize': self.figure_size,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.legend_size,
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Liberation Serif', 'serif'],
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': self.dpi,
        })

    def _apply_presentation_style(self) -> None:
        """Apply presentation-friendly style settings."""
        plt.style.use('default')
        mpl.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif'],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': self.dpi,
        })

    def _apply_web_style(self) -> None:
        """Apply web-friendly style settings."""
        plt.style.use('default')
        mpl.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif'],
            'axes.grid': True,
            'grid.alpha': 0.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 150,
        })

    def _apply_neurips_style(self) -> None:
        """Apply NeurIPS conference style settings."""
        plt.style.use('default')
        mpl.rcParams.update({
            'figure.figsize': self.figure_size,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.font_size,
            'xtick.labelsize': self.font_size - 1,
            'ytick.labelsize': self.font_size - 1,
            'legend.fontsize': self.legend_size,
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times New Roman', 'Times', 'serif'],
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.axisbelow': True,
            'figure.dpi': self.dpi,
        })

    def get_color_cycle(self, n_colors: Optional[int] = None) -> List[str]:
        """Get a cycle of colors for plotting.

        Args:
            n_colors: Number of colors needed. If None, returns all primary colors.

        Returns:
            List of color strings
        """
        if n_colors is None:
            return self.primary_colors

        # Cycle through colors if more are needed
        colors = []
        for i in range(n_colors):
            colors.append(self.primary_colors[i % len(self.primary_colors)])

        return colors

    def get_framework_colors(self) -> Dict[str, str]:
        """Get consistent colors for zkML frameworks.

        Returns:
            Dictionary mapping framework names to colors
        """
        return {
            'ezkl': self.primary_colors[0],     # Blue
            'risc_zero': self.primary_colors[1], # Orange
            'groth16': self.primary_colors[2],   # Green
            'plonky': self.primary_colors[3],    # Red
            'halo': self.primary_colors[4],      # Purple
        }

    def get_algorithm_colors(self) -> Dict[str, str]:
        """Get consistent colors for algorithms.

        Returns:
            Dictionary mapping algorithm names to colors
        """
        return {
            'rabin': self.primary_colors[0],     # Blue
            'shamir': self.primary_colors[1],    # Orange
            'sha256': self.primary_colors[2],    # Green
            'keccak256': self.primary_colors[3], # Red
        }
