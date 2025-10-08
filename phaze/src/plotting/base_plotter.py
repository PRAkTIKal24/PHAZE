"""Abstract base class for all PHAZE plotters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.figure
from .plot_config import PlotConfig


class BasePlotter(ABC):
    """Abstract base class for all PHAZE plotters.
    
    Provides a standard interface for generating plots from benchmark data
    with consistent styling and statistical analysis.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize the plotter with configuration.
        
        Args:
            config: Plot configuration object. If None, uses default config.
        """
        self.config = config or PlotConfig()
        self._setup_plotting()
    
    def _setup_plotting(self) -> None:
        """Setup matplotlib with the configured style."""
        self.config.apply_style()
    
    @abstractmethod
    def generate_plots(self, data: Dict[str, Any]) -> List[matplotlib.figure.Figure]:
        """Generate plots from benchmark data.
        
        Args:
            data: Benchmark data dictionary containing results and metrics
            
        Returns:
            List of matplotlib Figure objects
        """
        pass
    
    @abstractmethod
    def get_supported_plot_types(self) -> List[str]:
        """Get list of plot types supported by this plotter.
        
        Returns:
            List of supported plot type names
        """
        pass
    
    def save_plots(
        self,
        figures: List[matplotlib.figure.Figure],
        output_dir: Path,
        prefix: str = "",
        formats: Optional[List[str]] = None
    ) -> List[Path]:
        """Save plots to specified directory.
        
        Args:
            figures: List of matplotlib figures to save
            output_dir: Directory to save plots in
            prefix: Prefix for plot filenames
            formats: List of file formats to save (png, pdf, svg)
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        formats = formats or self.config.export_formats
        saved_files = []
        
        for i, fig in enumerate(figures):
            # Get plot title or use index
            title = getattr(fig, '_suptitle', None)
            if title and hasattr(title, 'get_text'):
                plot_name = title.get_text().lower().replace(' ', '_').replace('/', '_')
            else:
                plot_name = f"plot_{i+1}"
            
            # Remove invalid filename characters
            plot_name = "".join(c for c in plot_name if c.isalnum() or c in ('_', '-'))
            
            for fmt in formats:
                filename = f"{prefix}{plot_name}.{fmt}"
                filepath = output_dir / filename
                
                # Save with high DPI for publication quality
                fig.savefig(
                    filepath,
                    format=fmt,
                    dpi=self.config.dpi,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none'
                )
                saved_files.append(filepath)
        
        return saved_files
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate that data contains required fields for plotting.
        
        Args:
            data: Benchmark data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_fields = self.get_required_data_fields()
        return all(field in data for field in required_fields)
    
    @abstractmethod
    def get_required_data_fields(self) -> List[str]:
        """Get list of required data fields for this plotter.
        
        Returns:
            List of required field names in the data dictionary
        """
        pass
