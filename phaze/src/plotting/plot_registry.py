"""Registry system for PHAZE plot types."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from .base_plotter import BasePlotter


@dataclass
class PlotTypeInfo:
    """Information about a registered plot type."""

    name: str
    description: str
    plotter_class: Type[BasePlotter]
    category: str
    priority: int = 0  # Higher priority plotters appear first

    def __post_init__(self):
        """Validate plotter class."""
        if not issubclass(self.plotter_class, BasePlotter):
            raise ValueError(
                f"Plotter class {self.plotter_class} must inherit from BasePlotter"
            )


class PlotRegistry:
    """Registry for available plot types in PHAZE.

    Provides a centralized way to discover and instantiate plotters.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._plot_types: Dict[str, PlotTypeInfo] = {}
        self._categories: Dict[str, List[str]] = {}

    def register_plot_type(
        self,
        name: str,
        plotter_class: Type[BasePlotter],
        description: str,
        category: str = "general",
        priority: int = 0,
    ) -> None:
        """Register a new plot type.

        Args:
            name: Unique name for the plot type
            plotter_class: Class that implements the plotter
            description: Human-readable description
            category: Category for grouping (e.g., 'fingerprint', 'zkml')
            priority: Priority for ordering (higher = first)
        """
        if name in self._plot_types:
            raise ValueError(f"Plot type '{name}' is already registered")

        plot_info = PlotTypeInfo(
            name=name,
            description=description,
            plotter_class=plotter_class,
            category=category,
            priority=priority,
        )

        self._plot_types[name] = plot_info

        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)

        # Sort by priority within category
        self._categories[category].sort(
            key=lambda x: self._plot_types[x].priority, reverse=True
        )

    def get_plot_type(self, name: str) -> Optional[PlotTypeInfo]:
        """Get information about a registered plot type.

        Args:
            name: Name of the plot type

        Returns:
            PlotTypeInfo object or None if not found
        """
        return self._plot_types.get(name)

    def list_plot_types(self, category: Optional[str] = None) -> List[str]:
        """List available plot types.

        Args:
            category: Optional category filter

        Returns:
            List of plot type names
        """
        if category is None:
            # Return all plot types sorted by priority
            return sorted(
                self._plot_types.keys(),
                key=lambda x: self._plot_types[x].priority,
                reverse=True,
            )
        else:
            return self._categories.get(category, [])

    def list_categories(self) -> List[str]:
        """List available categories.

        Returns:
            List of category names
        """
        return sorted(self._categories.keys())

    def create_plotter(
        self, plot_type: str, config: Optional[Any] = None
    ) -> BasePlotter:
        """Create an instance of a registered plotter.

        Args:
            plot_type: Name of the plot type to create
            config: Configuration object to pass to plotter

        Returns:
            Instantiated plotter object

        Raises:
            ValueError: If plot type is not registered
        """
        plot_info = self.get_plot_type(plot_type)
        if plot_info is None:
            raise ValueError(f"Unknown plot type: {plot_type}")

        return plot_info.plotter_class(config)

    def get_compatible_plot_types(self, data_fields: List[str]) -> List[str]:
        """Get plot types that are compatible with given data fields.

        Args:
            data_fields: List of available data fields

        Returns:
            List of compatible plot type names
        """
        compatible = []

        for name, plot_info in self._plot_types.items():
            try:
                # Create temporary instance to check requirements
                plotter = plot_info.plotter_class()
                required_fields = plotter.get_required_data_fields()

                # Check if all required fields are available
                if all(field in data_fields for field in required_fields):
                    compatible.append(name)
            except Exception:
                # Skip if plotter can't be instantiated
                continue

        return compatible

    def get_plot_info_summary(self) -> Dict[str, Any]:
        """Get summary information about all registered plot types.

        Returns:
            Dictionary with summary information
        """
        summary = {
            "total_plot_types": len(self._plot_types),
            "categories": {},
            "plot_types": {},
        }

        # Category summary
        for category, plot_types in self._categories.items():
            summary["categories"][category] = {
                "count": len(plot_types),
                "plot_types": plot_types,
            }

        # Individual plot type info
        for name, plot_info in self._plot_types.items():
            summary["plot_types"][name] = {
                "description": plot_info.description,
                "category": plot_info.category,
                "priority": plot_info.priority,
                "class_name": plot_info.plotter_class.__name__,
            }

        return summary


# Global registry instance
_global_registry = PlotRegistry()


def register_plot_type(
    name: str,
    plotter_class: Type[BasePlotter],
    description: str,
    category: str = "general",
    priority: int = 0,
) -> None:
    """Register a plot type with the global registry.

    Args:
        name: Unique name for the plot type
        plotter_class: Class that implements the plotter
        description: Human-readable description
        category: Category for grouping
        priority: Priority for ordering
    """
    _global_registry.register_plot_type(
        name, plotter_class, description, category, priority
    )


def get_global_registry() -> PlotRegistry:
    """Get the global plot registry instance.

    Returns:
        Global PlotRegistry instance
    """
    return _global_registry
