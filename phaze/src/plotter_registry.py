"""Register all available PHAZE plotters."""

from .plotting.plot_registry import get_global_registry, register_plot_type
from .plotting.plotters.fingerprint_plotter import FingerprintPlotter
from .plotting.plotters.zkml_plotter import ZKMLPlotter

# Track if plotters are already registered
_plotters_registered = False


def register_all_plotters():
    """Register all available PHAZE plotters with the global registry."""
    global _plotters_registered

    if _plotters_registered:
        return  # Already registered, skip

    registry = get_global_registry()

    # Register fingerprinting plotters
    if "fingerprint" not in registry._plot_types:
        register_plot_type(
            name="fingerprint",
            plotter_class=FingerprintPlotter,
            description="Performance analysis plots for cryptographic fingerprinting algorithms",
            category="crypto",
            priority=10
        )

    # Register zkML framework plotters
    if "zkml-proof" not in registry._plot_types:
        register_plot_type(
            name="zkml-proof",
            plotter_class=ZKMLPlotter,
            description="Performance analysis plots for zkML proof generation and verification",
            category="zkml",
            priority=10
        )

    if "zkml-verify" not in registry._plot_types:
        register_plot_type(
            name="zkml-verify",
            plotter_class=ZKMLPlotter,
            description="Verification performance analysis plots for zkML frameworks",
            category="zkml",
            priority=9
        )

    _plotters_registered = True


# Auto-register when module is imported
register_all_plotters()
