"""Suvadu — does fine-tuning on personal agent traces actually help?

Package layout:
    suvadu.provenance  — the 5-identifier run manifest (config, code, data, seed, environment)
    suvadu.config      — versioned config contract with typed validation
    suvadu.cli         — entrypoints (train, baselines, eval)

Nothing in this package may report a metric that lacks a provenance manifest. See AGENTS.md §2.4.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
