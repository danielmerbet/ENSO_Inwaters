"""ENSO_Inwaters: super El Nino impacts on global inland waters (ISIMIP3a).

The package holds the reusable science code. The numbered scripts in
``workflow/`` are thin drivers that call into it, so that every step can
also be exercised from a notebook or a test.
"""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
