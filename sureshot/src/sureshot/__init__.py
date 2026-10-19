"""SureShot SAST - static Rust vulnerability detection with calibrated confidence."""

__version__ = "0.1.0"

# Import order note: `persist` and `scan` pull in XGBoost only inside the
# functions that need it, not at module level, so importing the package stays
# cheap for callers that only want the feature extractor.
from . import calibrate, data, features, metrics, persist, pipeline, scan

__all__ = ["calibrate", "data", "features", "metrics", "persist", "pipeline",
           "scan"]
