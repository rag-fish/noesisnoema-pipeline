"""
noesisnoema-pipeline CLI package.

Exposes the nn-pipeline command-line entrypoint and the testable
pipeline function that backs it.
"""

from .build_ragpack import app, run_pipeline

__all__ = ["app", "run_pipeline"]

