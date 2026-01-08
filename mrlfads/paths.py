"""Define default filesystem paths used throughout the MRLFADS codebase.

This module centralizes common directory paths for data storage, experiment
outputs, and related artifacts. The paths defined here are intended to be
imported by other modules within the MRLFADS package.
"""
import os
from pathlib import Path

homepath = str(Path.home())
datapath = os.path.join(homepath, "data")
resultpath = os.path.join(homepath, "runs")
raypath = resultpath