"""Impulse Response Downloader (irdl): Download, unpack and process impulse response datasets."""

import sys as _sys

from .base import _get_dataset_classes as _get_dataset_classes
from .ista import MiracleDataset as MiracleDataset
from .ista import SrirachaDataset as SrirachaDataset
from .sofa import FabianDataset as FabianDataset

__all__ = [dataset_class.__name__ for dataset_class in _get_dataset_classes(_sys.modules[__name__])]
