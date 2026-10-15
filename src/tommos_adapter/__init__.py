"""This subpackage provides an Ubermag adapter package for `tommos`."""

import importlib.metadata

import pytest

from . import scripts as scripts
from .tommos_driver import HysteresisDriver as HysteresisDriver
from .tommos_runner import TommosRunner as TommosRunner

__version__ = importlib.metadata.version(__package__)


def test():
    """Run all package tests.

    Examples:
    --------
    1. Run all tests.

    >>> import tommos_adapter
    ...
    >>> # tommos_adapter.test()

    """
    return pytest.main(["-v", "--pyargs", "tommos_adapter", "-l"])  # pragma: no cover
