"""Pytest fixtures shared across all test modules."""

import pytest
from bmt.toolkit import Toolkit


@pytest.fixture(scope="session")
def bmt():
    """Create a BMT Toolkit instance shared across all tests.

    Using session scope ensures BMT is only initialized once for the entire
    test run, significantly improving test performance.
    """
    return Toolkit()
