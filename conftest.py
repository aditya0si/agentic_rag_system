"""Repository-wide pytest policy.

Every collected test must carry one of the declared markers (``unit``,
``integration``, ``llm``). This is a guard against silent coverage loss: the CI
jobs select tests *by marker*, so an unmarked test file would simply never run.
Failing collection loudly is the only way an unmarked test can be noticed.
"""

import pytest

REQUIRED_MARKERS = ("unit", "integration", "llm")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    unmarked = sorted(
        item.nodeid
        for item in items
        if not any(item.get_closest_marker(marker) for marker in REQUIRED_MARKERS)
    )
    if unmarked:
        raise pytest.UsageError(
            "Unmarked test(s) found - every test must be marked 'unit', 'integration' or 'llm' "
            "so the CI marker selections pick it up:\n  " + "\n  ".join(unmarked)
        )
