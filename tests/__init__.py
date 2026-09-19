"""Cross-cutting regression suites (cache, metrics, security, golden set).

Package marker so these modules are imported as ``tests.*`` instead of bare
top-level module names, which collided with ``backend/tests/*`` during
collection and type-checking.
"""
