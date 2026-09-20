"""Agentic RAG Research Assistant - backend application package.

The application is imported as ``backend.*`` from the repository root (this is
also how the Docker image starts it: ``uvicorn backend.main:app``). Making the
package explicit keeps module identity stable for pytest's import mode and for
``mypy``, which previously could not check the tree at all because
``backend/test_*.py`` and ``backend/tests/test_*.py`` mapped onto the same
top-level module names.
"""
