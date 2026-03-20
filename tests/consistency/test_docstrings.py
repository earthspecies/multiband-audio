"""Verify all public classes and functions have docstrings."""

from __future__ import annotations

import inspect

import multiband_audio as mba


def _get_public_members() -> None:
    """Yield (name, obj) for all public exports.

    Yields
    ------
    tuple[str, object]
        Name and object for each public export.
    """
    for name in mba.__all__:
        obj = getattr(mba, name)
        if inspect.isclass(obj) or inspect.isfunction(obj):
            yield name, obj


def test_all_public_have_docstrings() -> None:
    """Every public class/function must have a docstring."""
    missing = []
    for name, obj in _get_public_members():
        if not obj.__doc__:
            missing.append(name)
    assert not missing, f"Missing docstrings: {missing}"


def test_all_public_classes_have_init_or_forward_docs() -> None:
    """Public classes should document __init__ or forward."""
    for name, obj in _get_public_members():
        if not inspect.isclass(obj):
            continue
        # At least the class docstring should exist (already tested above)
        # Check that forward has a docstring if it exists
        if hasattr(obj, "forward") and callable(obj.forward):
            method = obj.forward
            # Skip abstract methods
            if getattr(method, "__isabstractmethod__", False):
                continue
            assert method.__doc__, f"{name}.forward() missing docstring"
