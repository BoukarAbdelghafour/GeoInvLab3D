import pytest

from geoinvlab.methods.base import GeophysicalMethod
from geoinvlab.methods.registry import MethodRegistry


class DummyMethod(GeophysicalMethod):
    def __init__(self, identifier_value: str, name_value: str = "Dummy"):
        self._identifier = identifier_value
        self._name = name_value

    @property
    def name(self) -> str:
        return self._name

    @property
    def identifier(self) -> str:
        return self._identifier


def test_registry_is_empty_at_creation():
    registry = MethodRegistry()

    assert registry.available() == []
    assert registry.identifiers() == []


def test_register_method():
    registry = MethodRegistry()
    method = DummyMethod("dummy")

    registry.register(method)

    assert registry.get("dummy") is method
    assert registry.identifiers() == ["dummy"]
    assert registry.available() == [method]


def test_contains_registered_method():
    registry = MethodRegistry()
    method = DummyMethod("gravity")

    registry.register(method)

    assert "gravity" in registry
    assert "magnetic" not in registry


def test_register_multiple_methods():
    registry = MethodRegistry()

    gravity = DummyMethod("gravity", "Gravity")
    magnetic = DummyMethod("magnetic", "Magnetic")

    registry.register(gravity)
    registry.register(magnetic)

    assert registry.identifiers() == ["gravity", "magnetic"]
    assert registry.get("gravity") is gravity
    assert registry.get("magnetic") is magnetic


def test_register_empty_identifier_raises_error():
    registry = MethodRegistry()
    method = DummyMethod("")

    with pytest.raises(ValueError, match="Method identifier must not be empty"):
        registry.register(method)


def test_register_duplicate_identifier_raises_error():
    registry = MethodRegistry()

    first = DummyMethod("gravity")
    second = DummyMethod("gravity")

    registry.register(first)

    with pytest.raises(
        ValueError,
        match="Method 'gravity' is already registered",
    ):
        registry.register(second)


def test_get_unknown_method_raises_error():
    registry = MethodRegistry()

    with pytest.raises(
        KeyError,
        match="Unknown geophysical method: 'unknown'",
    ):
        registry.get("unknown")

