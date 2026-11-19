import pytest

from geoinvlab.methods.base import GeophysicalMethod


def test_geophysical_method_is_abstract():
    with pytest.raises(TypeError):
        GeophysicalMethod()


def test_concrete_method_implements_interface():
    class GravityMethod(GeophysicalMethod):
        @property
        def name(self) -> str:
            return "Gravity"

        @property
        def identifier(self) -> str:
            return "gravity"

    method = GravityMethod()

    assert method.name == "Gravity"
    assert method.identifier == "gravity"

