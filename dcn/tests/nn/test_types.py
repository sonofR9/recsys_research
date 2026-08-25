import copy
import pickle

import pytest

from dcn.nn.types import OutputDims


def test_a_dimension_is_reachable_by_name() -> None:
    dims = OutputDims({"shared": 4, "like": 2})

    assert dims.shared == 4
    assert dims.dims["like"] == 2


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        OutputDims({"shared": 4}).listen


def test_it_survives_being_copied_and_pickled() -> None:
    """Both hand the half-built object attribute lookups of their own, before
    `dims` exists to answer them."""
    dims = OutputDims({"shared": 4})

    assert copy.deepcopy(dims).shared == 4
    assert pickle.loads(pickle.dumps(dims)).shared == 4
