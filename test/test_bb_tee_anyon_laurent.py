import numpy as np
import pytest
from scipy import sparse

from BB_TEE_anyon import build_stabilizer_matrix_laurent


def test_build_stabilizer_matrix_laurent_basic() -> None:
    omega = [(1, 0), (0, 0), (1, 0)]  # unsorted with dup
    f = [(0, 0)]
    g = [(0, 0)]

    mat, site_index = build_stabilizer_matrix_laurent(omega, f, g)

    assert site_index == {(0, 0): 0, (1, 0): 1}
    assert sparse.issparse(mat)
    expected = np.array(
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert mat.shape == expected.shape
    assert np.array_equal(mat.toarray(), expected)


def test_build_stabilizer_matrix_laurent_strict_boundary() -> None:
    omega = [(0, 0)]
    f = [(0, 0), (1, 0)]
    g = [(0, 0)]

    mat_strict, _ = build_stabilizer_matrix_laurent(omega, f, g, strict=True)
    assert mat_strict.shape == (0, 4)

    mat_loose, _ = build_stabilizer_matrix_laurent(omega, f, g, strict=False)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(mat_loose.toarray(), expected)


def test_build_stabilizer_matrix_laurent_centers_validation() -> None:
    omega = [(0, 0), (1, 0)]
    with pytest.raises(ValueError):
        build_stabilizer_matrix_laurent(
            omega, [(0, 0)], [(0, 0)], centers=[(0, 0), (2, 0)]
        )


def test_build_stabilizer_matrix_laurent_commutation_guard() -> None:
    omega = [(0, 0), (1, 0)]
    f = [(1, 0)]
    g = [(-1, 0)]

    with pytest.raises(ValueError):
        build_stabilizer_matrix_laurent(omega, f, g, strict=False, check_commutation=True)

    mat, _ = build_stabilizer_matrix_laurent(
        omega, f, g, strict=False, check_commutation=False
    )
    assert mat.shape == (4, 8)
