"""Stabilizer-based entanglement entropy tools for BB codes.

Protocol summary (stabilizer-state method):
1) Build BB stabilizer generators from Hx/Hz.
2) Fix logicals to pick a stabilizer state (default: +1 eigenstate of all logical Z).
3) (Optional) create a stim.Tableau from the stabilizers.
4) For a subsystem A, compute S(A) = rank(G_A_bar) - |A_bar| where G_A_bar is the
   stabilizer generator matrix restricted to the qubits in the complement of A.
5) (Optional) Combine entropies of regions to estimate topological entanglement.
"""

from __future__ import annotations

from ast import main
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import stim
import ldpc.mod2 as mod2
import sympy as sp

import pytest

from minimal_ann_theory_logicalX import orthogonalize_logical_x_matrix
from minimal_ann_matrix import _monomial_basis, vector_to_poly
from minimal_ann_theory_logicalX import pair_css_logicals_from_polynomials

from copyreg import dispatch_table
import enum
from turtle import st
from bposd.css import css_code
from numpy import block
from bivariate_bicycle_codes import get_BB_Hx_Hz

x, y = sp.symbols("x y")


@dataclass(frozen=True)
class BBCodeSpec:
    """Minimal BB code specification."""

    a_poly: Sequence[Tuple[int, int]]
    b_poly: Sequence[Tuple[int, int]]
    l: int
    m: int


def _matrix_to_uint8(mat: object) -> np.ndarray:
    """Convert dense/sparse input to a 2D uint8 matrix modulo 2."""
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    arr = np.asarray(mat, dtype=np.uint8) % 2
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _pauli_from_support(
    support: np.ndarray,
    *,
    pauli: str,
) -> stim.PauliString:
    """Create a PauliString with a single Pauli type on a support mask."""
    xs = np.zeros_like(support, dtype=np.bool_)
    zs = np.zeros_like(support, dtype=np.bool_)
    if pauli == "X":
        xs = support.astype(np.bool_, copy=False)
    elif pauli == "Z":
        zs = support.astype(np.bool_, copy=False)
    else:
        raise ValueError("pauli must be 'X' or 'Z'")
    return stim.PauliString.from_numpy(xs=xs, zs=zs)


def _pauli_from_xz_support(
    xs: np.ndarray, zs: np.ndarray, *, sign: complex = 1
) -> stim.PauliString:
    """Create a PauliString from X/Z support masks."""
    if xs.shape != zs.shape:
        raise ValueError("X/Z supports must have the same shape")
    xs_b = xs.astype(np.bool_, copy=False)
    zs_b = zs.astype(np.bool_, copy=False)
    return stim.PauliString.from_numpy(xs=xs_b, zs=zs_b, sign=sign)


def _normalize_logical_phases(
    phases: Optional[Sequence[complex]],
    count: int,
) -> List[complex]:
    if phases is None:
        return [1] * count
    if len(phases) != count:
        raise ValueError("logicals_phase length must match number of logicals")
    return [complex(p) for p in phases]


def _stabilizers_from_css(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    logicals_x: Optional[np.ndarray] = None,
    logicals_z: Optional[np.ndarray] = None,
    logicals_phase: Optional[Sequence[complex]] = None,
) -> List[stim.PauliString]:
    """Return stabilizer generators as stim.PauliString objects."""
    stabs: List[stim.PauliString] = []
    for row in hx:
        stabs.append(_pauli_from_support(row, pauli="X"))
    for row in hz:
        stabs.append(_pauli_from_support(row, pauli="Z"))
    if logicals_x is not None or logicals_z is not None:
        if logicals_x is None or logicals_z is None:
            raise ValueError("Both logicals_x and logicals_z must be provided for mixed logicals")
        if logicals_x.shape != logicals_z.shape:
            raise ValueError("logicals_x and logicals_z must have the same shape")
        # print(stabs)
        # print(logicals_x)
        # print(logicals_z)
        phases = _normalize_logical_phases(logicals_phase, logicals_x.shape[0])
        for (x_row, z_row), phase in zip(zip(logicals_x, logicals_z), phases):
            stabs.append(_pauli_from_xz_support(x_row, z_row, sign=phase))
        # print(stabs)
    return stabs


def _stabilizer_matrix_from_css(
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    logicals_x: Optional[np.ndarray] = None,
    logicals_z: Optional[np.ndarray] = None,
    logicals_phase: Optional[Sequence[complex]] = None,
) -> np.ndarray:
    """Return stabilizer generators as an r x 2n binary matrix.

    Phases are ignored in the matrix representation.
    """
    hx_u = _matrix_to_uint8(hx)
    hz_u = _matrix_to_uint8(hz)
    if hx_u.shape[1] != hz_u.shape[1]:
        raise ValueError("hx and hz must have the same number of columns")
    num_qubits = hx_u.shape[1]

    hx_block = np.concatenate([hx_u, np.zeros_like(hx_u, dtype=np.uint8)], axis=1)
    hz_block = np.concatenate([np.zeros_like(hz_u, dtype=np.uint8), hz_u], axis=1)
    rows = [hx_block, hz_block]

    if logicals_x is not None or logicals_z is not None:
        if logicals_x is None or logicals_z is None:
            raise ValueError("Both logicals_x and logicals_z must be provided for mixed logicals")
        logicals_x = _matrix_to_uint8(logicals_x)
        logicals_z = _matrix_to_uint8(logicals_z)
        if logicals_x.shape != logicals_z.shape:
            raise ValueError("logicals_x and logicals_z must have the same shape")
        if logicals_x.shape[1] != num_qubits:
            raise ValueError("logicals_x/logicals_z must match number of qubits")
        if logicals_phase is not None:
            _normalize_logical_phases(logicals_phase, logicals_x.shape[0])
        rows.append(np.concatenate([logicals_x, logicals_z], axis=1))

    return np.vstack(rows).astype(np.uint8)


def _stabilizer_rank_for_qubits(
    stabilizers: Sequence[stim.PauliString],
    qubits: np.ndarray,
) -> int:
    """Return rank of stabilizers restricted to a qubit subset."""
    if qubits.size == 0:
        return 0
    num_rows = len(stabilizers)
    num_cols = 2 * qubits.size
    mat = np.zeros((num_rows, num_cols), dtype=np.uint8)
    for i, stab in enumerate(stabilizers):
        xs, zs = stab.to_numpy()
        mat[i, : qubits.size] = xs[qubits]
        mat[i, qubits.size :] = zs[qubits]
    return int(mod2.rank(mat))


def _normalize_stabilizer_matrix(
    stabilizer_matrix: object,
    *,
    num_qubits: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Return (matrix, num_qubits) for an r x 2n stabilizer matrix."""
    mat = _matrix_to_uint8(stabilizer_matrix)
    if mat.ndim != 2:
        raise ValueError("stabilizer_matrix must be 2D")
    if mat.shape[1] == 0:
        if num_qubits is None:
            raise ValueError("num_qubits must be provided when stabilizer_matrix has no columns")
        return mat, int(num_qubits)
    if mat.shape[1] % 2 != 0:
        raise ValueError("stabilizer_matrix must have 2n columns")
    return mat, int(mat.shape[1] // 2)


def _stabilizer_rank_for_qubits_matrix(
    stabilizer_matrix: np.ndarray,
    qubits: np.ndarray,
    *,
    num_qubits: Optional[int] = None,
) -> int:
    """Return rank of stabilizer matrix restricted to a qubit subset."""
    mat, total_qubits = _normalize_stabilizer_matrix(
        stabilizer_matrix,
        num_qubits=num_qubits,
    )
    cols = np.concatenate([qubits, qubits + total_qubits])
    sub = mat[:, cols]
    return int(mod2.rank(sub))


def _stabilizer_matrix_commutes(stabilizer_matrix: np.ndarray) -> bool:
    """Return True if all generators commute under the symplectic product."""
    mat, total_qubits = _normalize_stabilizer_matrix(stabilizer_matrix)
    if mat.size == 0:
        return True
    x = mat[:, :total_qubits]
    z = mat[:, total_qubits:]
    sym = (x @ z.T + z @ x.T) % 2
    return bool(np.all(sym == 0))


def _find_noncommuting_xz_pairs(
    logicals_x: np.ndarray, logicals_z: np.ndarray
) -> List[Tuple[int, int]]:
    """Return (x_index, z_index) pairs that anticommute."""
    x_u = _matrix_to_uint8(logicals_x)
    z_u = _matrix_to_uint8(logicals_z)
    if x_u.shape[1] != z_u.shape[1]:
        raise ValueError("logicals_x/logicals_z must match number of qubits")
    pairs: List[Tuple[int, int]] = []
    for i in range(x_u.shape[0]):
        for j in range(z_u.shape[0]):
            if int((x_u[i] @ z_u[j].T) % 2):
                pairs.append((i, j))
    return pairs


def _logical_commutation_matrix(
    logicals_x: np.ndarray, logicals_z: np.ndarray
) -> np.ndarray:
    """Return commutation matrix M_ij = <X_i, Z_j> mod 2."""
    x_u = _matrix_to_uint8(logicals_x)
    z_u = _matrix_to_uint8(logicals_z)
    if x_u.shape[1] != z_u.shape[1]:
        raise ValueError("logicals_x/logicals_z must match number of qubits")
    comm = np.zeros((x_u.shape[0], z_u.shape[0]), dtype=np.uint8)
    for i in range(x_u.shape[0]):
        for j in range(z_u.shape[0]):
            comm[i, j] = int((x_u[i] @ z_u[j].T) % 2)
    return comm


def _logical_pairing(
    logicals_x: np.ndarray, logicals_z: np.ndarray
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, List[int]]], List[Tuple[int, List[int]]]]:
    """Return unique X/Z pairs plus ambiguous X and Z rows."""
    comm = _logical_commutation_matrix(logicals_x, logicals_z)
    x_ambiguous: List[Tuple[int, List[int]]] = []
    z_ambiguous: List[Tuple[int, List[int]]] = []
    pairs: List[Tuple[int, int]] = []

    for i in range(comm.shape[0]):
        js = [j for j in range(comm.shape[1]) if comm[i, j] == 1]
        if len(js) == 1:
            pairs.append((i, js[0]))
        else:
            x_ambiguous.append((i, js))

    for j in range(comm.shape[1]):
        is_ = [i for i in range(comm.shape[0]) if comm[i, j] == 1]
        if len(is_) != 1:
            z_ambiguous.append((j, is_))

    return pairs, x_ambiguous, z_ambiguous


def _normalize_subsystem(subsystem: Sequence[int], num_qubits: int) -> np.ndarray:
    """Return a sorted, unique, validated numpy array of qubit indices."""
    if isinstance(subsystem, np.ndarray) and subsystem.dtype == np.bool_:
        if subsystem.size != num_qubits:
            raise ValueError("Boolean subsystem mask has wrong length")
        return np.flatnonzero(subsystem)
    qubits = np.array(sorted(set(int(q) for q in subsystem)), dtype=np.int64)
    if qubits.size and (qubits.min() < 0 or qubits.max() >= num_qubits):
        raise ValueError("Subsystem qubit indices out of range")
    return qubits


def _parse_polynomial(poly: object):
    import sympy as sp

    if poly is None:
        return sp.Integer(0)
    if isinstance(poly, sp.Expr):
        return poly
    if isinstance(poly, np.ndarray):
        if poly.ndim == 2 and poly.shape[1] == 2:
            terms = [(int(ix), int(iy)) for ix, iy in poly]
            x, y = sp.symbols("x y")
            expr = sum(x**ix * y**iy for ix, iy in terms)
            return sp.expand(expr, modulus=2)
    if isinstance(poly, (list, tuple)):
        if poly and isinstance(poly[0], (list, tuple)) and len(poly[0]) == 2:
            if all(isinstance(v, (int, np.integer)) for v in poly[0]):
                terms = [(int(ix), int(iy)) for ix, iy in poly]
                x, y = sp.symbols("x y")
                expr = sum(x**ix * y**iy for ix, iy in terms)
                return sp.expand(expr, modulus=2)
    return sp.sympify(poly)


def logical_vector_from_polynomial_pair(
    f_poly: object,
    g_poly: object,
    l: int,
    m: int,
) -> np.ndarray:
    """Return a 2*l*m logical-Z vector from a polynomial pair [f, g]."""
    from minimal_ann_matrix import build_qubit_logical_indicator

    f_expr = _parse_polynomial(f_poly)
    g_expr = _parse_polynomial(g_poly)
    vec_f = build_qubit_logical_indicator(f_expr, l, m, block=0)["vector"]
    vec_g = build_qubit_logical_indicator(g_expr, l, m, block=1)["vector"]
    return (vec_f + vec_g) % 2


def _vector_to_poly_pair(
    vector: np.ndarray, monomials: Sequence[sp.Expr], l: int, m: int
) -> Tuple[sp.Expr, sp.Expr]:
    """Convert a 2*l*m vector into a polynomial pair (block0, block1)."""
    block_size = l * m
    vec = vector.astype(np.uint8, copy=False)
    if vec.size != 2 * block_size:
        raise ValueError("Vector length does not match 2*l*m")
    poly_a = vector_to_poly(vec[:block_size], list(monomials))
    poly_b = vector_to_poly(vec[block_size:], list(monomials))
    return poly_a, poly_b


def logicals_from_polynomial_pairs(
    pairs: Union[Sequence[Sequence[object]], Sequence[object]],
    l: int,
    m: int,
    *,
    pauli: str = "Z",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (logicals_x, logicals_z) from polynomial pairs [f, g].

    For a single logical, pass [f, g]. For multiple logicals, pass a list
    of pairs like [[f1, g1], [f2, g2], ...].
    """
    def _is_exponent_pair_list(obj: object) -> bool:
        if isinstance(obj, np.ndarray):
            return obj.ndim == 2 and obj.shape[1] == 2
        if isinstance(obj, (list, tuple)) and obj:
            if isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2:
                return all(isinstance(v, (int, np.integer)) for v in obj[0])
        return False

    if isinstance(pairs, (list, tuple)) and len(pairs) == 2:
        a, b = pairs[0], pairs[1]
        a_is_pair = isinstance(a, (list, tuple)) and len(a) == 2 and not _is_exponent_pair_list(a)
        b_is_pair = isinstance(b, (list, tuple)) and len(b) == 2 and not _is_exponent_pair_list(b)
        if a_is_pair and b_is_pair:
            poly_pairs = [tuple(a), tuple(b)]
        else:
            poly_pairs = [(a, b)]
    else:
        poly_pairs = [tuple(p) for p in pairs]

    vectors = [logical_vector_from_polynomial_pair(f, g, l, m) for f, g in poly_pairs]
    if vectors:
        mat = np.vstack(vectors).astype(np.uint8)
    else:
        mat = np.zeros((0, 2 * l * m), dtype=np.uint8)

    pauli = pauli.upper()
    if pauli == "Z":
        return np.zeros_like(mat), mat
    if pauli == "X":
        return mat, np.zeros_like(mat)
    raise ValueError("pauli must be 'X' or 'Z'")


def entanglement_entropy(
    tableau: stim.Tableau,
    subsystem: Sequence[int],
) -> int:
    """Return S(A) = rank(G_A_bar) - |A_bar| in bits."""
    stabilizers = tableau.to_stabilizers()
    num_qubits = len(stabilizers)
    qubits = _normalize_subsystem(subsystem, num_qubits)
    if qubits.size == 0 or qubits.size == num_qubits:
        return 0
    complement = np.setdiff1d(np.arange(num_qubits), qubits, assume_unique=True)
    rank = _stabilizer_rank_for_qubits(stabilizers, complement)
    return int(rank - complement.size)


# def entanglement_entropy_from_stabilizers(
#     stabilizers: Sequence[stim.PauliString],
#     subsystem: Sequence[int],
#     *,
#     num_qubits: Optional[int] = None,
# ) -> int:
#     """Return S(A) = rank(G_A_bar) - |A_bar| from generators.

#     The stabilizers should span a pure stabilizer state (n independent generators).
#     """
#     if stabilizers:
#         xs, _ = stabilizers[0].to_numpy()
#         total_qubits = int(xs.size)
#     else:
#         if num_qubits is None:
#             raise ValueError("num_qubits must be provided when stabilizers is empty")
#         total_qubits = int(num_qubits)
#     qubits = _normalize_subsystem(subsystem, total_qubits)
#     if qubits.size == 0 or qubits.size == total_qubits:
#         return 0
#     complement = np.setdiff1d(np.arange(total_qubits), qubits, assume_unique=True)
#     rank = _stabilizer_rank_for_qubits(stabilizers, complement)
#     return int(rank - complement.size)


def entanglement_entropy_from_stabilizer_matrix(
    stabilizer_matrix: np.ndarray,
    subsystem: Sequence[int],
    *,
    num_qubits: Optional[int] = None,
) -> int:
    """Return S(A) = rank(G_A_bar) - |A_bar| + (n - rank(S))  in bits from r x 2n matrix."""
    mat, total_qubits = _normalize_stabilizer_matrix(stabilizer_matrix, num_qubits=num_qubits)
    qubits = _normalize_subsystem(subsystem, total_qubits)
    complement = np.setdiff1d(np.arange(total_qubits), qubits, assume_unique=True)
    stab_part_rank = _stabilizer_rank_for_qubits_matrix(mat, complement, num_qubits=total_qubits)
    stab_rank = mod2.rank(stabilizer_matrix)
    return int(stab_part_rank - complement.size + total_qubits - stab_rank)


def mutual_information_from_stabilizer_matrix(
    stabilizer_matrix: np.ndarray,
    region_a: Sequence[int],
    region_b: Sequence[int],
    *,
    num_qubits: Optional[int] = None,
) -> int:
    """Return I(A:B) = S(A) + S(B) - S(AB) in bits from r x 2n matrix."""
    mat, total_qubits = _normalize_stabilizer_matrix(stabilizer_matrix, num_qubits=num_qubits)
    a = _normalize_subsystem(region_a, total_qubits)
    b = _normalize_subsystem(region_b, total_qubits)
    ab = np.union1d(a, b)
    s_a = entanglement_entropy_from_stabilizer_matrix(mat, a, num_qubits=total_qubits)
    s_b = entanglement_entropy_from_stabilizer_matrix(mat, b, num_qubits=total_qubits)
    s_ab = entanglement_entropy_from_stabilizer_matrix(mat, ab, num_qubits=total_qubits)
    # print(f"s_a={s_a}, s_b={s_b}, s_ab={s_ab}")

    return int(s_a + s_b - s_ab)


def coherent_information_from_stabilizer_matrix(
    stabilizer_matrix: np.ndarray,
    reference: Sequence[int],
    region_a: Sequence[int],
    *,
    num_qubits: Optional[int] = None,
) -> int:
    """Return I_c(R>A) = S(A) - S(RA) in bits from r x 2n matrix."""
    mat, total_qubits = _normalize_stabilizer_matrix(stabilizer_matrix, num_qubits=num_qubits)
    r = _normalize_subsystem(reference, total_qubits)
    a = _normalize_subsystem(region_a, total_qubits)
    if np.intersect1d(r, a).size:
        raise ValueError("reference and region_a must be disjoint")
    ra = np.union1d(r, a)
    s_a = entanglement_entropy_from_stabilizer_matrix(mat, a, num_qubits=total_qubits)
    s_ra = entanglement_entropy_from_stabilizer_matrix(mat, ra, num_qubits=total_qubits)
    return int(s_a - s_ra)


def synergy_from_stabilizer_matrix(
    stabilizer_matrix: np.ndarray,
    reference: Sequence[int],
    region_a: Sequence[int],
    region_b: Sequence[int],
    *,
    num_qubits: Optional[int] = None,
) -> int:
    """Return Sigma = I(R:AB) - max(I(R:A), I(R:B)) in bits from r x 2n matrix."""
    mat, total_qubits = _normalize_stabilizer_matrix(stabilizer_matrix, num_qubits=num_qubits)
    r = _normalize_subsystem(reference, total_qubits)
    a = _normalize_subsystem(region_a, total_qubits)
    b = _normalize_subsystem(region_b, total_qubits)
    if np.intersect1d(r, a).size or np.intersect1d(r, b).size:
        raise ValueError("reference must be disjoint from region_a and region_b")
    if np.intersect1d(a, b).size:
        raise ValueError("region_a and region_b must be disjoint")
    ab = np.union1d(a, b)
    i_rab = mutual_information_from_stabilizer_matrix(mat, r, ab, num_qubits=total_qubits)
    i_ra = mutual_information_from_stabilizer_matrix(mat, r, a, num_qubits=total_qubits)
    i_rb = mutual_information_from_stabilizer_matrix(mat, r, b, num_qubits=total_qubits)
    return int(i_rab - max(i_ra, i_rb))


def bb_qubit_index(l: int, m: int, *, block: int, x: int, y: int) -> int:
    """Return flattened qubit index for BB block/x/y coordinates."""
    if block not in (0, 1):
        raise ValueError("block must be 0 or 1 for BB codes")
    if not (0 <= x < l and 0 <= y < m):
        raise ValueError("x/y out of range")
    return block * (l * m) + x * m + y


def bb_subsystem_from_coords(
    l: int,
    m: int,
    coords: Iterable[Tuple[int, int, int]],
) -> List[int]:
    """Return qubit indices for (block, x, y) coordinates."""
    return [bb_qubit_index(l, m, block=b, x=x, y=y) for b, x, y in coords]


def bb_rectangle_subsystem(
    l: int,
    m: int,
    *,
    block: int,
    x_range: range,
    y_range: range,
) -> List[int]:
    """Return qubit indices for a rectangular region on one BB block."""
    coords = [(block, x, y) for x in x_range for y in y_range]
    return bb_subsystem_from_coords(l, m, coords)


def build_bb_tableau(
    spec: BBCodeSpec,
    *,
    logicals_x: Optional[np.ndarray] = None,
    logicals_z: Optional[np.ndarray] = None,
    logicals_fg: Optional[Union[Sequence[Sequence[object]], Sequence[object]]] = None,
    logicals_fg_pauli: str = "Z",
    logicals_phase: Optional[Sequence[complex]] = None,
    allow_underconstrained: bool = False,
) -> stim.Tableau:
    """Build a tableau for a BB code stabilizer state.

    If logicals_fg is provided, it is interpreted as [f, g] polynomial pairs.
    If logicals_x/logicals_z are provided, they are interpreted as mixed Pauli
    logical stabilizers. Use logicals_phase to set per-logical phases.
    """
    from bposd.css import css_code
    from bivariate_bicycle_codes import get_BB_Hx_Hz

    hx, hz = get_BB_Hx_Hz(spec.a_poly, spec.b_poly, spec.l, spec.m)
    code = css_code(hx=hx, hz=hz, name=f"BB_{spec.l}x{spec.m}")

    hx_u = _matrix_to_uint8(code.hx)
    hz_u = _matrix_to_uint8(code.hz)
    total_qubits = hx_u.shape[1]

    if logicals_fg is not None:
        if logicals_x is not None or logicals_z is not None:
            raise ValueError("Use logicals_fg or logicals_x/logicals_z, not both")
        logicals_x, logicals_z = logicals_from_polynomial_pairs(
            logicals_fg,
            spec.l,
            spec.m,
            pauli=logicals_fg_pauli,
        )

    if logicals_x is not None or logicals_z is not None:
        if logicals_x is None or logicals_z is None:
            raise ValueError("Both logicals_x and logicals_z must be provided for mixed logicals")
        logicals_x = _matrix_to_uint8(logicals_x)
        logicals_z = _matrix_to_uint8(logicals_z)
        if logicals_x.shape != logicals_z.shape:
            raise ValueError("logicals_x and logicals_z must have the same shape")

    stabilizers = _stabilizers_from_css(
        hx_u,
        hz_u,
        logicals_x=logicals_x,
        logicals_z=logicals_z,
        logicals_phase=logicals_phase,
    )

    tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=allow_underconstrained,
    )
    return tableau


def build_bb_stabilizers(
    spec: BBCodeSpec,
    *,
    logicals_x: Optional[np.ndarray] = None,
    logicals_z: Optional[np.ndarray] = None,
    logicals_fg: Optional[Union[Sequence[Sequence[object]], Sequence[object]]] = None,
    logicals_fg_pauli: str = "Z",
    logicals_phase: Optional[Sequence[complex]] = None,
) -> Tuple[List[stim.PauliString], int]:
    """Build BB code stabilizers without constructing a tableau."""
    from bposd.css import css_code
    from bivariate_bicycle_codes import get_BB_Hx_Hz

    hx, hz = get_BB_Hx_Hz(spec.a_poly, spec.b_poly, spec.l, spec.m)
    code = css_code(hx=hx, hz=hz, name=f"BB_{spec.l}x{spec.m}")

    hx_u = _matrix_to_uint8(code.hx)
    hz_u = _matrix_to_uint8(code.hz)
    total_qubits = int(hx_u.shape[1])

    if logicals_fg is not None:
        if logicals_x is not None or logicals_z is not None:
            raise ValueError("Use logicals_fg or logicals_x/logicals_z, not both")
        logicals_x, logicals_z = logicals_from_polynomial_pairs(
            logicals_fg,
            spec.l,
            spec.m,
            pauli=logicals_fg_pauli,
        )

    if logicals_x is not None or logicals_z is not None:
        if logicals_x is None or logicals_z is None:
            raise ValueError("Both logicals_x and logicals_z must be provided for mixed logicals")
        logicals_x = _matrix_to_uint8(logicals_x)
        logicals_z = _matrix_to_uint8(logicals_z)
        if logicals_x.shape != logicals_z.shape:
            raise ValueError("logicals_x and logicals_z must have the same shape")

    stabilizers = _stabilizers_from_css(
        hx_u,
        hz_u,
        logicals_x=logicals_x,
        logicals_z=logicals_z,
        logicals_phase=logicals_phase,
    )
    return stabilizers, total_qubits


def build_bb_stabilizer_matrix(
    spec: BBCodeSpec,
    *,
    logicals_x: Optional[np.ndarray] = None,
    logicals_z: Optional[np.ndarray] = None,
    logicals_fg: Optional[Union[Sequence[Sequence[object]], Sequence[object]]] = None,
    logicals_fg_pauli: str = "Z",
    logicals_phase: Optional[Sequence[complex]] = None,
) -> Tuple[np.ndarray, int]:
    """Build BB code stabilizers as an r x 2n binary matrix."""
    from bposd.css import css_code
    from bivariate_bicycle_codes import get_BB_Hx_Hz

    hx, hz = get_BB_Hx_Hz(spec.a_poly, spec.b_poly, spec.l, spec.m)
    code = css_code(hx=hx, hz=hz, name=f"BB_{spec.l}x{spec.m}")

    hx_u = _matrix_to_uint8(code.hx)
    hz_u = _matrix_to_uint8(code.hz)
    total_qubits = int(hx_u.shape[1])

    if logicals_fg is not None:
        if logicals_x is not None or logicals_z is not None:
            raise ValueError("Use logicals_fg or logicals_x/logicals_z, not both")
        logicals_x, logicals_z = logicals_from_polynomial_pairs(
            logicals_fg,
            spec.l,
            spec.m,
            pauli=logicals_fg_pauli,
        )

    if logicals_x is not None or logicals_z is not None:
        if logicals_x is None or logicals_z is None:
            raise ValueError("Both logicals_x and logicals_z must be provided for mixed logicals")
        logicals_x = _matrix_to_uint8(logicals_x)
        logicals_z = _matrix_to_uint8(logicals_z)
        if logicals_x.shape != logicals_z.shape:
            raise ValueError("logicals_x and logicals_z must have the same shape")

    stabilizer_matrix = _stabilizer_matrix_from_css(
        hx_u,
        hz_u,
        logicals_x=logicals_x,
        logicals_z=logicals_z,
        logicals_phase=logicals_phase,
    )
    return stabilizer_matrix, total_qubits


def _get_ring_groebner(l: int, m: int) -> sp.GroebnerBasis:
    """Return the Groebner basis for the ambient ring relations."""
    return sp.groebner([x**l + 1, y**m + 1], x, y, modulus=2, order="lex")


def apply_periodic_boundary(poly: sp.Expr, l: int, m: int) -> sp.Expr:
    """Reduce `poly` modulo x^l + 1 and y^m + 1."""

    gb = _get_ring_groebner(l, m)
    _, rem = gb.reduce(sp.expand(poly, modulus=2))
    return sp.expand(rem)



def get_entanglement_info_EPR_logical(
    spec: BBCodeSpec,
    hx_u: np.ndarray,
    hz_u: np.ndarray,
    logicals_Z: Sequence[Sequence[object]],
    logicals_dualX: Sequence[Sequence[object]],
    logicals_Z_others: Sequence[Sequence[object]],
    logicals_dualX_others: Sequence[Sequence[object]],
) -> np.ndarray:

    logicals_fg_Z=logicals_Z
    logicals_fg_X=logicals_dualX
    # print(logicals_fg_X)
    # print(logicals_fg_Z)
    id_op, logicals_z = logicals_from_polynomial_pairs(
        logicals_fg_Z,
        spec.l,
        spec.m,
        pauli="Z",
    )
    logicals_x, _ = logicals_from_polynomial_pairs(
        logicals_fg_X,
        spec.l,
        spec.m,
        pauli="X",
    )
    logicals_x_others, _ = logicals_from_polynomial_pairs(
        logicals_dualX_others,
        spec.l,
        spec.m,
        pauli="X",
    )
    id_op_others, logicals_z_others = logicals_from_polynomial_pairs(
        logicals_Z_others,
        spec.l,
        spec.m,
        pauli="Z",
    )
    # print(logicals_x)
    # print(logicals_z)

    extra_qubits = np.zeros((hx_u.shape[0], 1), dtype=np.uint8)
    # print(extra_qubits)
    # print(hx_u)
    hx_u_epr = np.hstack((hx_u, extra_qubits))
    hz_u_epr = np.hstack((hz_u, extra_qubits))
    # print(hx_u_epr)
    # print(hz_u_epr)
    # print(hz_u_epr.shape)
    id_op_epr = np.hstack((id_op[0,:], np.array([0],dtype=np.uint8)))
    logicals_x_epr = np.hstack((logicals_x[0,:], np.array([1],dtype=np.uint8)))
    logicals_z_epr = np.hstack((logicals_z[0,:], np.array([1],dtype=np.uint8)))

    
    extra_qubits_logical_z_others = np.zeros((logicals_z_others.shape[0], 1), dtype=np.uint8)
    logicals_z_others = np.hstack((logicals_z_others, extra_qubits_logical_z_others))
    id_op_others = np.hstack((id_op_others, extra_qubits_logical_z_others))
    extra_qubits_logical_x_others = np.zeros((logicals_x_others.shape[0], 1), dtype=np.uint8)
    logicals_x_others = np.hstack((logicals_x_others, extra_qubits_logical_x_others))
    id_op_x_others = np.zeros_like(logicals_x_others, dtype=np.uint8)

    # print(np.vstack((logicals_x_epr, id_op_epr)))
    # print(np.vstack((id_op_epr, logicals_z_epr)))

    state_logicals = [
        (
            np.vstack((logicals_x_epr, id_op_epr)),
            np.vstack((id_op_epr, logicals_z_epr)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_epr)),
            np.vstack((id_op_epr, id_op_epr)),
        ),
        (
            np.vstack((id_op_epr, id_op_epr)),
            np.vstack((logicals_z_epr, id_op_epr)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_others)),
            np.vstack((id_op_epr, logicals_z_others)),
        ),
        (
            np.vstack((id_op_epr, id_op_others)),
            np.vstack((logicals_z_epr, logicals_z_others)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_epr, id_op_others)),
            np.vstack((id_op_epr, logicals_z_epr, logicals_z_others)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_epr, logicals_x_others)),
            np.vstack((id_op_epr, logicals_z_epr, id_op_x_others)),
        ),
    ]

    block0 = bb_rectangle_subsystem(
        spec.l, spec.m, block=0, x_range=range(spec.l), y_range=range(spec.m)
    )
    block1 = bb_rectangle_subsystem(
        spec.l, spec.m, block=1, x_range=range(spec.l), y_range=range(spec.m)
    )
    ref_Q = [2 * spec.l * spec.m]

    mi_summary = np.zeros((len(state_logicals), 3), dtype=np.int64)
    for state_index, (state_logicals_x, state_logicals_z) in enumerate(state_logicals):
        stabilizer_matrix = _stabilizer_matrix_from_css(
            hx_u_epr,
            hz_u_epr,
            logicals_x=state_logicals_x,
            logicals_z=state_logicals_z,
        )
        if not _stabilizer_matrix_commutes(stabilizer_matrix):
            non_commuting = _find_noncommuting_xz_pairs(
                state_logicals_x, state_logicals_z
            )
            detail = f" non-commuting X/Z pairs: {non_commuting}" if non_commuting else ""
            raise ValueError(
                f"Non-commuting stabilizer generators for state type {state_index}.{detail}"
            )
        mi_summary[state_index, 0] = mutual_information_from_stabilizer_matrix(
            stabilizer_matrix, block0, ref_Q
        )
        mi_summary[state_index, 1] = mutual_information_from_stabilizer_matrix(
            stabilizer_matrix, block1, ref_Q
        )
        mi_summary[state_index, 2] = mutual_information_from_stabilizer_matrix(
            stabilizer_matrix, block0 + block1, ref_Q
        )

    return mi_summary


def get_entropy_info_EPR_logical(
    spec: BBCodeSpec,
    hx_u: np.ndarray,
    hz_u: np.ndarray,
    logicals_Z: Sequence[Sequence[object]],
    logicals_dualX: Sequence[Sequence[object]],
    logicals_Z_others: Sequence[Sequence[object]],
    logicals_dualX_others: Sequence[Sequence[object]],
) -> np.ndarray:
    logicals_fg_Z = logicals_Z
    logicals_fg_X = logicals_dualX
    id_op, logicals_z = logicals_from_polynomial_pairs(
        logicals_fg_Z,
        spec.l,
        spec.m,
        pauli="Z",
    )
    logicals_x, _ = logicals_from_polynomial_pairs(
        logicals_fg_X,
        spec.l,
        spec.m,
        pauli="X",
    )
    logicals_x_others, _ = logicals_from_polynomial_pairs(
        logicals_dualX_others,
        spec.l,
        spec.m,
        pauli="X",
    )
    id_op_others, logicals_z_others = logicals_from_polynomial_pairs(
        logicals_Z_others,
        spec.l,
        spec.m,
        pauli="Z",
    )

    extra_qubits = np.zeros((hx_u.shape[0], 1), dtype=np.uint8)
    hx_u_epr = np.hstack((hx_u, extra_qubits))
    hz_u_epr = np.hstack((hz_u, extra_qubits))
    id_op_epr = np.hstack((id_op[0, :], np.array([0], dtype=np.uint8)))
    logicals_x_epr = np.hstack((logicals_x[0, :], np.array([1], dtype=np.uint8)))
    logicals_z_epr = np.hstack((logicals_z[0, :], np.array([1], dtype=np.uint8)))

    extra_qubits_logical_z_others = np.zeros((logicals_z_others.shape[0], 1), dtype=np.uint8)
    logicals_z_others = np.hstack((logicals_z_others, extra_qubits_logical_z_others))
    id_op_others = np.hstack((id_op_others, extra_qubits_logical_z_others))
    extra_qubits_logical_x_others = np.zeros((logicals_x_others.shape[0], 1), dtype=np.uint8)
    logicals_x_others = np.hstack((logicals_x_others, extra_qubits_logical_x_others))
    id_op_x_others = np.zeros_like(logicals_x_others, dtype=np.uint8)

    state_logicals = [
        (
            np.vstack((logicals_x_epr, id_op_epr)),
            np.vstack((id_op_epr, logicals_z_epr)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_epr)),
            np.vstack((id_op_epr, id_op_epr)),
        ),
        (
            np.vstack((id_op_epr, id_op_epr)),
            np.vstack((logicals_z_epr, id_op_epr)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_others)),
            np.vstack((id_op_epr, logicals_z_others)),
        ),
        (
            np.vstack((id_op_epr, id_op_others)),
            np.vstack((logicals_z_epr, logicals_z_others)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_epr, id_op_others)),
            np.vstack((id_op_epr, logicals_z_epr, logicals_z_others)),
        ),
        (
            np.vstack((logicals_x_epr, id_op_epr, logicals_x_others)),
            np.vstack((id_op_epr, logicals_z_epr, id_op_x_others)),
        ),
    ]

    block0 = bb_rectangle_subsystem(
        spec.l, spec.m, block=0, x_range=range(spec.l), y_range=range(spec.m)
    )
    block1 = bb_rectangle_subsystem(
        spec.l, spec.m, block=1, x_range=range(spec.l), y_range=range(spec.m)
    )
    ref_Q = [2 * spec.l * spec.m]

    entropy_summary = np.zeros((len(state_logicals), 7), dtype=np.int64)
    for state_index, (state_logicals_x, state_logicals_z) in enumerate(state_logicals):
        stabilizer_matrix = _stabilizer_matrix_from_css(
            hx_u_epr,
            hz_u_epr,
            logicals_x=state_logicals_x,
            logicals_z=state_logicals_z,
        )
        if not _stabilizer_matrix_commutes(stabilizer_matrix):
            non_commuting = _find_noncommuting_xz_pairs(
                state_logicals_x, state_logicals_z
            )
            detail = f" non-commuting X/Z pairs: {non_commuting}" if non_commuting else ""
            raise ValueError(
                f"Non-commuting stabilizer generators for state type {state_index}.{detail}"
            )
        entropy_summary[state_index, 0] = entanglement_entropy_from_stabilizer_matrix(
            stabilizer_matrix, ref_Q
        )
        entropy_summary[state_index, 1] = entanglement_entropy_from_stabilizer_matrix(
            stabilizer_matrix, ref_Q + block0
        )
        entropy_summary[state_index, 2] = entanglement_entropy_from_stabilizer_matrix(
            stabilizer_matrix, ref_Q + block1
        )
        entropy_summary[state_index, 3] = entanglement_entropy_from_stabilizer_matrix(
            stabilizer_matrix, ref_Q + block0 + block1
        )
        entropy_summary[state_index, 4] = entanglement_entropy_from_stabilizer_matrix(
            stabilizer_matrix, block0
        )
        entropy_summary[state_index, 5] = entanglement_entropy_from_stabilizer_matrix(
            stabilizer_matrix, block1
        )
        entropy_summary[state_index, 6] = entanglement_entropy_from_stabilizer_matrix(
            stabilizer_matrix, block0 + block1
        )

    return entropy_summary


def print_state_tables(
    entries: Sequence[Tuple[str, np.ndarray]],
    *,
    state_labels: Optional[Sequence[str]] = None,
) -> None:
    state_count = entries[0][1].shape[0]
    if state_labels is None:
        state_labels = [f"State type {i}" for i in range(state_count)]
    label_width = max(len("logical"), max(len(label) for label, _ in entries))
    header = (
        f"{'logical':<{label_width}}  {'I(R:A)':>8}  {'I(R:B)':>8}  {'I(R:AB)':>9}"
    )
    state_type_name = [
        r"<S_Z, S_X, X_{L_j} X_R, Z_{L_j} Z_R>",
        r"<S_Z, S_X, X_{L_j} X_R>",
        r"<S_Z, S_X, Z_{L_j} Z_R>",
        r"<S_Z, S_X, X_{L_j} X_R, \prod_{i \neq j} Z_{L_{i} >",
        r"<S_Z, S_X, Z_{L_j} Z_R, \prod_{i \neq j} Z_{L_{i} >",
        r"<S_Z, S_X, X_{L_j} X_R, Z_{L_j} Z_R, \prod_{i \neq j} Z_{L_{i}>",
        r"<S_Z, S_X, X_{L_j} X_R, Z_{L_j} Z_R, \prod_{i \neq j} X_{L_i}>",
    ]
    for state_index in range(state_count):
        print(f"\n{state_labels[state_index]}")
        print(f"{state_type_name[state_index]}")
        print(header)
        for label, mi_summary in entries:
            i_ra, i_rb, i_rab = mi_summary[state_index]
            print(
                f"{label:<{label_width}}  {i_ra:8d}  {i_rb:8d}  {i_rab:9d}"
            )


def print_entropy_tables(
    entries: Sequence[Tuple[str, np.ndarray]],
    *,
    state_labels: Optional[Sequence[str]] = None,
) -> None:
    state_count = entries[0][1].shape[0]
    if state_labels is None:
        state_labels = [f"State type {i}" for i in range(state_count)]
    label_width = max(len("logical"), max(len(label) for label, _ in entries))
    header = (
        f"{'logical':<{label_width}}  {'S(R)':>6}  {'S(RA)':>6}  {'S(RB)':>6}"
        f"  {'S(RAB)':>7}  {'S(A)':>6}  {'S(B)':>6}  {'S(AB)':>7}"
    )
    state_type_name = [
        r"<S_Z, S_X, X_{L_j} X_R, Z_{L_j} Z_R>",
        r"<S_Z, S_X, X_{L_j} X_R>",
        r"<S_Z, S_X, Z_{L_j} Z_R>",
        r"<S_Z, S_X, X_{L_j} X_R, \prod_{i \neq j} Z_{L_{i} >",
        r"<S_Z, S_X, Z_{L_j} Z_R, \prod_{i \neq j} Z_{L_{i} >",
        r"<S_Z, S_X, X_{L_j} X_R, Z_{L_j} Z_R, \prod_{i \neq j} Z_{L_{i}>",
        r"<S_Z, S_X, X_{L_j} X_R, Z_{L_j} Z_R, \prod_{i \neq j} X_{L_i}>",
    ]
    for state_index in range(state_count):
        print(f"\n{state_labels[state_index]}")
        print(f"{state_type_name[state_index]}")
        print(header)
        for label, entropy_summary in entries:
            s_r, s_ra, s_rb, s_rab, s_a, s_b, s_ab = entropy_summary[state_index]
            print(
                f"{label:<{label_width}}  {s_r:6d}  {s_ra:6d}  {s_rb:6d}"
                f"  {s_rab:7d}  {s_a:6d}  {s_b:6d}  {s_ab:7d}"
            )

def transpose_poly(poly, l, m):
    # p^T(x,y) = p(x^{-1}, y^{-1}) with x^{-1}=x^{l-1}, y^{-1}=y^{m-1} on the torus
    poly_T = sp.expand(poly.subs({x: x**(l-1), y: y**(m-1)}))
    return apply_periodic_boundary(poly_T, l, m)

def main_mutual_info(l, m, a_terms, b_terms, logicals_all_z, logicals_all_dualX):
    print("BB_stab_tor module loaded.")

    ##################################################
    # Tor1 test mutual information, for both tor and ann logical
    ##################################################
    x, y = sp.symbols("x y")


    all_x_raw, _ = logicals_from_polynomial_pairs(logicals_all_dualX, l, m, pauli="X")
    _, all_z_raw = logicals_from_polynomial_pairs(logicals_all_z, l, m, pauli="Z")
    x_orth_info = orthogonalize_logical_x_matrix(all_x_raw, all_z_raw)

    monomials = _monomial_basis(l, m)
    logicals_all_dualX_orth = [
        _vector_to_poly_pair(row, monomials, l, m)
        for row in x_orth_info["x_orthogonal"]
    ]
    logicals_all_dualX = logicals_all_dualX_orth

    # print(logicals_ann_c_dualX)
    for (i, lg) in enumerate(logicals_all_z):
        print(f"logical Z[{i}] = {lg}")
    for (i, lg) in enumerate(logicals_all_dualX):
        print(f"logical X[{i}] = {lg}")



    spec = BBCodeSpec(a_poly=a_terms, b_poly=b_terms, l=l, m=m)
    hx, hz = get_BB_Hx_Hz(spec.a_poly, spec.b_poly, spec.l, spec.m)
    code = css_code(hx=hx, hz=hz, name=f"BB_{spec.l}x{spec.m}")

    hx_u = _matrix_to_uint8(code.hx)
    hz_u = _matrix_to_uint8(code.hz)
    total_qubits = hx_u.shape[1]

    all_x, _ = logicals_from_polynomial_pairs(logicals_all_dualX, l, m, pauli="X")
    _, all_z = logicals_from_polynomial_pairs(logicals_all_z, l, m, pauli="Z")
    x_orth_info = orthogonalize_logical_x_matrix(all_x, all_z)
    x_pairing_basis = (
        x_orth_info["x_orthogonal"]
        if x_orth_info["x_orthogonal"].size
        else all_x
    )
    print("orthogonal X dimension:", x_orth_info["x_orthogonal"].shape[0])
    print("rank_commutation:", x_orth_info["rank_commutation"])
    # print(x_pairing_basis)
    pairs, x_ambiguous, z_ambiguous = _logical_pairing(x_pairing_basis, all_z)
    comm = _logical_commutation_matrix(all_x, all_z)
    print(comm)
    if pairs:
        print("\nLogical X/Z pairing (unique):")
        for x_idx, z_idx in pairs:
            print(f"  X {x_idx} <-> Z {z_idx}")
    if x_ambiguous:
        print("\nLogical X with non-unique partners:")
        for x_idx, z_indices in x_ambiguous:
            print(f"  X {x_idx}: {z_indices}")
    if z_ambiguous:
        print("\nLogical Z with non-unique partners:")
        for z_idx, x_indices in z_ambiguous:
            print(f"  Z {z_idx}: {x_indices}")


        # Verify entanglement entropy calculations
    total_logicals = len(logicals_all_z)
    logical_entries: List[Tuple[str, np.ndarray]] = []
    entropy_entries: List[Tuple[str, np.ndarray]] = []

    mi_ann_c = []
    for i in range(len(logicals_all_z)):
        logicals_z_others = [logicals_all_z[j] for j in range(total_logicals) if j != i]
        logicals_x_others = [logicals_all_dualX[j] for j in range(total_logicals) if j != i]
        mi_summary = get_entanglement_info_EPR_logical(
            spec,
            hx_u,
            hz_u,
            logicals_all_z[i],
            logicals_all_dualX[i],
            logicals_z_others,
            logicals_x_others,
        )
        entropy_summary = get_entropy_info_EPR_logical(
            spec,
            hx_u,
            hz_u,
            logicals_all_z[i],
            logicals_all_dualX[i],
            logicals_z_others,
            logicals_x_others,
        )
        mi_ann_c.append(mi_summary)
        logical_entries.append((f"index {i}", mi_summary))
        entropy_entries.append((f"index {i}", entropy_summary))

    print_state_tables(logical_entries)
    print_entropy_tables(entropy_entries)

    return logical_entries, entropy_entries

#  it will be imported with * only if it’s included in __all__ (or doesn’t start with _ when __all__ is absent).

# __all__ = [
#     "BBCodeSpec",
#     "build_bb_stabilizers",
#     "build_bb_stabilizer_matrix",
#     "build_bb_tableau",
#     "bb_qubit_index",
#     "bb_subsystem_from_coords",
#     "bb_rectangle_subsystem",
#     "entanglement_entropy_from_stabilizer_matrix",
#     "mutual_information_from_stabilizer_matrix",
#     "coherent_information_from_stabilizer_matrix",
#     "synergy_from_stabilizer_matrix",
#     "logical_vector_from_polynomial_pair",
#     "logicals_from_polynomial_pairs",
# ]


if __name__ == "__main__":
    print("\nl=m=6, tor case")
    l = 6
    m = 6
    c_expr = sp.sympify("x**3 + y + y**2")
    d_expr = sp.sympify("y**3 + x + x**2")
    a_terms = [(3, 0), (0, 1), (0, 2)]
    b_terms = [(0, 3), (1, 0), (2, 0)]
    poly_P = sp.sympify("x**3*y**4 + x**3*y**3 + x**3*y**2 + x**3*y + y**2 + 1" )
    poly_Q = sp.sympify("x**4*y**3 + x**3*y**3 + x**2*y**3 + x**2 + x*y**3 + 1" )
    standard_polys = [sp.sympify("1"), sp.sympify("y"), sp.sympify("y**2"), sp.sympify("y**3"), sp.sympify("x"), sp.sympify("x*y")]
    logicals_ann_c = [[apply_periodic_boundary(poly_P * expr, l, m), 0] for expr in standard_polys]
    logicals_ann_d = [[0, apply_periodic_boundary(poly_Q * expr, l, m)] for expr in standard_polys]
    poly_tor1_c_multiplier = sp.sympify("x**4 + x**3 + x*y**2 + x*y + x + y**2")
    poly_tor1_d_multiplier = sp.sympify("x**2 + x*y + x + y**3 + y + 1")
    standard_polys_tor = [sp.sympify("1"), sp.sympify("x"), sp.sympify("x**2"), sp.sympify("x**3")]
    logicals_tor1 = [[apply_periodic_boundary(poly_tor1_c_multiplier * expr, l, m), apply_periodic_boundary(poly_tor1_d_multiplier * expr, l, m)] for expr in standard_polys_tor]


    poly_P_dual = sp.sympify("x**5*y**3 + x**4*y**3 + x**2 + x*y**3 + y**3 + 1" )
    poly_Q_dual = sp.sympify("x**3*y**5 + x**3*y**4 + x**3*y + x**3 + y**2 + 1" )
    standard_polys = [sp.sympify("1"), sp.sympify("y"), sp.sympify("y**2"), sp.sympify("y**3"), sp.sympify("x"), sp.sympify("x*y")]
    # standard_polys = [sp.sympify("1"), sp.sympify("y**5"), sp.sympify("y**10"), sp.sympify("y**15"), sp.sympify("x**5"), sp.sympify("x**5*y**5")]
    logicals_ann_c_dualX = [[apply_periodic_boundary(poly_P_dual * expr, l, m), 0] for expr in standard_polys]
    logicals_ann_d_dualX = [[0, apply_periodic_boundary(poly_Q_dual * expr, l, m)] for expr in standard_polys]
    poly_tor1_c_multiplier_dualX = sp.sympify("x**3*y**4 + x**2*y + x + 1")
    poly_tor1_d_multiplier_dualX = sp.sympify("x**4*y + x**3*y + x*y**5 + y**5 + y + 1")
    standard_polys_tor_dualX = [sp.sympify("1"), sp.sympify("x"), sp.sympify("x**2"), sp.sympify("x**3")]
    # standard_polys_tor_dualX = [sp.sympify("1"), sp.sympify("x**5"), sp.sympify("x**10"), sp.sympify("x**15")]
    logicals_tor1_dualX = [[apply_periodic_boundary(poly_tor1_c_multiplier_dualX * expr, l, m), apply_periodic_boundary(poly_tor1_d_multiplier_dualX * expr, l, m)] for expr in standard_polys_tor_dualX]

    # logicals_all_z = logicals_ann_c + logicals_ann_d + logicals_tor1
    # logicals_all_dualX = logicals_ann_c_dualX + logicals_ann_d_dualX + logicals_tor1_dualX

    # logicals_all_z = logicals_ann_c + logicals_ann_d[0:2] + logicals_tor1
    # logicals_all_dualX = logicals_ann_c_dualX + logicals_ann_d_dualX[0:2] + logicals_tor1_dualX

    logicals_all_z = logicals_ann_c + logicals_ann_d[0:2] + logicals_tor1
    logicals_all_dualX = logicals_ann_c_dualX + logicals_ann_d_dualX[0:6] + logicals_tor1_dualX

    main_mutual_info(l, m, a_terms, b_terms, logicals_all_z, logicals_all_dualX)

    # from minimal_ann_theory_logicalX import pair_css_logicals_from_polynomials

    # paired = pair_css_logicals_from_polynomials(
    #     f_str="x^3 + y + y^2",
    #     g_str="y^3 + x + x^2",
    #     l=6,
    #     m=6,
    # )

    # # Polynomial pairs (one-to-one)
    # logicals_all_z = paired["z_polys"]
    # logicals_all_dualX = paired["x_polys"]


    # main()
    # pass