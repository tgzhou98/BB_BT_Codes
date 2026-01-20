#!/usr/bin/env python3

"""
Semiperiodic annihilator generators for bivariate bicycle codes.

This module focuses on the semiperiodic fast path described in Section 4.4 of the
paper.  Given polynomials ``f`` and ``g`` defined over the ring
GF(2)[x, y] / (x^ℓ + 1, y^m + 1), the annihilators Ann(f) and Ann(g) are generated
by principal polynomials P and Q when the inputs are semiperiodic.  Logical
operators are then obtained by translating P and Q across the ℓ×m lattice.

All linear-algebra based routines for computing annihilators have been removed.
Only the semiperiodic construction is retained: Ann(f) (resp. Ann(g)) is obtained
from the translation orbit of P (resp. Q), and no quotient Ann(f)/(g·Ann(f)) is
computed.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
import ldpc.mod2 as mod2
from sympy import expand, symbols
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from minimal_ann_matrix import (
    _monomial_basis,
    _multiplication_matrix,
    _solve_linear_mod2,
    build_qubit_logical_indicator,
    build_torsion_logical_indicator,
    compute_tor_1,
    poly_to_vector,
    vector_to_poly,
    verify_logical_z_equivalence,
    verify_logical_x_equivalence,
)
from bivariate_bicycle_codes import get_BB_Hx_Hz
from bposd.css import css_code

x, y = symbols("x y")

_ring_groebner_cache: Dict[Tuple[int, int], sp.GroebnerBasis] = {}
_fg_ring_groebner_cache: Dict[Tuple[str, str, int, int], sp.GroebnerBasis] = {}


def _get_ring_groebner(l: int, m: int) -> sp.GroebnerBasis:
    """Return (and cache) the Groebner basis for the ambient ring relations."""

    key = (l, m)
    gb = _ring_groebner_cache.get(key)
    if gb is None:
        gb = sp.groebner([x**l + 1, y**m + 1], [x, y], modulus=2)
        _ring_groebner_cache[key] = gb
    return gb


def _get_fg_ring_groebner(
    f_poly: sp.Expr, g_poly: sp.Expr, l: int, m: int
) -> sp.GroebnerBasis:
    """Return Groebner basis for ⟨f, g, x^l+1, y^m+1⟩ over GF(2)."""

    key = (sp.srepr(f_poly), sp.srepr(g_poly), l, m)
    gb = _fg_ring_groebner_cache.get(key)
    if gb is None:
        generators = [expand(f_poly, modulus=2), expand(g_poly, modulus=2), x**l + 1, y**m + 1]
        gb = sp.groebner(generators, x, y, modulus=2, order="lex")
        _fg_ring_groebner_cache[key] = gb
    return gb


def apply_periodic_boundary(poly: sp.Expr, l: int, m: int) -> sp.Expr:
    """Reduce `poly` modulo x^l + 1 and y^m + 1."""

    gb = _get_ring_groebner(l, m)
    _, rem = gb.reduce(expand(poly))
    return expand(rem)


def invert_polynomial(poly: sp.Expr, l: int, m: int) -> sp.Expr:
    """Return poly(x^-1, y^-1) reduced in GF(2)[x,y]/(x^l+1, y^m+1)."""

    reduced = apply_periodic_boundary(poly, l, m)
    try:
        poly_mod = sp.Poly(reduced, x, y, modulus=2)
    except sp.PolynomialError:
        poly_mod = sp.Poly(expand(reduced, modulus=2), x, y, modulus=2)

    inverted = sp.Integer(0)
    for (exp_x, exp_y), coeff in poly_mod.terms():
        if int(coeff) % 2 != 1:
            continue
        inv_x = (-exp_x) % l
        inv_y = (-exp_y) % m
        inverted += x**inv_x * y**inv_y

    return apply_periodic_boundary(expand(inverted, modulus=2), l, m)


def try_semiperiodic_decomposition(
    poly: sp.Expr, l: int, m: int
) -> Optional[Tuple[int, sp.Expr, sp.Expr, Tuple[int, int]]]:
    """Try to express `poly` as a semiperiodic term x^k + ζ(y).

    Returns (k, ζ(y), monomial_prefactor, (ax, ay)) if successful, else None.
    """

    reduced = apply_periodic_boundary(poly, l, m)
    try:
        poly_mod = sp.Poly(reduced, x, y, modulus=2)
    except sp.PolynomialError:
        poly_mod = sp.Poly(expand(reduced, modulus=2), x, y, modulus=2)

    support = [
        (exp_x % l, exp_y % m)
        for (exp_x, exp_y), coeff in poly_mod.terms()
        if int(coeff) % 2 == 1
    ]
    if not support:
        return None

    unique_x = sorted({exp_x for exp_x, _ in support})

    if len(unique_x) == 1:
        base_x = unique_x[0]
        y_candidates = sorted(b for (a, b) in support if a == base_x)
        for y_shift in y_candidates:
            zeta_coeffs: Dict[int, int] = {}
            base_parity = 0
            for a, b in support:
                if a != base_x:
                    continue
                rel_y = (b - y_shift) % m
                if rel_y == 0:
                    base_parity ^= 1
                else:
                    zeta_coeffs[rel_y] = (zeta_coeffs.get(rel_y, 0) + 1) % 2
            if base_parity == 0:
                continue

            zeta = sp.Integer(0)
            for exp_y in sorted(zeta_coeffs):
                if zeta_coeffs[exp_y] % 2 == 1:
                    zeta += y**exp_y

            k = l
            prefactor = (x**base_x) * (y**y_shift)
            candidate = apply_periodic_boundary(prefactor * (x**k + zeta), l, m)
            if candidate != reduced:
                continue
            return k, expand(zeta, modulus=2), expand(prefactor, modulus=2), (0, 0)
        return None

    if len(unique_x) != 2:
        return None

    counts = {exp_x: 0 for exp_x in unique_x}
    for exp_x, _ in support:
        counts[exp_x] += 1

    ordered_candidates = sorted(unique_x, key=lambda exp: (-counts[exp], exp))
    for base_x in ordered_candidates:
        other_x = next(exp for exp in unique_x if exp != base_x)
        k = (other_x - base_x) % l
        if k == 0 or l % k != 0:
            continue

        off_terms = [(a, b) for (a, b) in support if a == other_x]
        if len(off_terms) != 1:
            continue
        y_shift = off_terms[0][1]

        if any(a not in (base_x, other_x) for (a, _) in support):
            continue

        zeta_coeffs: Dict[int, int] = {}
        for a, b in support:
            if a != base_x:
                continue
            rel_y = (b - y_shift) % m
            zeta_coeffs[rel_y] = (zeta_coeffs.get(rel_y, 0) + 1) % 2

        zeta = sp.Integer(0)
        for exp_y in sorted(zeta_coeffs):
            if zeta_coeffs[exp_y] % 2 == 1:
                zeta += y**exp_y

        prefactor = (x**base_x) * (y**y_shift)
        candidate = apply_periodic_boundary(prefactor * (x**k + zeta), l, m)

        if candidate != reduced:
            continue

        return k, expand(zeta, modulus=2), expand(prefactor, modulus=2), (0, 0)

    return None


def _reduce_univariate_mod(poly: sp.Expr, modulus: sp.Expr, *, variable: sp.Symbol) -> sp.Expr:
    """Reduce a univariate polynomial modulo `modulus` in GF(2)."""

    base_poly = sp.Poly(expand(poly, modulus=2), variable, modulus=2)
    mod_poly = sp.Poly(modulus, variable, modulus=2)
    remainder = base_poly.rem(mod_poly)
    return expand(remainder.as_expr(), modulus=2)


def _compute_cyclic_ann_generator(
    zeta: sp.Expr, k: int, l: int, m: int
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return (χ̂(y), g(y), χ_full(y)) for the semiperiodic construction."""

    if k <= 0 or l % k != 0:
        raise ValueError("Semiperiodic construction requires k > 0 dividing l")

    k_prime = l // k
    zeta_clean = expand(zeta, modulus=2)
    chi_full = expand(zeta_clean ** k_prime + 1, modulus=2)

    mod_poly = sp.Poly(y**m - 1, y, modulus=2)
    chi_full_poly = sp.Poly(chi_full, y, modulus=2)
    chi_gcd_poly = mod_poly.gcd(chi_full_poly)
    chi_gcd_expr = expand(chi_gcd_poly.as_expr(), modulus=2)
    if chi_gcd_expr == 0:
        chi_gcd_poly = sp.Poly(1, y, modulus=2)
        chi_gcd_expr = sp.Integer(1)

    g_poly = mod_poly.quo(chi_gcd_poly)
    g_expr = expand(g_poly.as_expr(), modulus=2)
    g_expr = _reduce_univariate_mod(g_expr, y**m - 1, variable=y)

    return chi_gcd_expr, g_expr, chi_full


def compute_semiperiodic_generator_P(
    k: int,
    zeta: sp.Expr,
    l: int,
    m: int,
    *,
    g_expr: Optional[sp.Expr] = None,
) -> sp.Expr:
    """Construct the generator P for a semiperiodic input with parameter k."""

    if k <= 0 or l % k != 0:
        raise ValueError("Semiperiodic construction requires k > 0 dividing l")

    if g_expr is None:
        _, g_expr, _ = _compute_cyclic_ann_generator(zeta, k, l, m)
    g_expr = expand(g_expr, modulus=2)

    mod_y = sp.Poly(y**m - 1, y, modulus=2)
    zeta_poly = sp.Poly(expand(zeta, modulus=2), y, modulus=2)
    zeta_power = sp.Poly(1, y, modulus=2)

    k_prime = l // k
    sum_expr = sp.Integer(0)

    for i in range(k_prime):
        x_power = (l - i * k) % l
        term_expr = expand((x**x_power) * zeta_power.as_expr(), modulus=2)
        sum_expr = expand(sum_expr + term_expr, modulus=2)
        zeta_power = (zeta_power * zeta_poly).rem(mod_y)

    P_expr = expand(sum_expr * g_expr, modulus=2)
    P_expr = apply_periodic_boundary(P_expr, l, m)
    return P_expr


def _detect_semiperiodic_orientation_x(poly: sp.Expr, l: int, m: int) -> Dict[str, Optional[sp.Expr]]:
    """Detect semiperiodic structure along the x-axis."""

    info: Dict[str, Optional[sp.Expr]] = {"used": False, "orientation": "x"}
    decomposition = try_semiperiodic_decomposition(poly, l, m)
    if decomposition is None:
        info["reason"] = "not semiperiodic along x"
        return info

    k, zeta, prefactor, shift = decomposition
    if k <= 0 or l % k != 0:
        info["reason"] = "invalid k in semiperiodic decomposition"
        info["k"] = k
        return info

    chi_gcd, g_expr, chi_full = _compute_cyclic_ann_generator(zeta, k, l, m)
    P_expr = compute_semiperiodic_generator_P(k, zeta, l, m, g_expr=g_expr)

    info.update(
        {
            "used": True,
            "generator": P_expr,
            "k": k,
            "k_prime": l // k,
            "zeta": expand(zeta, modulus=2),
            "monomial_prefactor": prefactor,
            "shift": shift,
            "chi_reduced": chi_gcd,
            "chi_full": chi_full,
            "cyclic_generator": g_expr,
        }
    )
    return info


def _detect_semiperiodic_orientation_y(poly: sp.Expr, l: int, m: int) -> Dict[str, Optional[sp.Expr]]:
    """Detect semiperiodic structure along the y-axis by swapping variables."""

    swapped_poly = expand(poly.subs({x: y, y: x}, simultaneous=True))
    info_swapped = _detect_semiperiodic_orientation_x(swapped_poly, m, l)
    info: Dict[str, Optional[sp.Expr]] = {"used": False, "orientation": "y"}

    if not info_swapped.get("used"):
        info["reason"] = info_swapped.get("reason", "not semiperiodic along y")
        return info

    generator_swapped = info_swapped["generator"]
    generator_orig = expand(generator_swapped.subs({x: y, y: x}, simultaneous=True))
    generator_orig = apply_periodic_boundary(generator_orig, l, m)

    info.update(
        {
            "used": True,
            "generator": generator_orig,
            "k": info_swapped.get("k"),
            "k_prime": info_swapped.get("k_prime"),
            "zeta": expand(info_swapped["zeta"].subs({x: y, y: x}, simultaneous=True), modulus=2)
            if info_swapped.get("zeta") is not None
            else None,
            "monomial_prefactor": expand(
                info_swapped["monomial_prefactor"].subs({x: y, y: x}, simultaneous=True)
            )
            if info_swapped.get("monomial_prefactor") is not None
            else None,
            "chi_reduced": expand(
                info_swapped["chi_reduced"].subs({x: y, y: x}, simultaneous=True), modulus=2
            )
            if info_swapped.get("chi_reduced") is not None
            else None,
            "chi_full": expand(
                info_swapped["chi_full"].subs({x: y, y: x}, simultaneous=True), modulus=2
            )
            if info_swapped.get("chi_full") is not None
            else None,
            "cyclic_generator": expand(
                info_swapped["cyclic_generator"].subs({x: y, y: x}, simultaneous=True), modulus=2
            )
            if info_swapped.get("cyclic_generator") is not None
            else None,
            "shift": (
                info_swapped["shift"][1] % l if info_swapped.get("shift") else 0,
                info_swapped["shift"][0] % m if info_swapped.get("shift") else 0,
            ),
        }
    )
    return info


def detect_semiperiodic_generator(poly: sp.Expr, l: int, m: int) -> Dict[str, Optional[sp.Expr]]:
    """Detect semiperiodic structure along either axis."""

    info_x = _detect_semiperiodic_orientation_x(poly, l, m)
    if info_x.get("used"):
        return info_x

    info_y = _detect_semiperiodic_orientation_y(poly, l, m)
    if info_y.get("used"):
        return info_y

    merged: Dict[str, Optional[sp.Expr]] = {"used": False}
    reasons = []
    for reason in (info_x.get("reason"), info_y.get("reason")):
        if reason:
            reasons.append(reason)
    if reasons:
        merged["reason"] = "; ".join(reasons)
    return merged


def _leading_standard_monomial(
    poly: sp.Expr, gb: sp.GroebnerBasis
) -> Tuple[Optional[Tuple[int, int]], sp.Expr]:
    """Return (monomial exponents, remainder) for `poly` mod the given Groebner basis."""

    _, remainder = gb.reduce(expand(poly))
    if remainder == 0:
        return None, sp.Integer(0)
    poly_mod = sp.Poly(remainder, x, y, modulus=2)
    lm = poly_mod.LM(order="lex")
    return (lm[0], lm[1]), expand(remainder, modulus=2)


def translation_orbit(
    poly: sp.Expr,
    l: int,
    m: int,
    *,
    quotient_gb: sp.GroebnerBasis,
) -> List[Dict[str, object]]:
    """Enumerate distinct lattice translates of `poly` keyed by standard monomials."""

    base = apply_periodic_boundary(poly, l, m)
    if base == 0:
        return []

    orbit: Dict[Tuple[int, int], Dict[str, object]] = {}

    for ax in range(l):
        for ay in range(m):
            translated = apply_periodic_boundary(base * (x**ax) * (y**ay), l, m)
            if translated == 0:
                continue
            monomial_data = _leading_standard_monomial((x**ax) * (y**ay), quotient_gb)
            standard_monomial, remainder = monomial_data
            key = standard_monomial if standard_monomial is not None else (-1, -1)
            if key in orbit:
                continue
            orbit[key] = {
                "poly": translated,
                "translation": (ax % l, ay % m),
                "standard_monomial": standard_monomial,
                "remainder": remainder,
            }

    sorted_items = sorted(orbit.items(), key=lambda item: item[0])
    return [entry for _, entry in sorted_items]


def _to_uint8_matrix(data: Any) -> np.ndarray:
    """Return a 2D uint8 matrix reduced modulo 2."""

    if hasattr(data, "toarray"):
        dense = data.toarray()
    else:
        dense = np.asarray(data)
    dense = (dense.astype(np.uint8) % 2)
    if dense.ndim == 1:
        if dense.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        dense = dense.reshape(1, -1)
    return dense


def _row_basis(matrix: np.ndarray) -> np.ndarray:
    """Return a row basis for the given matrix over GF(2)."""

    if matrix.size == 0:
        return np.zeros((0, matrix.shape[1]), dtype=np.uint8)
    return _to_uint8_matrix(mod2.row_basis(matrix))


def _invert_gf2_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of a square GF(2) matrix."""

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("GF(2) matrix inversion requires a square matrix")

    size = matrix.shape[0]
    inv_cols = []
    for idx in range(size):
        e_vec = np.zeros(size, dtype=np.uint8)
        e_vec[idx] = 1
        col = _solve_linear_mod2(matrix, e_vec)
        if col is None:
            raise ValueError("GF(2) matrix is singular")
        inv_cols.append(col.astype(np.uint8))
    return np.stack(inv_cols, axis=1) % 2


def logicals_to_matrix(logicals: Dict[str, object], *, num_qubits: Optional[int] = None) -> np.ndarray:
    """Return a stacked logical matrix from the logicals dict."""

    ops = logicals.get("block1", []) + logicals.get("block2", []) + logicals.get("torsion", [])
    if ops:
        return np.vstack([entry["vector"].astype(np.uint8) for entry in ops])
    if num_qubits is None:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.zeros((0, num_qubits), dtype=np.uint8)


def _vector_to_poly_pair(
    vector: np.ndarray, monomials: List[sp.Expr], l: int, m: int
) -> Tuple[sp.Expr, sp.Expr]:
    """Convert a 2*l*m vector into a pair of polynomials for each block."""

    block_size = l * m
    vec = vector.astype(np.uint8, copy=False)
    if vec.size != 2 * block_size:
        raise ValueError("Vector length does not match 2*l*m")
    poly_a = vector_to_poly(vec[:block_size], monomials)
    poly_b = vector_to_poly(vec[block_size:], monomials)
    return poly_a, poly_b


def pair_css_logicals_from_code(
    code: css_code,
    l: int,
    m: int,
    *,
    as_polys: bool = True,
) -> Dict[str, object]:
    """Return one-to-one paired logical X/Z operators for a CSS code."""

    lx_matrix = _to_uint8_matrix(code.lx)
    lz_matrix = _to_uint8_matrix(code.lz)
    x_orth_info = orthogonalize_logical_x_matrix(lx_matrix, lz_matrix)
    x_orth = x_orth_info["x_orthogonal"]
    z_selected = x_orth_info["z_selected"]

    if not as_polys:
        return {
            "x_matrix": x_orth,
            "z_matrix": z_selected,
            "pivot_columns": x_orth_info["pivot_columns"],
            "rank_commutation": x_orth_info["rank_commutation"],
        }

    expected_cols = 2 * l * m
    if x_orth.size and x_orth.shape[1] != expected_cols:
        raise ValueError("Logical X length does not match 2*l*m for polynomial conversion")
    if z_selected.size and z_selected.shape[1] != expected_cols:
        raise ValueError("Logical Z length does not match 2*l*m for polynomial conversion")

    monomials = _monomial_basis(l, m)
    x_polys = [_vector_to_poly_pair(row, monomials, l, m) for row in x_orth]
    z_polys = [_vector_to_poly_pair(row, monomials, l, m) for row in z_selected]

    return {
        "x_matrix": x_orth,
        "z_matrix": z_selected,
        "x_polys": x_polys,
        "z_polys": z_polys,
        "pivot_columns": x_orth_info["pivot_columns"],
        "rank_commutation": x_orth_info["rank_commutation"],
    }


def pair_css_logicals_from_polynomials(
    f_str: Union[str, sp.Expr],
    g_str: Union[str, sp.Expr],
    l: int,
    m: int,
    *,
    as_polys: bool = True,
) -> Dict[str, object]:
    """Return paired logicals for the BB CSS code defined by f,g."""

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)

    monomials = _monomial_basis(l, m)
    f_terms = [
        (int(idx // m) % l, int(idx % m))
        for idx in np.where(poly_to_vector(f_poly, monomials, l, m) == 1)[0]
    ]
    g_terms = [
        (int(idx // m) % l, int(idx % m))
        for idx in np.where(poly_to_vector(g_poly, monomials, l, m) == 1)[0]
    ]

    hx, hz = get_BB_Hx_Hz(f_terms, g_terms, l, m)
    code = css_code(hx=hx, hz=hz, name=f"BB_{l}x{m}")
    return pair_css_logicals_from_code(code, l, m, as_polys=as_polys)


def _build_logicals_from_generators(
    generators: Dict[str, object],
    tor1_data: Dict[str, object],
    f_poly: sp.Expr,
    g_poly: sp.Expr,
    l: int,
    m: int,
) -> Dict[str, object]:
    """Build logical operator entries for Ann(f), Ann(g), and Tor₁."""

    monomials = _monomial_basis(l, m)
    mult_matrix_f = _multiplication_matrix(f_poly, monomials, l, m)
    mult_matrix_g = _multiplication_matrix(g_poly, monomials, l, m)

    block1_ops: List[Dict[str, object]] = []
    for idx, entry in enumerate(generators["ann_f_orbit"]):
        poly = entry["poly"]
        indicator = build_qubit_logical_indicator(poly, l, m, block=0)
        indicator["index"] = idx
        indicator["source"] = "Ann(f)"
        indicator["orbit_translation"] = entry.get("translation")
        block1_ops.append(indicator)

    block2_ops: List[Dict[str, object]] = []
    for idx, entry in enumerate(generators["ann_g_orbit"]):
        poly = entry["poly"]
        indicator = build_qubit_logical_indicator(poly, l, m, block=1)
        indicator["index"] = idx
        indicator["source"] = "Ann(g)"
        indicator["orbit_translation"] = entry.get("translation")
        block2_ops.append(indicator)

    torsion_ops = _build_torsion_orbit_logicals(
        tor1_data,
        f_poly,
        g_poly,
        l,
        m,
        monomials,
        mult_matrix_f,
        mult_matrix_g,
    )
    if torsion_ops is None:
        torsion_ops = []
        for idx, poly in enumerate(tor1_data.get("tor_basis", [])):
            tor_vec = poly_to_vector(poly, monomials, l, m)
            f_solution = _solve_linear_mod2(mult_matrix_f, tor_vec)
            g_solution = _solve_linear_mod2(mult_matrix_g, tor_vec)
            if f_solution is None or g_solution is None:
                raise ValueError(
                    "Failed to express Tor₁ generator as an f- and g-multiple in the ambient ring"
                )

            f_multiplier = vector_to_poly(f_solution, monomials)
            g_multiplier = vector_to_poly(g_solution, monomials)
            indicator = build_torsion_logical_indicator(poly, f_multiplier, g_multiplier, l, m)
            indicator["index"] = idx
            torsion_ops.append(indicator)

    return {
        "block1": block1_ops,
        "block2": block2_ops,
        "torsion": torsion_ops,
        "tor_details": tor1_data,
        "tor2_details": None,
    }


def _build_torsion_orbit_logicals(
    tor1_data: Dict[str, object],
    f_poly: sp.Expr,
    g_poly: sp.Expr,
    l: int,
    m: int,
    monomials: List[sp.Expr],
    mult_matrix_f: np.ndarray,
    mult_matrix_g: np.ndarray,
) -> Optional[List[Dict[str, object]]]:
    """Return torsion logicals generated by x-orbit shifts of one Tor₁ element."""

    tor_basis = tor1_data.get("tor_basis", [])
    if not tor_basis:
        return None

    tor_vec = poly_to_vector(tor_basis[0], monomials, l, m)
    f_solution = _solve_linear_mod2(mult_matrix_f, tor_vec)
    g_solution = _solve_linear_mod2(mult_matrix_g, tor_vec)
    if f_solution is None or g_solution is None:
        return None

    f_seed = vector_to_poly(f_solution, monomials)
    g_seed = vector_to_poly(g_solution, monomials)

    shifts = tor1_data.get("orbit_shifts")
    dimension = tor1_data.get("dimension")
    if shifts is None:
        if dimension is None:
            dimension = len(tor_basis)
        shifts = [x**idx for idx in range(int(dimension))]
    elif dimension is None:
        dimension = len(shifts)

    torsion_ops: List[Dict[str, object]] = []
    seen_keys: set[Tuple[str, str]] = set()
    for shift in shifts:
        shift_expr = sp.sympify(shift)
        f_multiplier = apply_periodic_boundary(f_seed * shift_expr, l, m)
        g_multiplier = apply_periodic_boundary(g_seed * shift_expr, l, m)
        torsion_poly = apply_periodic_boundary(f_poly * f_multiplier, l, m)
        key = (sp.srepr(f_multiplier), sp.srepr(g_multiplier))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        indicator = build_torsion_logical_indicator(
            torsion_poly, f_multiplier, g_multiplier, l, m
        )
        indicator["index"] = len(torsion_ops)
        indicator["orbit_shift"] = shift_expr
        torsion_ops.append(indicator)

    if not torsion_ops:
        return None
    if dimension is not None and len(torsion_ops) < int(dimension):
        return None
    return torsion_ops


def orthogonalize_logical_x_matrix(
    logicals_x: np.ndarray,
    logicals_z: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return a logical-X basis paired against logical-Z when possible."""

    x_matrix = _to_uint8_matrix(logicals_x)
    z_matrix = _to_uint8_matrix(logicals_z)

    if x_matrix.shape[1] != z_matrix.shape[1]:
        raise ValueError("Logical X/Z matrices must have the same number of qubits")

    x_basis = _row_basis(x_matrix)
    z_basis = _to_uint8_matrix(z_matrix)

    if x_basis.shape[0] == 0:
        return {
            "x_basis": x_basis,
            "z_basis": z_basis,
            "commutation": np.zeros((0, 0), dtype=np.uint8),
            "transform": np.zeros((0, 0), dtype=np.uint8),
            "x_orthogonal": x_basis,
            "commutation_orthogonal": np.zeros((0, 0), dtype=np.uint8),
            "z_selected": np.zeros((0, z_basis.shape[1]), dtype=np.uint8),
            "pivot_columns": np.zeros((0,), dtype=int),
            "rank_commutation": 0,
        }

    comm = (x_basis @ z_basis.T) % 2
    rank_comm = mod2.rank(comm)
    if rank_comm == 0:
        return {
            "x_basis": x_basis,
            "z_basis": z_basis,
            "commutation": comm,
            "transform": np.zeros((0, 0), dtype=np.uint8),
            "x_orthogonal": np.zeros((0, x_basis.shape[1]), dtype=np.uint8),
            "commutation_orthogonal": np.zeros((0, 0), dtype=np.uint8),
            "z_selected": np.zeros((0, z_basis.shape[1]), dtype=np.uint8),
            "pivot_columns": np.zeros((0,), dtype=int),
            "rank_commutation": rank_comm,
        }

    _, _, transform, pivot_cols = mod2.row_echelon(comm, full=True)
    x_reduced = (transform @ x_basis) % 2
    x_selected = x_reduced[:rank_comm]
    z_selected = z_basis[pivot_cols]
    comm_selected = (x_selected @ z_selected.T) % 2
    transform_selected = _invert_gf2_matrix(comm_selected)
    x_orth = (transform_selected @ x_selected) % 2
    comm_orth = (x_orth @ z_selected.T) % 2

    return {
        "x_basis": x_basis,
        "z_basis": z_basis,
        "commutation": comm,
        "transform": transform,
        "x_orthogonal": x_orth,
        "commutation_orthogonal": comm_orth,
        "z_selected": z_selected,
        "pivot_columns": np.asarray(pivot_cols, dtype=int),
        "rank_commutation": rank_comm,
    }


def compute_ann_generators(
    f_str: Union[str, sp.Expr],
    g_str: Union[str, sp.Expr],
    l: int,
    m: int,
    *,
    reduce_by_fg: bool = True,
) -> Dict[str, object]:
    """Compute semiperiodic generators P and Q together with their translation orbits.

    Set reduce_by_fg=False to keep the full translation lattice (no (f, g) quotient).
    """

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)

    info_f = detect_semiperiodic_generator(f_poly, l, m)
    if not info_f.get("used"):
        reason = info_f.get("reason", "f is not semiperiodic")
        raise ValueError(f"Cannot build Ann(f): {reason}")

    info_g = detect_semiperiodic_generator(g_poly, l, m)
    if not info_g.get("used"):
        reason = info_g.get("reason", "g is not semiperiodic")
        raise ValueError(f"Cannot build Ann(g): {reason}")

    P = info_f["generator"]
    Q = info_g["generator"]

    quotient_gb = (
        _get_fg_ring_groebner(f_poly, g_poly, l, m)
        if reduce_by_fg
        else _get_ring_groebner(l, m)
    )

    ann_f_orbit = (
        translation_orbit(P, l, m, quotient_gb=quotient_gb) if P is not None else []
    )
    ann_g_orbit = (
        translation_orbit(Q, l, m, quotient_gb=quotient_gb) if Q is not None else []
    )

    return {
        "P": P,
        "Q": Q,
        "ann_f_orbit": ann_f_orbit,
        "ann_g_orbit": ann_g_orbit,
        "semiperiodic_f": info_f,
        "semiperiodic_g": info_g,
    }


def compute_semiperiodic_tor1(
    f_str: Union[str, sp.Expr],
    g_str: Union[str, sp.Expr],
    l: int,
    m: int,
) -> Dict[str, object]:
    """Return Tor₁ data using the Gaussian-elimination helper."""

    return compute_tor_1(f_str, g_str, l, m)


def compare_semiperiodic_with_css(
    f_str: Union[str, sp.Expr],
    g_str: Union[str, sp.Expr],
    l: int,
    m: int,
    *,
    generators: Optional[Dict[str, object]] = None,
    tor1_data: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Compare semiperiodic Ann(f), Ann(g), and Tor₁ logicals against css_code outputs."""

    if generators is None:
        generators = compute_ann_generators(f_str, g_str, l, m, reduce_by_fg=True)
    if tor1_data is None:
        tor1_data = compute_semiperiodic_tor1(f_str, g_str, l, m)

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)

    z_logicals = _build_logicals_from_generators(
        generators,
        tor1_data,
        f_poly,
        g_poly,
        l,
        m,
    )
    block1_ops = z_logicals["block1"]
    block2_ops = z_logicals["block2"]
    torsion_ops = z_logicals["torsion"]

    print("Logical Z torsion operators (Tor_1):")

    print("Tor_1 details:")
    if tor1_data:
        dimension = tor1_data.get("dimension")
        if dimension is not None:
            print(f"  dimension = {dimension}")
        tor_basis = tor1_data.get("tor_basis", [])
        if tor_basis:
            print("  tor_basis:")
            for idx, basis_poly in enumerate(tor_basis):
                print(f"    [{idx}] {basis_poly}")
                if idx < len(torsion_ops):
                    entry = torsion_ops[idx]
                    f_mult = entry.get("f_multiplier")
                    g_mult = entry.get("g_multiplier")
                    if f_mult is not None and g_mult is not None:
                        print(f"      f_multiplier = {f_mult}")
                        print(f"      g_multiplier = {g_mult}")
        else:
            print("  tor_basis: (empty)")
    else:
        print("  No Tor_1 data was provided.")

    logicals: Dict[str, object] = z_logicals

    equivalence = verify_logical_z_equivalence(
        f_str,
        g_str,
        l,
        m,
        logicals=logicals,
    )

    x_f_poly = invert_polynomial(g_poly, l, m)
    x_g_poly = invert_polynomial(f_poly, l, m)
    x_generators = compute_ann_generators(x_f_poly, x_g_poly, l, m, reduce_by_fg=True)
    x_tor1_data = compute_semiperiodic_tor1(x_f_poly, x_g_poly, l, m)
    x_logicals = _build_logicals_from_generators(
        x_generators,
        x_tor1_data,
        x_f_poly,
        x_g_poly,
        l,
        m,
    )

    z_matrix = logicals_to_matrix(z_logicals, num_qubits=2 * l * m)
    x_matrix = logicals_to_matrix(x_logicals, num_qubits=z_matrix.shape[1])
    x_orth = orthogonalize_logical_x_matrix(x_matrix, z_matrix)
    x_logicals["x_orthogonal"] = x_orth["x_orthogonal"]
    logicals["x_orthogonal"] = x_orth["x_orthogonal"]

    print("Logical X (orthogonalized) matrix:")
    if x_orth["x_orthogonal"].size:
        for idx, row in enumerate(x_orth["x_orthogonal"]):
            print(f"  X_orth[{idx}] = {row}")
    else:
        print("  (empty)")

    print("Logical Z/X pairing (orthogonalized):")
    pivot_cols = x_orth.get("pivot_columns", np.zeros((0,), dtype=int))
    if pivot_cols.size and x_orth["x_orthogonal"].size:
        monomials = _monomial_basis(l, m)
        for idx, z_idx in enumerate(pivot_cols):
            z_idx_int = int(z_idx)
            print(f"  X_orth[{idx}] <-> Z[{z_idx_int}]")
            if z_matrix.size:
                z_poly_a, z_poly_b = _vector_to_poly_pair(
                    z_matrix[z_idx_int], monomials, l, m
                )
                print(f"    Z[{z_idx_int}] = [{z_poly_a}, {z_poly_b}]")
    else:
        print("  (no paired logicals)")

    print("Logical X (orthogonalized) polynomials:")
    if x_orth["x_orthogonal"].size:
        monomials = _monomial_basis(l, m)
        for idx, row in enumerate(x_orth["x_orthogonal"]):
            poly_a, poly_b = _vector_to_poly_pair(row, monomials, l, m)
            print(f"  X_orth[{idx}] = [{poly_a}, {poly_b}]")
    else:
        print("  (empty)")

    print(
        "  CSS & Poly:  rank(css ∪ Z)={rank_css}, rank(poly ∪ Z)={rank_poly}, rank(css ∪ poly ∪ Z)={rank_union}, rank(Z stabilizer)={rank_z}".format(
            rank_css=equivalence["rank_css_space"],
            rank_poly=equivalence["rank_poly_space"],
            rank_union=equivalence["rank_union_space"],
            rank_z=equivalence["rank_z_stabilizer"],
        )
    )
    print(
        "  Poly:        rank(block1 ∪ Z)={rank_b1}, rank(block2 ∪ Z)={rank_b2},  rank(block1 ∪ block2 ∪ Z)={rank_b12}, rank(torsion 1 ∪ Z)={rank_tor}".format(
            rank_b1=equivalence["rank_block1_z_union"],
            rank_b2=equivalence["rank_block2_z_union"],
            rank_b12=equivalence["rank_block12_z_union"],
            rank_tor=equivalence["rank_torsion_z_union"],
        )
    )
    print(
        "  Tor_2:       rank={rank_t2}, rank(Tor_2 ∪ Z)={rank_t2u}, rank(Tor_2 ∩ Z)={rank_t2i}".format(
            rank_t2=equivalence["rank_tor2"],
            rank_t2u=equivalence["rank_tor2_z_union"],
            rank_t2i=equivalence["rank_tor2_z_intersection"],
        )
    )

    x_equivalence = verify_logical_x_equivalence(
        f_str,
        g_str,
        l,
        m,
        logicals=x_logicals,
    )
    print(
        "  CSS & Poly:  rank(css ∪ X)={rank_css}, rank(poly ∪ X)={rank_poly}, "
        "rank(css ∪ poly ∪ X)={rank_union}, rank(X stabilizer)={rank_x}".format(
            rank_css=x_equivalence["rank_css_space"],
            rank_poly=x_equivalence["rank_poly_space"],
            rank_union=x_equivalence["rank_union_space"],
            rank_x=x_equivalence["rank_x_stabilizer"],
        )
    )
    print(
        "  Poly:        rank(block1 ∪ X)={rank_b1}, rank(block2 ∪ X)={rank_b2},  "
        "rank(block1 ∪ block2 ∪ X)={rank_b12}, rank(torsion 1 ∪ X)={rank_tor}".format(
            rank_b1=x_equivalence["rank_block1_x_union"],
            rank_b2=x_equivalence["rank_block2_x_union"],
            rank_b12=x_equivalence["rank_block12_x_union"],
            rank_tor=x_equivalence["rank_torsion_x_union"],
        )
    )

    return {
        "generators": generators,
        "tor1": tor1_data,
        "logicals": logicals,
        "x_orthogonal": x_orth["x_orthogonal"],
        "x_orthogonal_details": x_orth,
        "x_logicals": x_logicals,
        "equivalence": equivalence,
        "x_equivalence": x_equivalence,
    }


def _format_orbit_entry(entry: Dict[str, object]) -> str:
    poly = entry["poly"]
    translation = entry.get("translation")
    monomial = entry.get("standard_monomial")
    return f"translation={translation}, standard={monomial}, poly={poly}"


def run_test_examples() -> None:
    """Run a couple of examples illustrating the semiperiodic fast path."""

    test_cases = [
        # ("x + y^3 + y^4", "y + x^3 + x^4", 7, 7),
        ("x^3 + y + y^2", "y^3 + x + x^2", 3, 3),
        # ("x^3 + y + y^2", "y^3 + x + x^2", 6, 6),
        # ("y^15 + x^5 + x^10", "x^15 + y^5 + y^10", 6, 6),
        # ("x^3 + y + y^2", "y^3 + x + x^2", 12, 6),
    ]

    for idx, (f_str, g_str, l, m) in enumerate(test_cases, start=1):
        print(
            f"\n=== Example {idx}: ring GF(2)[x,y]/(x^{l}+1, y^{m}+1), "
            f"f={f_str}, g={g_str} ==="
        )
        try:
            result = compute_ann_generators(f_str, g_str, l, m)
        except ValueError as exc:
            print(f"  ✗ {exc}")
            continue

        P = result["P"]
        Q = result["Q"]
        print(f"  ✓ P(x, y) = {P}")
        print(f"  ✓ Q(x, y) = {Q}")

        orbit_f = result["ann_f_orbit"]
        orbit_g = result["ann_g_orbit"]
        print(f"  Ann(f) orbit size = {len(orbit_f)}")
        for entry in orbit_f:
            print("    ", _format_orbit_entry(entry))

        print(f"  Ann(g) orbit size = {len(orbit_g)}")
        for entry in orbit_g:
            print("    ", _format_orbit_entry(entry))

        swapped = apply_periodic_boundary(P.subs({x: y, y: x}), l, m)
        if swapped == Q:
            print("  ✓ Verified Q(x, y) = P(y, x).")

        comparison = compare_semiperiodic_with_css(
            f_str,
            g_str,
            l,
            m,
            generators=result,
        )

if __name__ == "__main__":
    run_test_examples()
