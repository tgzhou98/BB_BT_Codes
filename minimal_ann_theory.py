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

import sympy as sp
from sympy import expand, symbols
from typing import Dict, List, Optional, Tuple

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
)

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


def compute_ann_generators(
    f_str: str, g_str: str, l: int, m: int
) -> Dict[str, object]:
    """Compute semiperiodic generators P and Q together with their translation orbits."""

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

    fg_gb = _get_fg_ring_groebner(f_poly, g_poly, l, m)

    ann_f_orbit = (
        translation_orbit(P, l, m, quotient_gb=fg_gb) if P is not None else []
    )
    ann_g_orbit = (
        translation_orbit(Q, l, m, quotient_gb=fg_gb) if Q is not None else []
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
    f_str: str,
    g_str: str,
    l: int,
    m: int,
) -> Dict[str, object]:
    """Return Tor₁ data using the Gaussian-elimination helper."""

    return compute_tor_1(f_str, g_str, l, m)


def compare_semiperiodic_with_css(
    f_str: str,
    g_str: str,
    l: int,
    m: int,
    *,
    generators: Optional[Dict[str, object]] = None,
    tor1_data: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Compare semiperiodic Ann(f), Ann(g), and Tor₁ logicals against css_code outputs."""

    if generators is None:
        generators = compute_ann_generators(f_str, g_str, l, m)
    if tor1_data is None:
        tor1_data = compute_semiperiodic_tor1(f_str, g_str, l, m)

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)
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

    torsion_ops: List[Dict[str, object]] = []
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

    logicals: Dict[str, object] = {
        "block1": block1_ops,
        "block2": block2_ops,
        "torsion": torsion_ops,
        "tor_details": tor1_data,
        "tor2_details": None,
    }

    equivalence = verify_logical_z_equivalence(
        f_str,
        g_str,
        l,
        m,
        logicals=logicals,
    )

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

    return {
        "generators": generators,
        "tor1": tor1_data,
        "logicals": logicals,
        "equivalence": equivalence,
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
        # ("x^3 + y + y^2", "y^3 + x + x^2", 3, 3),
        # ("x^3 + y + y^2", "y^3 + x + x^2", 6, 6),
        # ("x^3 + y + y^2", "y^3 + x + x^2", 6, 6),
        ("x^3 + y + y^2", "y^3 + x + x^2", 12, 12),
        # ("y^15 + x^5 + x^10", "x^15 + y^5 + y^10", 6, 6),
        # ("x^3 + y + y^2", "y^3 + x + x^2", 3, 3),
        # ("1 + x", "1 + y", 3, 3),
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
