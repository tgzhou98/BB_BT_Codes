#!/usr/bin/env python3

"""
Matrix-based implementation for computing Ann(f)/(g Ann(f)) using direct Gaussian elimination.

Algorithm:
1. Compute Ann(f) as matrix M_f where rows are generators in basis x^i y^j
2. Compute g*Ann(f) as matrix M_g
3. Find quotient M_f/M_g using Gaussian elimination:
   - Stack matrices vertically: [[M_f], [M_g]]
   - Use Gaussian elimination to find rows of M_f independent from M_g
   - Independent rows form basis for quotient space
"""

import numpy as np
import sympy as sp
from sympy import symbols, expand
import ldpc.mod2 as mod2
from typing import Any, Dict, List, Optional, Tuple

from bivariate_bicycle_codes import get_BB_Hx_Hz
from bposd.css import css_code

# Set up polynomial variables
x, y = symbols('x y')

# Cache for Groebner bases of the ambient ring relations A = <x^l+1, y^m+1>
_ring_groebner_cache: Dict[Tuple[int, int], sp.GroebnerBasis] = {}

def _get_ring_groebner(l: int, m: int) -> sp.GroebnerBasis:
    key = (l, m)
    gb = _ring_groebner_cache.get(key)
    if gb is None:
        gb = sp.groebner([x**l + 1, y**m + 1], [x, y], modulus=2)
        _ring_groebner_cache[key] = gb
    return gb


def _monomial_basis(l: int, m: int) -> List[sp.Expr]:
    """Return ordered basis [x^i y^j] with 0 <= i < l, 0 <= j < m."""

    # Ascending lex / row-major order: (0,0), (0,1), ..., (l-1, m-1)
    return [x**i * y**j for i in range(l) for j in range(m)]

def setup_ring(l: int, m: int):
    """Setup polynomial ring GF(2)[x,y] with periodic boundary x^l+1, y^m+1"""
    monomials = _monomial_basis(l, m)

    print(f"Ring: GF(2)[x,y] with {len(monomials)} monomials")
    print(f"Boundary conditions: x^{l} = 1, y^{m} = 1")
    return monomials

def apply_periodic_boundary(poly, l: int, m: int):
    """Reduce polynomial modulo A = <x^l+1, y^m+1> using Groebner remainder."""
    G_rel = _get_ring_groebner(l, m)
    _, rem = G_rel.reduce(expand(poly))
    return expand(rem)

def poly_to_vector(poly, monomials, l: int, m: int):
    """Convert polynomial to coefficient vector modulo A via Groebner reduction."""
    poly_reduced = apply_periodic_boundary(poly, l, m)
    vector = np.zeros(len(monomials), dtype=np.uint8)
    if poly_reduced == 0:
        return vector
    terms = sp.Add.make_args(poly_reduced) if poly_reduced.is_Add else [poly_reduced]
    for term in terms:
        coeff, monom_part = term.as_coeff_Mul()
        if int(coeff) % 2 == 1:
            idx = monomials.index(monom_part)
            vector[idx] = 1
    return vector

def vector_to_poly(vector, monomials):
    """Convert coefficient vector back to polynomial"""
    poly = 0
    for i, coeff in enumerate(vector):
        if coeff % 2 == 1:  # In GF(2)
            poly += monomials[i]
    return poly


def _multiplication_matrix(
    poly: sp.Expr, monomials: List[sp.Expr], l: int, m: int
) -> np.ndarray:
    """Return matrix for multiplication by `poly` in the ambient ring."""

    N = len(monomials)
    M = np.zeros((N, N), dtype=np.uint8)
    for j, mon_j in enumerate(monomials):
        prod = expand(mon_j * poly)
        M[:, j] = poly_to_vector(prod, monomials, l, m)
    return M


def _solve_linear_mod2(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """Solve A x = b over GF(2). Returns one solution or None if inconsistent."""

    A = (A.astype(np.uint8) % 2).copy()
    b = (b.astype(np.uint8) % 2).copy().reshape(-1)
    n_rows, n_cols = A.shape
    aug = np.concatenate([A, b.reshape(-1, 1)], axis=1).astype(np.uint8)

    row = 0
    pivots: List[int] = []
    for col in range(n_cols):
        pivot = None
        for r in range(row, n_rows):
            if aug[r, col] & 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n_rows):
            if r != row and (aug[r, col] & 1):
                aug[r, :] ^= aug[row, :]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break

    for r in range(n_rows):
        if not aug[r, :n_cols].any() and (aug[r, n_cols] & 1):
            return None

    x = np.zeros(n_cols, dtype=np.uint8)
    for r in range(min(len(pivots), n_rows)):
        col = pivots[r]
        s = 0
        for c in range(col + 1, n_cols):
            if aug[r, c] & 1:
                s ^= x[c]
        x[col] = (aug[r, n_cols] ^ s) & 1
    return x


def _row_basis_from_polynomials(
    polys: List[sp.Expr], monomials: List[sp.Expr], l: int, m: int
) -> Tuple[np.ndarray, List[sp.Expr]]:
    """Return independent row basis vectors/polys for given generators in R."""

    if not polys:
        return np.zeros((0, len(monomials)), dtype=np.uint8), []

    vectors = [poly_to_vector(poly, monomials, l, m) for poly in polys]
    stacked = np.vstack(vectors).astype(np.uint8)

    basis_sparse = mod2.row_basis(stacked)
    if hasattr(basis_sparse, "toarray"):
        basis_matrix = basis_sparse.toarray().astype(np.uint8)
    else:
        basis_matrix = np.asarray(basis_sparse, dtype=np.uint8)

    if basis_matrix.size == 0:
        basis_matrix = np.zeros((0, len(monomials)), dtype=np.uint8)

    basis_polys = [vector_to_poly(row, monomials) for row in basis_matrix]
    return basis_matrix, basis_polys



def _principal_ideal_basis(
    poly: sp.Expr, monomials: List[sp.Expr], l: int, m: int
) -> Tuple[np.ndarray, List[sp.Expr]]:
    """Return basis vectors/polys spanning the principal ideal (poly) in R modulo periods."""

    if poly == 0:
        zero_basis = np.zeros((0, len(monomials)), dtype=np.uint8)
        return zero_basis, []

    ambient = [x**l + 1, y**m + 1]
    ideal_generators = [sp.expand(poly), *ambient]

    gb = sp.groebner(ideal_generators, x, y, modulus=2, order="lex")
    basis_polys = [sp.expand(g.as_expr()) for g in gb.polys]

    return _row_basis_from_polynomials(basis_polys, monomials, l, m)


def _intersection_ideal_basis_via_elimination(
    f_poly: sp.Expr, g_poly: sp.Expr, monomials: List[sp.Expr], l: int, m: int
) -> Tuple[np.ndarray, List[sp.Expr]]:
    """Compute generators for (⟨f, periods⟩ ∩ ⟨g, periods⟩) in R using elimination."""

    t = sp.symbols('t')
    ambient = [x**l + 1, y**m + 1]
    gens_I = [sp.expand(f_poly), *ambient]
    gens_J = [sp.expand(g_poly), *ambient]
    K_gens = [t * gi for gi in gens_I] + [(1 + t) * gj for gj in gens_J]
    gb = sp.groebner(K_gens, t, x, y, order='lex', modulus=2)

    inter_poly = [sp.expand(g.as_expr()) for g in gb.polys if not g.as_expr().has(t)]
    if not inter_poly:
        inter_poly = [sp.expand(f_poly * g_poly)]

    return _row_basis_from_polynomials(inter_poly, monomials, l, m)


def _annihilator_generators_via_colon(
    f_poly: sp.Expr,
    l: int,
    m: int,
) -> Tuple[List[sp.Expr], Dict[str, Any]]:
    """Return generators of Ann(f) = (I : f) using an elimination trick."""

    u = sp.symbols("u")
    g1 = x**l + 1
    g2 = y**m + 1

    ring_gb = _get_ring_groebner(l, m)

    colon_generators: List[sp.Expr] = []

    ideal_with_aux = sp.groebner([g1, g2, u * f_poly - 1], u, x, y, order="lex", modulus=2)

    elimination = [sp.expand(g.as_expr()) for g in ideal_with_aux.polys if not g.as_expr().has(u)]

    candidate_polys: List[sp.Expr] = []
    candidate_seen: set[str] = set()

    for h in elimination:
        h_reduced = apply_periodic_boundary(h, l, m)
        if h_reduced == 0:
            continue
        key = sp.srepr(h_reduced)
        if key in candidate_seen:
            continue
        candidate_seen.add(key)

        _, remainder = ring_gb.reduce(sp.expand(h_reduced * f_poly))
        remainder_reduced = apply_periodic_boundary(remainder, l, m)

        if remainder_reduced == 0:
            candidate_polys.append(h_reduced)
        else:
            rem_key = sp.srepr(remainder_reduced)
            if rem_key not in candidate_seen:
                candidate_seen.add(rem_key)
                candidate_polys.append(remainder_reduced)

    if not candidate_polys:
        return [], {
            "ring_groebner": ring_gb,
            "elimination_generators": elimination,
            "candidate_generators": [],
            "colon_groebner": None,
        }

    colon_gb = sp.groebner(candidate_polys, x, y, order="lex", modulus=2)

    colon_seen: set[str] = set()
    for g in colon_gb.polys:
        expr = apply_periodic_boundary(g.as_expr(), l, m)
        if expr != 0:
            key = sp.srepr(expr)
            if key in colon_seen:
                continue
            colon_seen.add(key)
            colon_generators.append(expr)

    return colon_generators, {
        "ring_groebner": ring_gb,
        "elimination_generators": elimination,
        "candidate_generators": candidate_polys,
        "colon_groebner": colon_gb,
    }


def compute_ann_f_matrix(f_poly, monomials, l: int, m: int):
    """Compute Ann(f) using a Gröbner colon ideal computation."""

    print(f"\n=== Computing Ann(f) via Groebner colon where f = {f_poly} ===")

    ann_generators, ann_details = _annihilator_generators_via_colon(f_poly, l, m)

    generators_unique: List[sp.Expr] = []
    seen_keys: set[str] = set()
    for poly in ann_generators:
        poly_reduced = apply_periodic_boundary(sp.expand(poly), l, m)
        if poly_reduced == 0:
            continue
        key = sp.srepr(poly_reduced)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        generators_unique.append(poly_reduced)

    ann_polys = generators_unique
    ann_matrix = np.vstack(
        [poly_to_vector(poly, monomials, l, m) for poly in ann_polys]
    ).astype(np.uint8) if ann_polys else np.zeros((0, len(monomials)), dtype=np.uint8)

    print(f"Ann(f) has {len(ann_polys)} Groebner generators:")
    for i, gen in enumerate(ann_polys):
        print(f"  Ann(f)[{i}]: {gen}")

    print(f"\n=== VERIFICATION: f * Ann(f) = 0 ===")
    all_products_zero = True
    for i, h_poly in enumerate(ann_polys):
        product = expand(f_poly * h_poly)
        product_reduced = apply_periodic_boundary(product, l, m)
        if product_reduced != 0:
            print(f"  ERROR: f * Ann(f)[{i}] = {product_reduced} ≠ 0")
            all_products_zero = False
        else:
            print(f"  ✓ f * Ann(f)[{i}] = 0")

    if all_products_zero:
        print("✓ VERIFIED: All f * Ann(f) products are zero in the polynomial ring")
    else:
        print("✗ VERIFICATION FAILED: Some f * Ann(f) products are non-zero")

    return ann_matrix, ann_polys, {
        **ann_details,
        "ann_generators": ann_polys,
    }

def _leading_exponents_from_groebner(gb: sp.GroebnerBasis) -> List[Tuple[int, int]]:
    """Return leading monomial exponents for a Groebner basis over GF(2)."""

    leading_exps: List[Tuple[int, int]] = []
    for poly in gb.polys:
        expr = sp.expand(poly.as_expr())
        if expr == 0:
            continue
        poly_obj = sp.Poly(expr, x, y, modulus=2)
        monom = poly_obj.LM()
        exps = tuple(int(e) for e in monom)
        leading_exps.append(exps)
    return leading_exps


def _monomial_exponent(expr: sp.Expr) -> Tuple[int, int]:
    """Return exponent pair (a, b) for monomial expr in GF(2)[x, y]."""

    poly = sp.Poly(expr, x, y, modulus=2)
    monoms = poly.monoms()
    if not monoms:
        return (0, 0)
    exp = monoms[0]
    return int(exp[0]), int(exp[1])


def _standard_monomials_from_groebner(
    gb: sp.GroebnerBasis, monomials: List[sp.Expr]
) -> List[sp.Expr]:
    """Return monomials comprising a standard basis of R/⟨gb⟩."""

    leading_exps = _leading_exponents_from_groebner(gb)

    def _divides(div: Tuple[int, int], target: Tuple[int, int]) -> bool:
        return div[0] <= target[0] and div[1] <= target[1]

    basis: List[sp.Expr] = []
    for mon in monomials:
        exp_mon = _monomial_exponent(mon)
        if any(_divides(lm, exp_mon) for lm in leading_exps if lm != (0, 0)):
            continue
        basis.append(mon)
    return basis


def _compute_ann_quotient_via_groebner(
    ann_basis: List[sp.Expr],
    g_poly: sp.Expr,
    monomials: List[sp.Expr],
    l: int,
    m: int,
) -> Tuple[np.ndarray, List[sp.Expr], Dict[str, Any]]:
    """Return Ann(f)/(g·Ann(f)) using Groebner reduction against g·Ann(f)."""

    if not ann_basis:
        empty = np.zeros((0, len(monomials)), dtype=np.uint8)
        return empty, [], {
            "g_ann_groebner": None,
            "standard_monomials": [],
            "candidates": [],
        }

    periods = [x**l + 1, y**m + 1]
    g_ann_generators = [sp.expand(g_poly * h) for h in ann_basis]
    g_ann_generators.extend(periods)

    g_ann_gb = sp.groebner(g_ann_generators, x, y, modulus=2, order="lex")

    if len(g_ann_gb.polys) == 1 and sp.expand(g_ann_gb.polys[0].as_expr()) == 1:
        empty = np.zeros((0, len(monomials)), dtype=np.uint8)
        return empty, [], {
            "g_ann_groebner": g_ann_gb,
            "standard_monomials": [],
            "candidates": [],
        }

    standard_basis = _standard_monomials_from_groebner(g_ann_gb, monomials)

    candidate_polys = _generate_remainders_over_standard_basis(
        ann_basis,
        standard_basis,
        g_ann_gb,
        monomials,
        l,
        m,
    )

    quotient_matrix, quotient_basis = _row_basis_from_polynomials(
        candidate_polys, monomials, l, m
    )

    return quotient_matrix, quotient_basis, {
        "g_ann_groebner": g_ann_gb,
        "standard_monomials": standard_basis,
        "candidates": candidate_polys,
    }


def _generate_remainders_over_standard_basis(
    base_polys: List[sp.Expr],
    standard_monomials: List[sp.Expr],
    groebner: sp.GroebnerBasis,
    monomials: List[sp.Expr],
    l: int,
    m: int,
) -> List[sp.Expr]:
    """Spread generators across a standard monomial basis and collect remainders."""

    if not base_polys or not standard_monomials:
        return []

    seen: set[str] = set()
    candidates: List[sp.Expr] = []

    for h in base_polys:
        h_expr = sp.expand(h)
        if h_expr == 0:
            continue
        for mon in standard_monomials:
            candidate = sp.expand(mon * h_expr)
            _, remainder = groebner.reduce(candidate)
            remainder_expr = apply_periodic_boundary(remainder, l, m)
            if remainder_expr == 0:
                continue
            key = sp.srepr(remainder_expr)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(remainder_expr)

    return candidates


def compute_tor_1(
    f_str: str, g_str: str, l: int, m: int
) -> Dict[str, Any]:
    """Compute (I ∩ J)/(IJ) ⊂ R with I=⟨f, periods⟩, J=⟨g, periods⟩ using Gröbner bases.

    Current method is to calculate the intersection ideal and product ideal separately,
    Use G_{IJ} to find standard monomials by R/G_{IJ}, and find the (I ∩ J) reduced in that basis
    """
    

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)
    monomials = _monomial_basis(l, m)
    periods = [x**l + 1, y**m + 1]

    # Step 1: Groebner bases for the principal ideals I and J inside R
    ideal_f_matrix, ideal_f_basis = _principal_ideal_basis(f_poly, monomials, l, m)
    ideal_g_matrix, ideal_g_basis = _principal_ideal_basis(g_poly, monomials, l, m)

    # Step 2: Intersection ideal via elimination variable t
    intersection_matrix, intersection_basis = _intersection_ideal_basis_via_elimination(
        f_poly, g_poly, monomials, l, m
    )

    # Step 3: Product ideal IJ and its Groebner basis
    gens_I = [sp.expand(f_poly), *periods]
    gens_J = [sp.expand(g_poly), *periods]
    product_generators = [sp.expand(gi * gj) for gi in gens_I for gj in gens_J]
    product_gb = sp.groebner(product_generators + periods, x, y, modulus=2, order="lex")
    product_basis_polys = [sp.expand(p.as_expr()) for p in product_gb.polys]
    product_matrix, product_basis = _row_basis_from_polynomials(
        product_basis_polys, monomials, l, m
    )

    # Standard monomials of R / IJ
    standard_monomials = _standard_monomials_from_groebner(product_gb, monomials)

    # Step 4: Generate span of (I ∩ J)/(IJ)
    candidate_polys = _generate_remainders_over_standard_basis(
        intersection_basis,
        standard_monomials,
        product_gb,
        monomials,
        l,
        m,
    )

    # Row basis is required to eliminate duplicates
    tor_matrix, tor_basis = _row_basis_from_polynomials(candidate_polys, monomials, l, m)

    return {
        "ideal_f_matrix": ideal_f_matrix,
        "ideal_f_basis": ideal_f_basis,
        "ideal_g_matrix": ideal_g_matrix,
        "ideal_g_basis": ideal_g_basis,
        "intersection_matrix": intersection_matrix,
        "intersection_basis": intersection_basis,
        "product_matrix": product_matrix,
        "product_basis": product_basis,
        "tor_matrix": tor_matrix,
        "tor_basis": tor_basis,
        "standard_monomials": standard_monomials,
        "tor_candidates": candidate_polys,
        "dimension": len(tor_basis),
    }


def _polynomial_to_block_indicator(poly: sp.Expr, l: int, m: int) -> np.ndarray:
    """Return l x m binary array marking qubit positions touched by poly."""

    block = np.zeros((l, m), dtype=np.uint8)
    if poly == 0:
        return block

    basis = _monomial_basis(l, m)
    vec = poly_to_vector(poly, basis, l, m)

    for idx in np.where(vec == 1)[0]:
        row = int(idx // m) % l
        col = int(idx % m)
        block[row, col] = 1

    return block


def _poly_to_exponent_pairs(poly: sp.Expr, l: int, m: int) -> List[Tuple[int, int]]:
    """Convert polynomial into list of exponent pairs (a, b) with coefficients in GF(2)."""

    basis = _monomial_basis(l, m)
    vec = poly_to_vector(poly, basis, l, m)
    pairs: List[Tuple[int, int]] = []
    for idx in np.where(vec == 1)[0]:
        a = int(idx // m) % l
        b = int(idx % m)
        pairs.append((a, b))
    return pairs


def build_qubit_logical_indicator(
    poly: sp.Expr,
    l: int,
    m: int,
    block: Any,
) -> Dict[str, Any]:
    """Map polynomial to flattened qubit indicator for block1, block2, or torsion."""

    qubit_tensor = np.zeros((2, l, m), dtype=np.uint8)
    indicator = _polynomial_to_block_indicator(poly, l, m)

    if block == 0:
        qubit_tensor[0] = indicator
        block_label: Any = 1
    elif block == 1:
        qubit_tensor[1] = indicator
        block_label = 2
    elif block == "torsion":
        qubit_tensor[0] = indicator
        qubit_tensor[1] = indicator
        block_label = "torsion"
    else:
        raise ValueError("block must be 0, 1, or 'torsion'")

    qubit_vector = qubit_tensor.reshape(-1)

    return {
        "poly": poly,
        "block": block_label,
        "tensor": qubit_tensor,
        "vector": qubit_vector,
    }


def build_torsion_logical_indicator(
    torsion_poly: sp.Expr,
    f_multiplier: sp.Expr,
    g_multiplier: sp.Expr,
    l: int,
    m: int,
) -> Dict[str, Any]:
    """Map torsion element to paired qubit support on the two blocks."""

    qubit_tensor = np.zeros((2, l, m), dtype=np.uint8)
    qubit_tensor[0] = _polynomial_to_block_indicator(f_multiplier, l, m)
    qubit_tensor[1] = _polynomial_to_block_indicator(g_multiplier, l, m)

    qubit_vector = qubit_tensor.reshape(-1)

    return {
        "poly": torsion_poly,
        "block": "torsion",
        "tensor": qubit_tensor,
        "vector": qubit_vector,
        "f_multiplier": f_multiplier,
        "g_multiplier": g_multiplier,
    }


def compute_logical_qubit_operators(f_str: str, g_str: str, l: int, m: int) -> Dict[str, Any]:
    """Return logical Z operators as qubit occupancy vectors for both blocks."""

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)

    result_f = compute_ann_quotient_matrix(f_str, g_str, l, m)
    result_g = compute_ann_quotient_matrix(g_str, f_str, l, m)

    block1_ops = []
    for idx, poly in enumerate(result_f["quotient_basis"]):
        entry = build_qubit_logical_indicator(poly, l, m, block=0)
        entry["index"] = idx
        block1_ops.append(entry)

    block2_ops = []
    for idx, poly in enumerate(result_g["quotient_basis"]):
        entry = build_qubit_logical_indicator(poly, l, m, block=1)
        entry["index"] = idx
        block2_ops.append(entry)

    torsion = compute_tor_1(f_str, g_str, l, m)
    torsion_ops = []
    if torsion["tor_basis"]:
        monomials = _monomial_basis(l, m)
        mult_matrix_f = _multiplication_matrix(f_poly, monomials, l, m)
        mult_matrix_g = _multiplication_matrix(g_poly, monomials, l, m)
    else:
        monomials = []
        mult_matrix_f = None
        mult_matrix_g = None

    for idx, poly in enumerate(torsion["tor_basis"]):
        if mult_matrix_f is None or mult_matrix_g is None:
            raise RuntimeError("Expected torsion multiplication matrices to be initialized")

        tor_vec = poly_to_vector(poly, monomials, l, m)
        f_solution = _solve_linear_mod2(mult_matrix_f, tor_vec)
        g_solution = _solve_linear_mod2(mult_matrix_g, tor_vec)

        if f_solution is None or g_solution is None:
            raise ValueError(
                "Failed to express torsion generator as both f-multiple and g-multiple"
            )

        f_multiplier = vector_to_poly(f_solution, monomials)
        g_multiplier = vector_to_poly(g_solution, monomials)

        entry = build_torsion_logical_indicator(
            poly,
            f_multiplier,
            g_multiplier,
            l,
            m,
        )
        entry["index"] = idx
        torsion_ops.append(entry)

    all_vectors = [op["vector"] for op in block1_ops + block2_ops + torsion_ops]
    # all_vectors = [op["vector"] for op in block1_ops + block2_ops]
    if all_vectors:
        logical_matrix = np.vstack(all_vectors).astype(np.uint8)
        span_rank = mod2.rank(logical_matrix)
    else:
        logical_matrix = np.zeros((0, 2 * l * m), dtype=np.uint8)
        span_rank = 0

    return {
        "block1": block1_ops,
        "block2": block2_ops,
        "torsion": torsion_ops,
        "tor_details": torsion,
        "matrix": logical_matrix,
        "independence": {
            "matrix": logical_matrix,
            "rank": span_rank,
            "count": logical_matrix.shape[0],
            "independent": span_rank == logical_matrix.shape[0],
        },
    }


def verify_logical_z_equivalence(
    f_str: str,
    g_str: str,
    l: int,
    m: int,
    logicals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Show polynomial logical Zs match css_code logical Zs up to Z stabilizers."""

    if logicals is None:
        logicals = compute_logical_qubit_operators(f_str, g_str, l, m)

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)
    f_terms = _poly_to_exponent_pairs(f_poly, l, m)
    g_terms = _poly_to_exponent_pairs(g_poly, l, m)

    Hx, Hz = get_BB_Hx_Hz(f_terms, g_terms, l, m)

    code = css_code(hx=Hx, hz=Hz, name=f"BB_{l}x{m}")

    lz_matrix = code.lz.toarray().astype(np.uint8)

    z_stab_sparse = mod2.row_basis(Hz)

    z_stab_basis = z_stab_sparse.toarray().astype(np.uint8)

    if z_stab_basis.size == 0:
        z_stab_basis = np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    elif z_stab_basis.ndim == 1:
        z_stab_basis = z_stab_basis.reshape(1, -1)

    z_stab_rank = z_stab_basis.shape[0] if z_stab_basis.size else 0

    # Print full CSS logical Z matrix for inspection
    if lz_matrix.size:
        print("CSS logical Z matrix (each row is a binary vector):")
        for i, row in enumerate(lz_matrix):
            print(f"  css_logical_z[{i}] = {row}")
    else:
        print("CSS logical Z matrix is empty")

    poly_entries = logicals["block1"] + logicals["block2"] + logicals["torsion"]
    # poly_entries = logicals["block1"] + logicals["block2"]
    
    # DEBUG: Show poly_entries structure and shape
    print(f"DEBUG: len(logicals['block1']): {len(logicals['block1'])}")
    print(f"DEBUG: len(logicals['block2']): {len(logicals['block2'])}")
    print(f"DEBUG: len(logicals.get('torsion', [])): {len(logicals.get('torsion', []))}")
    print(f"DEBUG: total poly_entries length: {len(poly_entries)}")
    if poly_entries:
        print(f"DEBUG: first poly_entry vector shape: {poly_entries[0]['vector'].shape}")
        print(f"DEBUG: poly_entries structure: list of {len(poly_entries)} entries, each with 'vector' key")
    else:
        print(f"DEBUG: poly_entries is empty")

    poly_matrix = (
        np.vstack([entry["vector"] for entry in poly_entries]).astype(np.uint8)
        if poly_entries
        else np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    )


    css_stack = (
        np.vstack([z_stab_basis, lz_matrix])
        if lz_matrix.size or z_stab_basis.size
        else np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    )
    poly_stack = (
        np.vstack([z_stab_basis, poly_matrix])
        if poly_matrix.size or z_stab_basis.size
        else np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    )
    # rank(css ∪ Z)=42, rank(poly ∪ Z)=45, rank(union)=54, rank(Z stabilizer)=30
    # 2025/09/20
    # NOTE: rank_poly - rank_Z (= 15) is actually < rank_poly_alone (= 16)
    # This is expected: some polynomial vectors may be linearly dependent on Z stabilizers
    print(f"DEBUG: css_stack shape: {css_stack.shape}")
    print(f"DEBUG: poly_stack shape: {poly_stack.shape}")
    
    rank_css = mod2.rank(css_stack)
    rank_poly = mod2.rank(poly_stack)
    rank_union = (
        mod2.rank(np.vstack([z_stab_basis, lz_matrix, poly_matrix]))
    )
    
    def in_z_stabilizer_span(vec: np.ndarray) -> bool:
        if z_stab_basis.size == 0:
            return not np.any(vec)
        stacked = np.vstack([z_stab_basis, vec])
        return mod2.rank(stacked) == mod2.rank(z_stab_basis)

    matches: List[Dict[str, Any]] = []
    unmatched_polynomial: List[Dict[str, Any]] = []

    for entry in poly_entries:
        vec = entry["vector"].astype(np.uint8)
        found = False
        for idx, css_vec in enumerate(lz_matrix):
            diff = vec ^ css_vec
            if in_z_stabilizer_span(diff):
                matches.append(
                    {
                        "poly_block": entry["block"],
                        "poly_index": entry["index"],
                        "css_lz_index": idx,
                    }
                )
                found = True
                break
        if not found:
            unmatched_polynomial.append(entry)

    unmatched_css = []
    for idx, css_vec in enumerate(lz_matrix):
        found = False
        for entry in poly_entries:
            diff = entry["vector"] ^ css_vec
            if in_z_stabilizer_span(diff):
                found = True
                break
        if not found:
            unmatched_css.append(idx)

    poly_not_in_css_span: List[int] = []
    css_not_in_poly_span: List[int] = []

    if css_stack.size:
        for idx, vec in enumerate(poly_matrix):
            stacked = np.vstack([css_stack, vec])
            if mod2.rank(stacked) > rank_css:
                poly_not_in_css_span.append(idx)

    if poly_stack.size:
        for idx, vec in enumerate(lz_matrix):
            stacked = np.vstack([poly_stack, vec])
            if mod2.rank(stacked) > rank_poly:
                css_not_in_poly_span.append(idx)

    return {
        "matches": matches,
        "unmatched_polynomial": unmatched_polynomial,
        "unmatched_css": unmatched_css,
        "poly_not_in_css_span": poly_not_in_css_span,
        "css_not_in_poly_span": css_not_in_poly_span,
        "rank_css_space": rank_css,
        "rank_poly_space": rank_poly,
        "rank_union_space": rank_union,
        "rank_z_stabilizer": z_stab_rank,
        "lz_matrix": lz_matrix,
        "z_stabilizer_basis": z_stab_basis,
    }


def _express_vector_in_basis(
    target: np.ndarray,
    basis: np.ndarray,
    labels: List[str],
) -> Optional[List[str]]:
    """Return labels whose XOR reproduces `target`, or None if outside the span."""

    target = target.astype(np.uint8, copy=False)
    if basis.size == 0:
        return [] if not np.any(target) else None

    stacked = np.vstack([basis, target])
    row_echelon_result, _, transform, _ = mod2.row_echelon(stacked, full=True)
    if np.any(row_echelon_result[-1]):
        return None

    coeffs = np.asarray(transform[-1, :-1]).astype(np.uint8) % 2
    return [labels[idx] for idx, val in enumerate(coeffs) if val]


def _express_with_preference(
    target: np.ndarray,
    primary_basis: np.ndarray,
    primary_labels: List[str],
    fallback_basis: np.ndarray,
    fallback_labels: List[str],
) -> Optional[List[str]]:
    """Prefer expressing `target` using `primary_basis` before falling back."""

    if primary_basis.size:
        primary_combo = _express_vector_in_basis(target, primary_basis, primary_labels)
        if primary_combo is not None:
            return primary_combo
    elif not np.any(target):
        return []

    return _express_vector_in_basis(target, fallback_basis, fallback_labels)


def _print_logical_equivalence_details(
    logicals: Dict[str, Any], equivalence: Dict[str, Any]
) -> None:
    """Print Z stabilizers and CSS↔polynomial logical relationships explicitly."""

    z_basis = equivalence["z_stabilizer_basis"].astype(np.uint8, copy=False)
    # if z_basis.size:
    #     print("Z stabilizer basis (each row is a binary array):")
    #     # for idx, row in enumerate(z_basis):
    #     #     print(f"  stabilizer_z[{idx}] = {row}")
    # else:
    #     print("Z stabilizer basis (empty)")

    poly_entries = logicals["block1"] + logicals["block2"]

    css_matrix = equivalence["lz_matrix"].astype(np.uint8, copy=False)

    num_qubits = 0
    if poly_entries:
        num_qubits = poly_entries[0]["vector"].size
    elif z_basis.size:
        num_qubits = z_basis.shape[1]
    elif css_matrix.size:
        num_qubits = css_matrix.shape[1]

    if poly_entries:
        poly_matrix = np.vstack([entry["vector"] for entry in poly_entries]).astype(np.uint8)
    else:
        poly_matrix = np.zeros((0, num_qubits), dtype=np.uint8)

    poly_labels = [f"poly_logical[{idx}]" for idx in range(poly_matrix.shape[0])]
    css_labels = [f"css_logical_z[{idx}]" for idx in range(css_matrix.shape[0])]
    stab_labels = [f"stabilizer_z[{idx}]" for idx in range(z_basis.shape[0])]

    css_basis = (
        np.vstack([poly_matrix, z_basis])
        if poly_matrix.size or z_basis.size
        else np.zeros((0, num_qubits), dtype=np.uint8)
    )
    css_basis_labels = poly_labels + stab_labels

    poly_basis = (
        np.vstack([css_matrix, z_basis])
        if css_matrix.size or z_basis.size
        else np.zeros((0, num_qubits), dtype=np.uint8)
    )
    poly_basis_labels = css_labels + stab_labels

    if css_matrix.size:
        print("CSS logical Z expressed via polynomial logicals and Z stabilizers:")
        for idx, vec in enumerate(css_matrix):
            combo = _express_with_preference(
                vec,
                poly_matrix,
                poly_labels,
                css_basis,
                css_basis_labels,
            )
            if combo is None:
                print(
                    "  css_logical_z[{idx}] cannot be expressed via polynomial logicals and Z stabilizers".format(
                        idx=idx
                    )
                )
            else:
                rhs = " + ".join(combo) if combo else "0"
                print(f"  css_logical_z[{idx}] = {rhs}")

    if poly_matrix.size:
        print("Polynomial logical Z expressed via css_code logicals and Z stabilizers:")
        for idx, vec in enumerate(poly_matrix):
            combo = _express_with_preference(
                vec,
                css_matrix,
                css_labels,
                poly_basis,
                poly_basis_labels,
            )
            if combo is None:
                print(
                    "  poly_logical[{idx}] cannot be expressed via css_code logicals and Z stabilizers".format(
                        idx=idx
                    )
                )
            else:
                rhs = " + ".join(combo) if combo else "0"
                print(f"  poly_logical[{idx}] = {rhs}")


def compute_ann_quotient_matrix(f_str: str, g_str: str, l: int, m: int):
    """Compute Ann(f)/(g Ann(f)) using Groebner-based standard monomials."""

    print(f"=== Computing Ann({f_str})/(g Ann({f_str})) via Groebner reduction ===")
    print(f"Ring: GF(2)[x,y]/(x^{l}+1, y^{m}+1)")
    print()

    # Setup
    monomials = setup_ring(l, m)

    # Parse polynomials
    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)

    print(f"f = {f_poly}")
    print(f"g = {g_poly}")

    # Step 1: Compute Ann(f) as matrix M_f
    M_f, ann_f_polys, ann_details = compute_ann_f_matrix(f_poly, monomials, l, m)

    # Step 2: Groebner quotient against g·Ann(f)
    quotient_matrix, quotient_polys, g_ann_details = _compute_ann_quotient_via_groebner(
        ann_f_polys,
        g_poly,
        monomials,
        l,
        m,
    )

    if g_ann_details["standard_monomials"]:
        print(
            "Standard monomials for R/(g·Ann(f)):",
            [str(mon) for mon in g_ann_details["standard_monomials"]],
        )
    else:
        print("Standard monomial basis for R/(g·Ann(f)) is empty or trivial.")

    print(f"\n=== Final Result ===")
    print(f"Dimension of Ann({f_str})/(g Ann({f_str})): {len(quotient_polys)}")
    print(f"Quotient basis:")
    for i, poly in enumerate(quotient_polys):
        print(f"  QuotientBasis[{i}]: {poly}")

    return {
        "M_f": M_f,
        "M_g": None,
        "quotient_matrix": quotient_matrix,
        "quotient_basis": quotient_polys,
        "dimension": len(quotient_polys),
        "g_ann_details": g_ann_details,
        "ann_details": ann_details,
    }

def compute_ann_quotient_symmetric(f_str: str, g_str: str, l: int, m: int):
    """Compute both Ann(f)/(g Ann(f)) and Ann(g)/(f Ann(g))"""

    print(f"=== Computing Both Quotients: Ann(f)/(g Ann(f)) and Ann(g)/(f Ann(g)) ===")
    print(f"Ring: GF(2)[x,y]/(x^{l}+1, y^{m}+1)")
    print(f"f = {f_str}, g = {g_str}")
    print()

    # Compute Ann(f)/(g Ann(f))
    print(">>> PART 1: Computing Ann(f)/(g Ann(f)) <<<")
    result_f = compute_ann_quotient_matrix(f_str, g_str, l, m)

    print("\n" + "="*80)

    # Compute Ann(g)/(f Ann(g))
    print(">>> PART 2: Computing Ann(g)/(f Ann(g)) <<<")
    result_g = compute_ann_quotient_matrix(g_str, f_str, l, m)

    print(f"\n=== Summary ===")
    print(f"Dimension of Ann({f_str})/(g Ann({f_str})): {result_f['dimension']}")
    print(f"Dimension of Ann({g_str})/(f Ann({g_str})): {result_g['dimension']}")

    return {
        "ann_f_quotient": result_f,
        "ann_g_quotient": result_g
    }

def run_test_examples():
    """Run test examples"""

    test_cases = [
        # ("1 + x", "1 + y", 3, 3),
        # ("1 + x + x*y", "1 + y + x*y", 3, 3),
        # ("1 + x + x*y", "1 + y + x*y", 6, 6),
        ("x^3 + y + y^2", "y^3 + x + x^2", 3, 3),
        ("x^3 + y + y^2", "y^3 + x + x^2", 6, 6),
        # ("x^3 + y + y^2", "y^3 + x + x^2", 12, 6),
        # ("x+1", "y+1+x^2", 2, 2),
    ]

    print("=== Running Test Examples ===")

    for i, (f, g, l, m) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: f={f}, g={g}, ring=GF(2)[x,y]/(x^{l}+1,y^{m}+1) ---")
        try:
            # quotient_summary = compute_ann_quotient_symmetric(f, g, l, m)
            # print(f"Ann(f)/(g Ann(f)) dimension = {quotient_summary['ann_f_quotient']['dimension']}")
            # print(f"Ann(g)/(f Ann(g)) dimension = {quotient_summary['ann_g_quotient']['dimension']}")

            logicals = compute_logical_qubit_operators(f, g, l, m)

            print("Logical Z operators on block 1 (Ann(f)/(g Ann(f))):")
            for entry in logicals["block1"]:
                print(f"  index {entry['index']}, poly {entry['poly']}")
                # print("    tensor=", np.array2string(entry["tensor"], separator=", "))
                # print("    vector=", entry["vector"])
            print("\n")

            print("Logical Z operators on block 2 (Ann(g)/(f Ann(g))):")
            for entry in logicals["block2"]:
                print(f"  index {entry['index']}, poly {entry['poly']}")
                # print("    tensor=", np.array2string(entry["tensor"], separator=", "))
                # print("    vector=", entry["vector"])
            print("\n")

            print("Logical Z torsion operators (Tor_1):")
            if logicals["torsion"]:
                for entry in logicals["torsion"]:
                    print(f"  index {entry['index']}, poly {entry['poly']}")
                    if "f_multiplier" in entry and "g_multiplier" in entry:
                        print(f"    f_multiplier = {entry['f_multiplier']}")
                        print(f"    g_multiplier = {entry['g_multiplier']}")
                    # print("    tensor=", np.array2string(entry["tensor"], separator=", "))
                    # print("    vector=", entry["vector"])
                print("\n")
            else:
                print("  (none)")

            indep = logicals["independence"]
            print(
                f"Logical operator rank check: rank={indep['rank']} count={indep['count']} independent={indep['independent']}"
            )
            print("Stacked logical Z matrix shape:", logicals["matrix"].shape)
            print(logicals["matrix"])

            equivalence = verify_logical_z_equivalence(f, g, l, m, logicals)
            print(
                f"Polynomial ↔ css_code logical Z matches: {len(equivalence['matches'])}; "
                f"unmatched polynomial (pairwise)={len(equivalence['unmatched_polynomial'])}; "
                f"unmatched css (pairwise)={len(equivalence['unmatched_css'])}"
            )
            print(
                "  rank(css ∪ Z)={rank_css}, rank(poly ∪ Z)={rank_poly}, rank(union)={rank_union}, rank(Z stabilizer)={rank_z}".format(
                    rank_css=equivalence["rank_css_space"],
                    rank_poly=equivalence["rank_poly_space"],
                    rank_union=equivalence["rank_union_space"],
                    rank_z=equivalence["rank_z_stabilizer"],
                )
            )
            _print_logical_equivalence_details(logicals, equivalence)
            if equivalence["poly_not_in_css_span"]:
                print(
                    "  ⚠ Polynomial vectors outside span(css ∪ Z) indices:",
                    equivalence["poly_not_in_css_span"],
                )
            if equivalence["css_not_in_poly_span"]:
                print(
                    "  ⚠ css_code vectors outside span(polynomials ∪ Z) indices:",
                    equivalence["css_not_in_poly_span"],
                )

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 80)

if __name__ == "__main__":
    # Test with simple example
    # print("=== Single Test Example ===")
    # print("Testing with f = x + 1, g = y + 1, in GF(2)[x,y]/(x^3+1, y^3+1)")
    # result = compute_ann_quotient_matrix("x + 1", "y + 1", 3, 3)
    # print("\n" + "="*80)

    # Run test suite
    run_test_examples()
