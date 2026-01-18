"""Laurent-polynomial helpers and ABC geometry builder."""

from __future__ import annotations

from re import I
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from ldpc import mod2
from scipy import sparse

import matplotlib.pyplot as plt

Site = Tuple[int, int]


def _normalize_sites(region_sites: Iterable[Site]) -> List[Site]:
    return sorted({(int(x), int(y)) for x, y in region_sites})


class LaurentPoly:
    """Laurent polynomial in F2[x^{+-1}, y^{+-1}] represented by a term dict."""

    def __init__(
        self,
        terms: Optional[
            Union[Mapping[Tuple[int, int], int], Sequence[Tuple[int, int]]]
        ] = None,
    ) -> None:
        self.terms: Dict[Tuple[int, int], int] = {}
        if terms is None:
            return
        if isinstance(terms, Mapping):
            items = terms.items()
        else:
            items = [(pair, 1) for pair in terms]
        for (ex, ey), coef in items:
            if int(coef) % 2 == 0:
                continue
            key = (int(ex), int(ey))
            if key in self.terms:
                del self.terms[key]
            else:
                self.terms[key] = 1

    @classmethod
    def from_terms(cls, terms: Sequence[Tuple[int, int]]) -> "LaurentPoly":
        return cls(terms)

    def support(self) -> List[Tuple[int, int]]:
        return sorted(self.terms.keys())

    def add(self, other: "LaurentPoly") -> "LaurentPoly":
        return self ^ other

    def shift(self, dx: int, dy: int) -> "LaurentPoly":
        shifted = {(ex + int(dx), ey + int(dy)): 1 for ex, ey in self.terms}
        return LaurentPoly(shifted)

    def antipode(self) -> "LaurentPoly":
        return LaurentPoly({(-ex, -ey): 1 for ex, ey in self.terms})

    def __xor__(self, other: "LaurentPoly") -> "LaurentPoly":
        terms: Dict[Tuple[int, int], int] = dict(self.terms)
        for key in other.terms:
            if key in terms:
                del terms[key]
            else:
                terms[key] = 1
        return LaurentPoly(terms)

    def __add__(self, other: "LaurentPoly") -> "LaurentPoly":
        return self.__xor__(other)

    def __repr__(self) -> str:
        if not self.terms:
            return "LaurentPoly(0)"
        parts = [f"x^{ex} y^{ey}" for ex, ey in self.support()]
        return f"LaurentPoly({', '.join(parts)})"


def build_full_lattice_sites(L: int) -> List[Site]:
    """Return all sites for a square LxL lattice."""
    if int(L) <= 0:
        raise ValueError("L must be positive")
    return _normalize_sites((x, y) for x in range(int(L)) for y in range(int(L)))


def build_site_index(sites: Sequence[Site]) -> Dict[Site, int]:
    """Return a deterministic site->index mapping."""
    return {site: idx for idx, site in enumerate(_normalize_sites(sites))}


def qubits_from_sites(
    sites: Sequence[Site],
    site_index: Mapping[Site, int],
    *,
    include_edge1: bool = True,
    include_edge2: bool = True,
) -> List[int]:
    """Return qubit indices for edge-1/edge-2 on the given sites."""
    if not include_edge1 and not include_edge2:
        return []
    n_sites = len(site_index)
    indices: List[int] = []
    for site in sites:
        key = (int(site[0]), int(site[1]))
        if key not in site_index:
            raise KeyError(f"Site {key} not found in site_index")
        base = site_index[key]
        if include_edge1:
            indices.append(base)
        if include_edge2:
            indices.append(base + n_sites)
    return _unique_sorted(indices)


def _ensure_laurent(value: object) -> LaurentPoly:
    if isinstance(value, LaurentPoly):
        return value
    if isinstance(value, (list, tuple)):
        return LaurentPoly.from_terms(value)  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return LaurentPoly(value)  # type: ignore[arg-type]
    raise TypeError("f/g must be LaurentPoly, mapping, or list of (ex, ey) pairs")


def _laurent_term_string(ex: int, ey: int) -> str:
    parts: List[str] = []
    if ex:
        parts.append("x" if ex == 1 else f"x^{ex}")
    if ey:
        parts.append("y" if ey == 1 else f"y^{ey}")
    return "*".join(parts) if parts else "1"


def _laurent_poly_string(poly: LaurentPoly) -> str:
    support = sorted(
        poly.support(), key=lambda term: (abs(term[0]) + abs(term[1]), term[0], term[1])
    )
    if not support:
        return "0"
    return "+".join(_laurent_term_string(ex, ey) for ex, ey in support)


def _laurent_poly_slug(poly: LaurentPoly) -> str:
    support = sorted(
        poly.support(), key=lambda term: (abs(term[0]) + abs(term[1]), term[0], term[1])
    )
    if not support:
        return "0"

    def term_slug(ex: int, ey: int) -> str:
        parts: List[str] = []
        if ex:
            parts.append("x" if ex == 1 else f"x{ex}")
        if ey:
            parts.append("y" if ey == 1 else f"y{ey}")
        return "".join(parts) if parts else "1"

    return "+".join(term_slug(ex, ey) for ex, ey in support)


def _row_Av_columns(
    center: Site,
    f_support: Sequence[Tuple[int, int]],
    g_support: Sequence[Tuple[int, int]],
    site_index: Mapping[Site, int],
    *,
    strict: bool = True,
) -> Optional[List[int]]:
    """Return column indices for X-type stabilizer row."""
    n_sites = len(site_index)
    x0, y0 = int(center[0]), int(center[1])
    cols: List[int] = []

    for dx, dy in f_support:
        site = (x0 + dx, y0 + dy)
        if site not in site_index:
            if strict:
                return None
            continue
        cols.append(site_index[site])

    for dx, dy in g_support:
        site = (x0 + dx, y0 + dy)
        if site not in site_index:
            if strict:
                return None
            continue
        cols.append(n_sites + site_index[site])

    return cols or None


def _row_Bp_columns(
    center: Site,
    f_support: Sequence[Tuple[int, int]],
    g_support: Sequence[Tuple[int, int]],
    site_index: Mapping[Site, int],
    *,
    strict: bool = True,
) -> Optional[List[int]]:
    """Return column indices for Z-type stabilizer row."""
    n_sites = len(site_index)
    n = 2 * n_sites
    x0, y0 = int(center[0]), int(center[1])
    cols: List[int] = []

    for dx, dy in g_support:
        site = (x0 + dx, y0 + dy)
        if site not in site_index:
            if strict:
                return None
            continue
        cols.append(n + site_index[site])

    for dx, dy in f_support:
        site = (x0 + dx, y0 + dy)
        if site not in site_index:
            if strict:
                return None
            continue
        cols.append(n + n_sites + site_index[site])

    return cols or None


def _stabilizer_matrix_commutes(stabilizer_matrix: object) -> bool:
    mat, total_qubits = _normalize_stabilizer_matrix(stabilizer_matrix)
    if mat.shape[0] == 0:
        return True
    x = mat[:, :total_qubits]
    z = mat[:, total_qubits:]
    sym = (x @ z.T) + (z @ x.T)
    if sym.data.size:
        sym.data %= 2
        sym.eliminate_zeros()
    return sym.nnz == 0


def _to_csr_uint8(mat: object) -> sparse.csr_matrix:
    if sparse.issparse(mat):
        csr = mat.tocsr().astype(np.uint8)
        if csr.data.size:
            csr.data %= 2
            csr.eliminate_zeros()
        return csr
    return sparse.csr_matrix(np.asarray(mat, dtype=np.uint8) % 2)


# I've checked this function
def build_stabilizer_matrix_laurent(
    omega_sites: Sequence[Site],
    f: object,
    g: object,
    *,
    centers: Optional[Sequence[Site]] = None,
    strict: bool = True,
    check_commutation: bool = True,
) -> Tuple[sparse.csr_matrix, Dict[Site, int]]:
    """Build stabilizers using Laurent-polynomial generators on an open region.

    Returns a CSR stabilizer matrix over F2.
    """
    sites = _normalize_sites(omega_sites)
    site_index = {site: idx for idx, site in enumerate(sites)}
    if centers is None:
        centers_list = sites
    else:
        centers_list = _normalize_sites(centers)
        missing = [center for center in centers_list if center not in site_index]
        if missing:
            raise ValueError(f"Centers not in omega_sites: {missing[:5]}")

    f_poly = _ensure_laurent(f)
    g_poly = _ensure_laurent(g)
    f_support = f_poly.support()
    g_support = g_poly.support()
    f_anti_support = f_poly.antipode().support()
    g_anti_support = g_poly.antipode().support()

    n_sites = len(site_index)
    n = 2 * n_sites
    row_indices: List[int] = []
    col_indices: List[int] = []
    data: List[int] = []
    row_count = 0

    for center in centers_list:
        cols = _row_Av_columns(center, f_support, g_support, site_index, strict=strict)
        if cols:
            row_indices.extend([row_count] * len(cols))
            col_indices.extend(cols)
            data.extend([1] * len(cols))
            row_count += 1

        cols = _row_Bp_columns(
            center, f_anti_support, g_anti_support, site_index, strict=strict
        )
        if cols:
            row_indices.extend([row_count] * len(cols))
            col_indices.extend(cols)
            data.extend([1] * len(cols))
            row_count += 1

    if row_count:
        # Coordinate version of the sparse matrix
        mat = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(row_count, 2 * n),
            dtype=np.uint8,
        )
        mat.sum_duplicates()
        if mat.data.size:
            mat.data %= 2
            mat.eliminate_zeros()
        stabilizer_matrix = mat.tocsr()
    else:
        stabilizer_matrix = sparse.csr_matrix((0, 2 * n), dtype=np.uint8)
    if check_commutation and not _stabilizer_matrix_commutes(stabilizer_matrix):
        raise ValueError("Non-commuting stabilizers in Laurent construction")
    return stabilizer_matrix, site_index


def _unique_sorted(indices: Iterable[int]) -> List[int]:
    return sorted({int(q) for q in indices})


def _normalize_subsystem(subsystem: Sequence[int], total_qubits: int) -> np.ndarray:
    qubits = np.array(sorted({int(q) for q in subsystem}), dtype=np.int64)
    if np.any(qubits < 0) or np.any(qubits >= total_qubits):
        raise ValueError("subsystem indices out of range")
    return qubits


def _normalize_stabilizer_matrix(
    stabilizer_matrix: object,
    *,
    num_qubits: Optional[int] = None,
) -> Tuple[sparse.csr_matrix, int]:
    mat = _to_csr_uint8(stabilizer_matrix)
    if len(mat.shape) != 2:
        raise ValueError("stabilizer_matrix must be 2D")
    cols = mat.shape[1]
    if cols == 0:
        if num_qubits is None:
            raise ValueError("num_qubits must be provided for empty matrix")
        total_qubits = int(num_qubits)
    else:
        if cols % 2 != 0:
            raise ValueError("stabilizer_matrix must have 2n columns")
        total_qubits = cols // 2
        if num_qubits is not None and int(num_qubits) != total_qubits:
            raise ValueError("num_qubits does not match stabilizer_matrix")
    return mat, total_qubits


def _stabilizer_rank_for_qubits_matrix(
    stabilizer_matrix: object,
    qubits: np.ndarray,
    *,
    num_qubits: Optional[int] = None,
) -> int:
    mat, total_qubits = _normalize_stabilizer_matrix(
        stabilizer_matrix, num_qubits=num_qubits
    )
    cols = np.concatenate([qubits, qubits + total_qubits])
    sub = mat[:, cols]
    return int(mod2.rank(sub, method="sparse"))


def entanglement_entropy_from_stabilizer_matrix(
    stabilizer_matrix: object,
    subsystem: Sequence[int],
    *,
    num_qubits: Optional[int] = None,
) -> int:
    """Return S(A) in bits from an r x 2n stabilizer matrix."""
    mat, total_qubits = _normalize_stabilizer_matrix(
        stabilizer_matrix, num_qubits=num_qubits
    )
    qubits = _normalize_subsystem(subsystem, total_qubits)
    complement = np.setdiff1d(np.arange(total_qubits), qubits, assume_unique=True)
    stab_part_rank = _stabilizer_rank_for_qubits_matrix(
        mat, complement, num_qubits=total_qubits
    )
    stab_rank = mod2.rank(mat, method="sparse")
    return int(stab_part_rank - complement.size + total_qubits - stab_rank)


def conditional_mutual_information_from_stabilizer_matrix(
    stabilizer_matrix: object,
    region_a: Sequence[int],
    region_b: Sequence[int],
    region_c: Sequence[int],
    *,
    num_qubits: Optional[int] = None,
) -> int:
    """Return I(A:C|B) = S(AB) + S(BC) - S(B) - S(ABC) in bits."""
    a = _unique_sorted(region_a)
    b = _unique_sorted(region_b)
    c = _unique_sorted(region_c)
    ab = _unique_sorted(a + b)
    bc = _unique_sorted(b + c)
    abc = _unique_sorted(a + b + c)

    s_ab = entanglement_entropy_from_stabilizer_matrix(
        stabilizer_matrix, ab, num_qubits=num_qubits
    )
    s_bc = entanglement_entropy_from_stabilizer_matrix(
        stabilizer_matrix, bc, num_qubits=num_qubits
    )
    s_b = entanglement_entropy_from_stabilizer_matrix(
        stabilizer_matrix, b, num_qubits=num_qubits
    )
    s_abc = entanglement_entropy_from_stabilizer_matrix(
        stabilizer_matrix, abc, num_qubits=num_qubits
    )
    return int(s_ab + s_bc - s_b - s_abc)


def cmi_from_stabilizers(
    stabilizer_matrix: object,
    region_a: Sequence[int],
    region_b: Sequence[int],
    region_c: Sequence[int],
    *,
    num_qubits: Optional[int] = None,
) -> int:
    return conditional_mutual_information_from_stabilizer_matrix(
        stabilizer_matrix,
        region_a,
        region_b,
        region_c,
        num_qubits=num_qubits,
    )


def build_geometry_ABC(
    Lv1: int,
    Lv2: int,
    Lv3: int,
    Lh1: int,
    Lh2: int,
    *,
    lattice_size: Optional[int] = None,
    origin: Optional[Site] = None,
) -> Tuple[List[Site], List[Site], List[Site], List[Site], Dict[str, int]]:
    """Return Omega, A, B, C, and meta for the rectangular annulus geometry."""
    for name, value in [
        ("Lv1", Lv1),
        ("Lv2", Lv2),
        ("Lv3", Lv3),
        ("Lh1", Lh1),
        ("Lh2", Lh2),
    ]:
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    if Lv1 < Lv3:
        raise ValueError("Lv1 must be >= Lv3 for the inner rectangle to fit")

    Wout = 2 * Lh1 + Lh2
    Hout = 2 * Lv1 + Lv2
    Win = Lh2
    Hin = 2 * Lv3 + Lv2

    if lattice_size is not None and origin is not None:
        raise ValueError("Use only one of lattice_size or origin")
    if lattice_size is not None:
        L = int(lattice_size)
        if L < Wout or L < Hout:
            raise ValueError("lattice_size too small for annulus geometry")
        x0 = (L - Wout) // 2
        y0 = (L - Hout) // 2
    else:
        x0, y0 = origin if origin is not None else (0, 0)
        x0 = int(x0)
        y0 = int(y0)

    x_in0 = Lh1
    x_in1 = x_in0 + Win
    y_in0 = Lv1 - Lv3
    y_in1 = y_in0 + Hin

    outer = {(x0 + x, y0 + y) for x in range(Wout) for y in range(Hout)}
    inner = {
        (x0 + x, y0 + y) for x in range(x_in0, x_in1) for y in range(y_in0, y_in1)
    }
    omega = _normalize_sites(outer.difference(inner))

    y_mid0 = y0 + Lv1
    y_mid1 = y0 + Lv1 + Lv2
    x_left0 = x0
    x_left1 = x0 + Lh1
    x_right0 = x0 + Lh1 + Lh2
    x_right1 = x0 + 2 * Lh1 + Lh2

    a_sites = [
        (x, y)
        for (x, y) in omega
        if y_mid0 <= y < y_mid1 and x_left0 <= x < x_left1
    ]
    c_sites = [
        (x, y)
        for (x, y) in omega
        if y_mid0 <= y < y_mid1 and x_right0 <= x < x_right1
    ]

    a_set = set(a_sites)
    c_set = set(c_sites)
    b_sites = [site for site in omega if site not in a_set and site not in c_set]

    if a_set.intersection(c_set):
        raise AssertionError("A and C are not disjoint")
    if set(b_sites).intersection(a_set) or set(b_sites).intersection(c_set):
        raise AssertionError("B overlaps A or C")
    if set(omega) != a_set.union(c_set).union(b_sites):
        raise AssertionError("A, B, C do not cover Omega")

    meta = {
        "Wout": Wout,
        "Hout": Hout,
        "Win": Win,
        "Hin": Hin,
        "x0": x0,
        "y0": y0,
        "x_in0": x_in0,
        "x_in1": x_in1,
        "y_in0": y_in0,
        "y_in1": y_in1,
        "y_mid0": y_mid0,
        "y_mid1": y_mid1,
        "x_left0": x_left0,
        "x_left1": x_left1,
        "x_right0": x_right0,
        "x_right1": x_right1,
    }
    return omega, a_sites, b_sites, c_sites, meta


def plot_regions(
    A_sites: Sequence[Site],
    B_sites: Sequence[Site],
    C_sites: Sequence[Site],
    *,
    W: Optional[int] = None,
    H: Optional[int] = None,
    full_sites: Optional[Sequence[Site]] = None,
    show_full_lattice: bool = False,
    ax: object = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[object, object]:
    """Plot A/B/C sites for a quick geometry check."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    if full_sites:
        xs, ys = zip(*_normalize_sites(full_sites))
        ax.scatter(xs, ys, s=8, marker="s", color="#e0e0e0", label="Lattice")
    elif show_full_lattice and W is not None and H is not None:
        xs = [x for x in range(W) for _ in range(H)]
        ys = [y for _ in range(W) for y in range(H)]
        ax.scatter(xs, ys, s=8, marker="s", color="#e0e0e0", label="Lattice")

    if B_sites:
        xs, ys = zip(*B_sites)
        ax.scatter(xs, ys, s=12, marker="s", color="#9e9e9e", label="B")
    if A_sites:
        xs, ys = zip(*A_sites)
        ax.scatter(xs, ys, s=18, marker="s", color="#4f81bd", label="A")
    if C_sites:
        xs, ys = zip(*C_sites)
        ax.scatter(xs, ys, s=18, marker="s", color="#c0504d", label="C")

    if W is None or H is None:
        if full_sites:
            xs, ys = zip(*_normalize_sites(full_sites))
            W = max(xs) + 1
            H = max(ys) + 1
    if W is not None and H is not None:
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


__all__ = [
    "LaurentPoly",
    "build_geometry_ABC",
    "build_full_lattice_sites",
    "build_site_index",
    "build_stabilizer_matrix_laurent",
    "cmi_from_stabilizers",
    "conditional_mutual_information_from_stabilizer_matrix",
    "entanglement_entropy_from_stabilizer_matrix",
    "qubits_from_sites",
    "plot_regions",
]

if __name__ == "__main__":
    L = 100
    omega_full = build_full_lattice_sites(L)
    # f = LaurentPoly.from_terms([(0,0),(1,0)])
    # g = LaurentPoly.from_terms([(0,0),(0,1)])
    # f = LaurentPoly.from_terms([(0,0),(1,0),(1,1)])
    # g = LaurentPoly.from_terms([(0,0),(0,1),(1,1)])
    f = LaurentPoly.from_terms([(0,0),(1,0),(-1,3)])
    g = LaurentPoly.from_terms([(0,0),(0,1),(3,-1)])
    # f = LaurentPoly.from_terms([(0,0),(-1,3),(-1,4)])
    # g = LaurentPoly.from_terms([(0,0),(3,-1),(4,-1)])
    stab, site_index = build_stabilizer_matrix_laurent(
        omega_full, f, g, strict=True, check_commutation=True
    )
    # print("site index:", site_index)
    # print("Stabilizer matrix shape:", stab[2,:])
    lABC = 18

    Lv1 = lABC  
    Lv2 = lABC
    Lv3 = lABC - 6
    l_list = [-6, -5, -4, -3, -2, 0, 2, 4, 6, 10, 14]
    I_AB_Cboth_list = []
    I_AB_Cedge1_list = []
    I_AB_Cedge2_list = []
    for l in l_list:
        Lh1 = lABC + l
        Lh2 = lABC + l
        omega, A_sites, B_sites, C_sites, meta = build_geometry_ABC(
            Lv1, Lv2, Lv3, Lh1, Lh2, lattice_size=L
        )
        plot_regions(
            A_sites,
            B_sites,
            C_sites,
            W=L,
            H=L,
            full_sites=omega_full,
            save_path="abc_regions.png",
        )

        A = qubits_from_sites(A_sites, site_index, include_edge1=True, include_edge2=True)
        B = qubits_from_sites(B_sites, site_index, include_edge1=True, include_edge2=True)
        A_edge1 = qubits_from_sites(A_sites, site_index, include_edge1=True, include_edge2=False)
        A_edge2 = qubits_from_sites(A_sites, site_index, include_edge1=False, include_edge2=True)
        B_edge1 = qubits_from_sites(B_sites, site_index, include_edge1=True, include_edge2=False)
        B_edge2 = qubits_from_sites(B_sites, site_index, include_edge1=False, include_edge2=True)
        C_both = qubits_from_sites(C_sites, site_index, include_edge1=True, include_edge2=True)
        C_edge1 = qubits_from_sites(C_sites, site_index, include_edge1=True, include_edge2=False)
        C_edge2 = qubits_from_sites(C_sites, site_index, include_edge1=False, include_edge2=True)

        n = 2 * len(site_index)
        # I_ABCedge1 = cmi_from_stabilizers(stab, A, B_edge1, C_both, num_qubits=n)
        # I_ABCedge2 = cmi_from_stabilizers(stab, A, B_edge2, C_both, num_qubits=n)
        # print(r"I(A_v:C_{v}|B_v)) =", I_ABCedge1)

        I_AB_Cboth = cmi_from_stabilizers(stab, A, B, C_both, num_qubits=n)
        I_AB_Cedge1 = cmi_from_stabilizers(stab, A, B, C_edge1, num_qubits=n)
        I_AB_Cedge2 = cmi_from_stabilizers(stab, A, B, C_edge2, num_qubits=n)
        # I_AB_Cedge1 = cmi_from_stabilizers(stab, A, B_edge1, C_both, num_qubits=n)
        # I_AB_Cedge2 = cmi_from_stabilizers(stab, A, B_edge2, C_both, num_qubits=n)
        print(r"I(A:C_{vh}|B)=", I_AB_Cboth)
        print(r"I(A:C_{v}|B)=", I_AB_Cedge1)
        print(r"I(A:C_{h}|B)=", I_AB_Cedge2)

        I_AB_Cboth_list.append(I_AB_Cboth)
        I_AB_Cedge1_list.append(I_AB_Cedge1)
        I_AB_Cedge2_list.append(I_AB_Cedge2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(l_list, I_AB_Cboth_list, s=50, marker="o", color="#4f81bd", label=r"I(A:B|C_{vh})")
    ax.scatter(l_list, I_AB_Cedge1_list, s=50, marker="o", color="#c0504d", label=r"I(A:B|C_{v})")
    ax.scatter(l_list, I_AB_Cedge2_list, s=50, marker="o", color="#9e9e9e", label=r"I(A:B|C_{h})")
    f_str = _laurent_poly_string(f)
    g_str = _laurent_poly_string(g)
    file_stub = f"I_AC_B_{_laurent_poly_slug(f)}_and_{_laurent_poly_slug(g)}"
    ax.set_xlabel("l")
    ax.set_ylabel("I(A:C|B)")
    ax.legend(loc="upper right")
    ax.set_title(f"{f_str} and {g_str} TEE anyon model")
    fig.savefig(f"{file_stub}.png", dpi=200, bbox_inches="tight")
