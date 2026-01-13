import numpy as np
import pytest
import stim

from BB_stab_tor import (
    entanglement_entropy,
    entanglement_entropy_from_stabilizer_matrix,
    mutual_information_from_stabilizer_matrix,
)


def stabilizer_matrix_from_tableau(tableau: stim.Tableau) -> np.ndarray:
    stabs = tableau.to_stabilizers()
    num_qubits = len(stabs)
    mat = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
    for i, stab in enumerate(stabs):
        xs, zs = stab.to_numpy()
        mat[i, :num_qubits] = xs
        mat[i, num_qubits:] = zs
    return mat


def entropy_from_state_vector(
    state: np.ndarray, subsystem: list[int], num_qubits: int
) -> float:
    if not subsystem or len(subsystem) == num_qubits:
        return 0.0
    subsystem = sorted(set(int(q) for q in subsystem))
    traced = [q for q in range(num_qubits) if q not in subsystem]
    psi = state.reshape([2] * num_qubits)
    perm = subsystem + traced
    psi_perm = np.transpose(psi, axes=perm)
    dim_keep = 2 ** len(subsystem)
    dim_trace = 2 ** len(traced)
    psi_mat = psi_perm.reshape(dim_keep, dim_trace)
    rho = psi_mat @ psi_mat.conj().T
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


@pytest.mark.parametrize(
    ("tableau", "subsystem"),
    [
        (stim.Tableau(2), [0]),
        (stim.Tableau.from_circuit(stim.Circuit("H 0\nCNOT 0 1")), [0]),
        (stim.Tableau.from_circuit(stim.Circuit("H 0\nCNOT 0 1\nCNOT 0 2")), [0, 1]),
    ],
)
def test_entanglement_entropy_pure_states(tableau: stim.Tableau, subsystem: list[int]) -> None:
    state = np.array(tableau.to_state_vector(), dtype=np.complex128)
    num_qubits = int(np.log2(state.size))
    assert 2**num_qubits == state.size
    expected = entropy_from_state_vector(state, subsystem, num_qubits=num_qubits)
    assert entanglement_entropy(tableau, subsystem) == pytest.approx(expected, abs=1e-7)

    stab_matrix = stabilizer_matrix_from_tableau(tableau)
    assert entanglement_entropy_from_stabilizer_matrix(stab_matrix, subsystem) == pytest.approx(
        expected, abs=1e-7
    )


def test_entanglement_entropy_maximally_mixed() -> None:
    stabilizer_matrix = np.zeros((0, 4), dtype=np.uint8)
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0]) == 1
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [1]) == 1
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0, 1]) == 2
    assert mutual_information_from_stabilizer_matrix(stabilizer_matrix, [0], [1]) == 0


def test_entanglement_entropy_product_pure_mixed() -> None:
    stabilizer_matrix = np.array([[0, 0, 1, 0]], dtype=np.uint8)
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0]) == 0
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [1]) == 1
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0, 1]) == 1
    assert mutual_information_from_stabilizer_matrix(stabilizer_matrix, [0], [1]) == 0


def test_entanglement_entropy_classical_correlation() -> None:
    stabilizer_matrix = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0]) == 1
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [1]) == 1
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0, 1]) == 1
    assert mutual_information_from_stabilizer_matrix(stabilizer_matrix, [0], [1]) == 1


def test_entanglement_entropy_bell_pair_product() -> None:
    tableau = stim.Tableau.from_circuit(stim.Circuit("H 0\nCNOT 0 1\nH 2\nCNOT 2 3"))
    stabilizer_matrix = stabilizer_matrix_from_tableau(tableau)
    assert entanglement_entropy(tableau, [0]) == 1
    assert entanglement_entropy(tableau, [0, 1]) == 0
    assert entanglement_entropy(tableau, [0, 2]) == 2
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0]) == 1
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0, 1]) == 0
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0, 2]) == 2


def test_entanglement_entropy_pure_complement_symmetry() -> None:
    tableau = stim.Tableau.from_circuit(
        stim.Circuit("H 0\nCNOT 0 1\nH 2\nCNOT 2 3\nCNOT 1 2")
    )
    stabilizer_matrix = stabilizer_matrix_from_tableau(tableau)
    subsystem = [0, 2]
    complement = [1, 3]
    assert entanglement_entropy(tableau, subsystem) == entanglement_entropy(tableau, complement)
    assert entanglement_entropy_from_stabilizer_matrix(
        stabilizer_matrix, subsystem
    ) == entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, complement)


def test_entanglement_entropy_partial_pure_state() -> None:
    stabilizer_matrix = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0, 1, 2]) == 1
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [0, 1]) == 0
    assert entanglement_entropy_from_stabilizer_matrix(stabilizer_matrix, [2]) == 1
