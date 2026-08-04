from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

from experiments import PROJECT_ROOT
from script import utility


OFFICIAL_ADJ_PATH = PROJECT_ROOT / "data" / "pemsd7-m" / "adj_official_stgcn.npz"

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "gso_construction_verification"


def original_scaled_laplacian(
    adjacency: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Reproduce the original TensorFlow repository's
    scaled_laplacian(W) implementation.

    Important detail:
    for an isolated node with degree 0, the unscaled
    Laplacian row remains zero. After scaling and
    subtracting the identity, its diagonal becomes -1.
    """

    adjacency = np.asarray(adjacency, dtype=np.float64)

    node_count = adjacency.shape[0]
    degree = np.sum(adjacency, axis=1)

    laplacian = -adjacency.copy()

    laplacian[np.diag_indices_from(laplacian)] = degree

    for row in range(node_count):
        for column in range(node_count):
            if degree[row] > 0 and degree[column] > 0:
                laplacian[row, column] = laplacian[row, column] / np.sqrt(
                    degree[row] * degree[column]
                )

    largest_eigenvalue = float(eigs(laplacian, k=1, which="LR")[0][0].real)

    scaled_laplacian = 2.0 * laplacian / largest_eigenvalue - np.identity(
        node_count, dtype=np.float64
    )

    return (scaled_laplacian, largest_eigenvalue, degree)


def current_utility_gso(adjacency_sparse: sp.spmatrix) -> tuple[np.ndarray, list[str]]:
    """
    Build the GSO using the current repository utility code.
    Any SciPy warnings are captured for the report.
    """

    captured_messages = []

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        normalized_laplacian = utility.calc_gso(adjacency_sparse, "sym_norm_lap")

        scaled_gso = utility.calc_chebynet_gso(normalized_laplacian)

    for item in captured:
        captured_messages.append(f"{item.category.__name__}: " f"{item.message}")

    return (scaled_gso.toarray().astype(np.float64), captured_messages)


def matrix_statistics(matrix: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(matrix, dtype=np.float64)

    return {
        "shape": list(matrix.shape),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "mean": float(matrix.mean()),
        "nonzero_count": int(np.count_nonzero(matrix)),
        "diagonal_min": float(np.diag(matrix).min()),
        "diagonal_max": float(np.diag(matrix).max()),
        "symmetry_max_abs_difference": float(np.max(np.abs(matrix - matrix.T))),
        "finite": bool(np.isfinite(matrix).all()),
    }


def comparison_statistics(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )

    return {
        "max_abs_difference": float(np.max(np.abs(difference))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "exact_equal": bool(np.array_equal(first, second)),
        "allclose_atol_1e-8": bool(np.allclose(first, second, rtol=0.0, atol=1e-8)),
        "allclose_atol_1e-6": bool(np.allclose(first, second, rtol=0.0, atol=1e-6)),
    }


def main() -> None:
    print("=" * 78)
    print("Official PeMSD7(M) GSO Construction Verification")
    print("=" * 78)

    if not OFFICIAL_ADJ_PATH.exists():
        raise FileNotFoundError(
            "Official adjacency matrix not found:\n"
            f"{OFFICIAL_ADJ_PATH}\n\n"
            "Run `python -m experiments.verification.prepare_official_adjacency` first."
        )

    adjacency_sparse = sp.load_npz(OFFICIAL_ADJ_PATH).tocsc()

    adjacency = adjacency_sparse.toarray().astype(np.float64)

    (original_gso, original_lambda_max, degree) = original_scaled_laplacian(adjacency)

    (current_gso, current_warnings) = current_utility_gso(adjacency_sparse)

    isolated_mask = degree == 0

    isolated_indices = np.flatnonzero(isolated_mask)

    original_isolated_diagonal = np.diag(original_gso)[isolated_mask]

    current_isolated_diagonal = np.diag(current_gso)[isolated_mask]

    comparison = comparison_statistics(current_gso, original_gso)

    report = {
        "official_adjacency_path": str(OFFICIAL_ADJ_PATH),
        "adjacency_nonzero_count": int(adjacency_sparse.nnz),
        "isolated_node_count": int(isolated_mask.sum()),
        "isolated_node_indices": isolated_indices.tolist(),
        "original_largest_eigenvalue": original_lambda_max,
        "current_utility_warnings": current_warnings,
        "current_utility_gso": matrix_statistics(current_gso),
        "original_tensorflow_gso": matrix_statistics(original_gso),
        "isolated_node_diagonal": {
            "current_unique_values": np.unique(
                np.round(current_isolated_diagonal, decimals=12)
            ).tolist(),
            "original_unique_values": np.unique(
                np.round(original_isolated_diagonal, decimals=12)
            ).tolist(),
        },
        "current_vs_original": comparison,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIRECTORY / "gso_comparison.json"

    text_path = OUTPUT_DIRECTORY / "gso_comparison.txt"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "Official PeMSD7(M) GSO Construction Verification",
        "=" * 62,
        "",
        f"Adjacency non-zero count: " f"{adjacency_sparse.nnz}",
        f"Isolated node count: " f"{isolated_mask.sum()}",
        f"Isolated node indices: " f"{isolated_indices.tolist()}",
        "",
        "Original TensorFlow construction:",
        f"  largest eigenvalue: " f"{original_lambda_max:.12g}",
        f"  GSO min: " f"{original_gso.min():.12g}",
        f"  GSO max: " f"{original_gso.max():.12g}",
        "",
        "Current PyTorch utility construction:",
        f"  GSO min: " f"{current_gso.min():.12g}",
        f"  GSO max: " f"{current_gso.max():.12g}",
        "",
        "Isolated-node diagonal values:",
        (
            "  current utility: "
            f"{report['isolated_node_diagonal']['current_unique_values']}"
        ),
        (
            "  original TensorFlow: "
            f"{report['isolated_node_diagonal']['original_unique_values']}"
        ),
        "",
        "Current vs original:",
        ("  max abs difference: " f"{comparison['max_abs_difference']:.12g}"),
        ("  mean abs difference: " f"{comparison['mean_abs_difference']:.12g}"),
        ("  RMSE difference: " f"{comparison['rmse_difference']:.12g}"),
        ("  exact equal: " f"{comparison['exact_equal']}"),
        ("  allclose atol=1e-8: " f"{comparison['allclose_atol_1e-8']}"),
        ("  allclose atol=1e-6: " f"{comparison['allclose_atol_1e-6']}"),
        "",
        "Captured warnings:",
    ]

    if current_warnings:
        for message in current_warnings:
            lines.append(f"  - {message}")
    else:
        lines.append("  None")

    if comparison["allclose_atol_1e-6"]:
        conclusion = (
            "CONCLUSION: The current utility GSO matches "
            "the original TensorFlow scaled Laplacian."
        )
    else:
        conclusion = (
            "CONCLUSION: The current utility GSO does not "
            "match the original TensorFlow scaled Laplacian. "
            "A separate original-GSO option is required before "
            "claiming full graph-pipeline alignment."
        )

    lines.extend(["", conclusion])

    text_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("Adjacency non-zero count:", adjacency_sparse.nnz)
    print("Isolated node count:", int(isolated_mask.sum()))
    print("Isolated node indices:", isolated_indices.tolist())
    print()
    print("Original largest eigenvalue:", original_lambda_max)
    print("Original GSO range:", (float(original_gso.min()), float(original_gso.max())))
    print("Current GSO range:", (float(current_gso.min()), float(current_gso.max())))
    print()
    print(
        "Isolated-node diagonal " "(current):",
        report["isolated_node_diagonal"]["current_unique_values"],
    )
    print(
        "Isolated-node diagonal " "(original):",
        report["isolated_node_diagonal"]["original_unique_values"],
    )
    print()
    print("Maximum absolute GSO difference:", comparison["max_abs_difference"])
    print("Allclose at 1e-6:", comparison["allclose_atol_1e-6"])
    print()
    print(conclusion)
    print()
    print("Generated files:")
    print(json_path)
    print(text_path)


if __name__ == "__main__":
    main()
