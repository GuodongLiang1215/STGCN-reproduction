from __future__ import annotations

import csv
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from experiments import PROJECT_ROOT


CURRENT_ADJ_PATH = PROJECT_ROOT / "data" / "pemsd7-m" / "adj.npz"

OFFICIAL_DATA_DIRECTORY = PROJECT_ROOT / "data" / "pemsd7-official"

OFFICIAL_ZIP_PATH = OFFICIAL_DATA_DIRECTORY / "PeMSD7_Full.zip"

OFFICIAL_ZIP_URL = (
    "https://github.com/VeritasYin/"
    "STGCN_IJCAI-18/raw/master/"
    "dataset/PeMSD7_Full.zip"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "results" / "graph_matrix_verification"


def sha256_file(path: Path) -> str:
    """Calculate the SHA256 hash of a file."""

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def canonical_array_sha256(array: np.ndarray) -> str:
    """
    Calculate a format-independent hash for a matrix.

    The matrix is converted to contiguous float64 bytes first,
    so CSV and NPZ storage formats can still be compared.
    """

    canonical = np.ascontiguousarray(array, dtype=np.float64)

    return hashlib.sha256(canonical.tobytes()).hexdigest()


def matrix_statistics(name: str, matrix: np.ndarray) -> dict[str, object]:
    """Return useful descriptive statistics for one matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)

    return {
        "name": name,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "mean": float(matrix.mean()),
        "nonzero_count": int(np.count_nonzero(matrix)),
        "diagonal_max_abs": float(np.max(np.abs(np.diag(matrix)))),
        "symmetry_max_abs_difference": float(np.max(np.abs(matrix - matrix.T))),
        "canonical_sha256": canonical_array_sha256(matrix),
    }


def compare_matrices(
    comparison_name: str, first: np.ndarray, second: np.ndarray
) -> dict[str, object]:
    """Compare two matrices element by element."""

    first = np.asarray(first, dtype=np.float64)

    second = np.asarray(second, dtype=np.float64)

    if first.shape != second.shape:
        return {
            "comparison": comparison_name,
            "same_shape": False,
            "first_shape": str(first.shape),
            "second_shape": str(second.shape),
            "max_abs_difference": np.nan,
            "mean_abs_difference": np.nan,
            "rmse_difference": np.nan,
            "allclose_atol_1e-8": False,
            "allclose_atol_1e-6": False,
            "allclose_atol_1e-5": False,
            "exact_equal": False,
        }

    difference = first - second

    return {
        "comparison": comparison_name,
        "same_shape": True,
        "first_shape": str(first.shape),
        "second_shape": str(second.shape),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "allclose_atol_1e-8": bool(np.allclose(first, second, rtol=0.0, atol=1e-8)),
        "allclose_atol_1e-6": bool(np.allclose(first, second, rtol=0.0, atol=1e-6)),
        "allclose_atol_1e-5": bool(np.allclose(first, second, rtol=0.0, atol=1e-5)),
        "exact_equal": bool(np.array_equal(first, second)),
    }


def original_stgcn_weight_matrix(
    source_matrix: np.ndarray, sigma2: float = 0.1, epsilon: float = 0.5
) -> np.ndarray:
    """
    Reproduce the original TensorFlow repository's
    weight_matrix() preprocessing.

    For a non-binary source matrix:

        W = W / 10000
        A = exp(-(W^2) / sigma2)
        A[A < epsilon] = 0
        diagonal = 0
    """

    source_matrix = np.asarray(source_matrix, dtype=np.float64)

    unique_values = set(np.unique(source_matrix).tolist())

    if unique_values == {0.0, 1.0}:
        return source_matrix.copy()

    node_count = source_matrix.shape[0]

    scaled = source_matrix / 10000.0
    squared = scaled * scaled

    affinity = np.exp(-squared / sigma2)

    off_diagonal_mask = np.ones((node_count, node_count), dtype=np.float64) - np.eye(
        node_count, dtype=np.float64
    )

    weighted_adjacency = affinity * (affinity >= epsilon) * off_diagonal_mask

    return weighted_adjacency


def max_symmetrize(matrix: np.ndarray) -> np.ndarray:
    """
    Match the max-based symmetrization used by the
    modern PyTorch utility code:

        max(A, A.T)
    """

    return np.maximum(matrix, matrix.T)


def prepare_official_dataset() -> Path:
    """Download and extract the official PeMSD7 archive."""

    OFFICIAL_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    existing_files = list(OFFICIAL_DATA_DIRECTORY.rglob("PeMSD7_W_228.csv"))

    if existing_files:
        return existing_files[0]

    if not OFFICIAL_ZIP_PATH.exists():
        print("Downloading official PeMSD7 archive...")
        print(OFFICIAL_ZIP_URL)

        urllib.request.urlretrieve(OFFICIAL_ZIP_URL, OFFICIAL_ZIP_PATH)

        print("Downloaded:", OFFICIAL_ZIP_PATH)
    else:
        print("Using existing ZIP:", OFFICIAL_ZIP_PATH)

    print("Extracting official dataset...")

    with zipfile.ZipFile(OFFICIAL_ZIP_PATH, "r") as archive:
        archive.extractall(OFFICIAL_DATA_DIRECTORY)

    matrix_files = list(OFFICIAL_DATA_DIRECTORY.rglob("PeMSD7_W_228.csv"))

    if not matrix_files:
        raise FileNotFoundError(
            "The ZIP was extracted, but " "PeMSD7_W_228.csv was not found."
        )

    return matrix_files[0]


def main() -> None:
    print("=" * 78)
    print("PeMSD7(M) Graph Matrix Verification")
    print("=" * 78)

    if not CURRENT_ADJ_PATH.exists():
        raise FileNotFoundError(
            f"Current adjacency matrix not found: " f"{CURRENT_ADJ_PATH}"
        )

    official_matrix_path = prepare_official_dataset()

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nCurrent matrix:")
    print(CURRENT_ADJ_PATH)

    print("\nOfficial source matrix:")
    print(official_matrix_path)

    # Current PyTorch adjacency
    current_sparse = sp.load_npz(CURRENT_ADJ_PATH)

    current_adjacency = current_sparse.toarray().astype(np.float64)

    # Official source matrix
    official_source = pd.read_csv(official_matrix_path, header=None).to_numpy(
        dtype=np.float64
    )

    if official_source.shape != (228, 228):
        raise ValueError(
            "Unexpected official matrix shape: " f"{official_source.shape}"
        )

    # Reproduce original TensorFlow preprocessing
    official_weighted = original_stgcn_weight_matrix(
        official_source, sigma2=0.1, epsilon=0.5
    )

    # Also compare against the symmetrized version,
    # because the modern PyTorch utility symmetrizes A.
    official_weighted_sym = max_symmetrize(official_weighted)

    matrices = {
        "current_adj_npz": current_adjacency,
        "official_source_csv": official_source,
        "official_after_original_weight_matrix": official_weighted,
        "official_after_original_weight_matrix_symmetrized": official_weighted_sym,
    }

    statistics = [matrix_statistics(name, matrix) for name, matrix in matrices.items()]

    comparisons = [
        compare_matrices(
            ("current_adj_npz vs " "official_source_csv"),
            current_adjacency,
            official_source,
        ),
        compare_matrices(
            ("current_adj_npz vs " "official_after_original_weight_matrix"),
            current_adjacency,
            official_weighted,
        ),
        compare_matrices(
            (
                "current_adj_npz vs "
                "official_after_original_weight_matrix_"
                "symmetrized"
            ),
            current_adjacency,
            official_weighted_sym,
        ),
    ]

    # Save CSV files

    statistics_path = OUTPUT_DIRECTORY / "matrix_statistics.csv"

    comparison_path = OUTPUT_DIRECTORY / "matrix_comparison.csv"

    with statistics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(statistics[0].keys()))
        writer.writeheader()
        writer.writerows(statistics)

    with comparison_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    # Human-readable report

    current_vs_original = comparisons[1]
    current_vs_sym = comparisons[2]

    if current_vs_original["allclose_atol_1e-8"]:
        conclusion = (
            "CONCLUSION: The current adj.npz "
            "matches the adjacency produced by "
            "the original STGCN preprocessing "
            "within atol=1e-8."
        )
    elif current_vs_sym["allclose_atol_1e-8"]:
        conclusion = (
            "CONCLUSION: The current adj.npz "
            "matches the max-symmetrized version "
            "of the original STGCN adjacency "
            "within atol=1e-8."
        )
    elif current_vs_original["allclose_atol_1e-5"]:
        conclusion = (
            "CONCLUSION: The matrices are nearly "
            "identical, but contain small numerical "
            "differences up to atol=1e-5."
        )
    elif current_vs_sym["allclose_atol_1e-5"]:
        conclusion = (
            "CONCLUSION: The current matrix is "
            "nearly identical to the symmetrized "
            "official adjacency, with differences "
            "up to atol=1e-5."
        )
    else:
        conclusion = (
            "CONCLUSION: The current adj.npz does "
            "not match the official graph matrix "
            "under the tested preprocessing paths. "
            "This is a plausible source of the "
            "remaining reproduction gap."
        )

    report_lines = [
        "PeMSD7(M) Graph Matrix Verification",
        "=" * 60,
        "",
        f"Current NPZ: {CURRENT_ADJ_PATH}",
        f"Current NPZ SHA256: " f"{sha256_file(CURRENT_ADJ_PATH)}",
        "",
        f"Official CSV: {official_matrix_path}",
        f"Official CSV SHA256: " f"{sha256_file(official_matrix_path)}",
        "",
        "Matrix statistics:",
    ]

    for row in statistics:
        report_lines.extend(
            [
                "",
                f"[{row['name']}]",
                (f"shape = " f"({row['rows']}, " f"{row['columns']})"),
                f"min = {row['minimum']:.12g}",
                f"max = {row['maximum']:.12g}",
                f"mean = {row['mean']:.12g}",
                (f"nonzero_count = " f"{row['nonzero_count']}"),
                ("diagonal_max_abs = " f"{row['diagonal_max_abs']:.12g}"),
                (
                    "symmetry_max_abs_difference = "
                    f"{row['symmetry_max_abs_difference']:.12g}"
                ),
                ("canonical_sha256 = " f"{row['canonical_sha256']}"),
            ]
        )

    report_lines.extend(["", "Comparisons:"])

    for row in comparisons:
        report_lines.extend(
            [
                "",
                f"[{row['comparison']}]",
                (f"max_abs_difference = " f"{row['max_abs_difference']}"),
                (f"mean_abs_difference = " f"{row['mean_abs_difference']}"),
                (f"rmse_difference = " f"{row['rmse_difference']}"),
                (f"exact_equal = " f"{row['exact_equal']}"),
                (f"allclose_atol_1e-8 = " f"{row['allclose_atol_1e-8']}"),
                (f"allclose_atol_1e-6 = " f"{row['allclose_atol_1e-6']}"),
                (f"allclose_atol_1e-5 = " f"{row['allclose_atol_1e-5']}"),
            ]
        )

    report_lines.extend(["", conclusion])

    report_path = OUTPUT_DIRECTORY / "graph_matrix_verification.txt"

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print("Matrix statistics")
    print("=" * 78)

    for row in statistics:
        print(
            f"{row['name']}: "
            f"shape=({row['rows']}, "
            f"{row['columns']}), "
            f"min={row['minimum']:.6g}, "
            f"max={row['maximum']:.6g}, "
            f"nnz={row['nonzero_count']}, "
            f"sym_diff="
            f"{row['symmetry_max_abs_difference']:.6g}"
        )

    print("\n" + "=" * 78)
    print("Comparisons")
    print("=" * 78)

    for row in comparisons:
        print(row["comparison"])
        print("  max abs difference:", row["max_abs_difference"])
        print("  mean abs difference:", row["mean_abs_difference"])
        print("  exact equal:", row["exact_equal"])
        print("  allclose (1e-8):", row["allclose_atol_1e-8"])
        print("  allclose (1e-5):", row["allclose_atol_1e-5"])

    print("\n" + conclusion)

    print("\nGenerated files:")
    print(statistics_path)
    print(comparison_path)
    print(report_path)


if __name__ == "__main__":
    main()
