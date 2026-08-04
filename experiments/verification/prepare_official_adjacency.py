from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from experiments import PROJECT_ROOT


OFFICIAL_DISTANCE_PATH = PROJECT_ROOT / "data" / "pemsd7-official" / "PeMSD7_W_228.csv"

CURRENT_ADJ_PATH = PROJECT_ROOT / "data" / "pemsd7-m" / "adj.npz"

OUTPUT_ADJ_PATH = PROJECT_ROOT / "data" / "pemsd7-m" / "adj_official_stgcn.npz"

OUTPUT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "pemsd7-m" / "adj_official_stgcn_metadata.json"
)

SIGMA2 = 0.1
EPSILON = 0.5
EXPECTED_NODE_COUNT = 228


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash of a file."""

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def matrix_summary(name: str, matrix: np.ndarray) -> dict[str, object]:
    """Return useful statistics for one matrix."""

    matrix = np.asarray(matrix, dtype=np.float64)

    return {
        "name": name,
        "shape": list(matrix.shape),
        "minimum": float(matrix.min()),
        "maximum": float(matrix.max()),
        "mean": float(matrix.mean()),
        "nonzero_count": int(np.count_nonzero(matrix)),
        "density": float(np.count_nonzero(matrix) / matrix.size),
        "diagonal_nonzero_count": int(np.count_nonzero(np.diag(matrix))),
        "diagonal_max_abs": float(np.max(np.abs(np.diag(matrix)))),
        "symmetry_max_abs_difference": float(np.max(np.abs(matrix - matrix.T))),
    }


def build_official_stgcn_adjacency(
    distance_matrix: np.ndarray, sigma2: float, epsilon: float
) -> np.ndarray:
    """
    Reproduce the graph preprocessing in the original
    STGCN TensorFlow repository.

    Steps:
      1. Divide distance values by 10000.
      2. Apply exp(-(distance^2) / sigma2).
      3. Keep values greater than or equal to epsilon.
      4. Remove the diagonal.
    """

    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)

    if distance_matrix.shape != (EXPECTED_NODE_COUNT, EXPECTED_NODE_COUNT):
        raise ValueError(
            "Unexpected distance matrix shape: " f"{distance_matrix.shape}"
        )

    scaled_distance = distance_matrix / 10000.0

    affinity = np.exp(-(scaled_distance * scaled_distance) / sigma2)

    off_diagonal_mask = np.ones_like(affinity, dtype=np.float64) - np.eye(
        EXPECTED_NODE_COUNT, dtype=np.float64
    )

    adjacency = affinity * (affinity >= epsilon) * off_diagonal_mask

    return adjacency


def main() -> None:
    print("=" * 78)
    print("Prepare Official STGCN PeMSD7(M) Adjacency")
    print("=" * 78)

    if not OFFICIAL_DISTANCE_PATH.exists():
        raise FileNotFoundError(
            "Official distance matrix not found:\n"
            f"{OFFICIAL_DISTANCE_PATH}\n\n"
            "Run `python -m experiments.verification.compare_graph_matrices` first."
        )

    distance_matrix = pd.read_csv(OFFICIAL_DISTANCE_PATH, header=None).to_numpy(
        dtype=np.float64
    )

    official_adjacency = build_official_stgcn_adjacency(
        distance_matrix=distance_matrix, sigma2=SIGMA2, epsilon=EPSILON
    )

    if not np.allclose(official_adjacency, official_adjacency.T, rtol=0.0, atol=1e-12):
        raise ValueError("The generated official adjacency " "is not symmetric.")

    if np.count_nonzero(np.diag(official_adjacency)) != 0:
        raise ValueError(
            "The generated official adjacency " "contains non-zero diagonal values."
        )

    OUTPUT_ADJ_PATH.parent.mkdir(parents=True, exist_ok=True)

    official_sparse = sp.csc_matrix(official_adjacency.astype(np.float32))

    sp.save_npz(OUTPUT_ADJ_PATH, official_sparse)

    # Load it back to make sure the saved file is valid.
    reloaded = sp.load_npz(OUTPUT_ADJ_PATH).toarray().astype(np.float64)

    save_max_abs_difference = float(np.max(np.abs(reloaded - official_adjacency)))

    if not np.allclose(reloaded, official_adjacency, rtol=0.0, atol=1e-6):
        raise ValueError(
            "The saved NPZ does not match the " "generated adjacency within atol=1e-6."
        )

    official_summary = matrix_summary("official_stgcn_adjacency", reloaded)

    metadata = {
        "source_distance_file": str(OFFICIAL_DISTANCE_PATH),
        "source_distance_sha256": sha256_file(OFFICIAL_DISTANCE_PATH),
        "output_adjacency_file": str(OUTPUT_ADJ_PATH),
        "output_adjacency_sha256": sha256_file(OUTPUT_ADJ_PATH),
        "construction": {
            "distance_divisor": 10000.0,
            "sigma2": SIGMA2,
            "epsilon": EPSILON,
            "diagonal_removed": True,
        },
        "save_reload_max_abs_difference": save_max_abs_difference,
        "official_adjacency_summary": official_summary,
    }

    if CURRENT_ADJ_PATH.exists():
        current_adjacency = sp.load_npz(CURRENT_ADJ_PATH).toarray().astype(np.float64)

        current_summary = matrix_summary("current_adj", current_adjacency)

        metadata["current_adjacency_file"] = str(CURRENT_ADJ_PATH)

        metadata["current_adjacency_sha256"] = sha256_file(CURRENT_ADJ_PATH)

        metadata["current_adjacency_summary"] = current_summary

        if current_adjacency.shape == reloaded.shape:
            difference = current_adjacency - reloaded

            metadata["current_vs_official"] = {
                "max_abs_difference": float(np.max(np.abs(difference))),
                "mean_abs_difference": float(np.mean(np.abs(difference))),
                "rmse_difference": float(np.sqrt(np.mean(difference**2))),
                "exact_equal": bool(np.array_equal(current_adjacency, reloaded)),
                "allclose_atol_1e-6": bool(
                    np.allclose(current_adjacency, reloaded, rtol=0.0, atol=1e-6)
                ),
            }

    OUTPUT_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nOfficial adjacency created:")
    print(OUTPUT_ADJ_PATH)

    print("\nMetadata saved:")
    print(OUTPUT_METADATA_PATH)

    print("\nOfficial adjacency statistics:")
    print("  shape:", tuple(official_summary["shape"]))
    print("  min:", official_summary["minimum"])
    print("  max:", official_summary["maximum"])
    print("  mean:", official_summary["mean"])
    print("  non-zero values:", official_summary["nonzero_count"])
    print("  density:", f"{official_summary['density']:.6%}")
    print("  non-zero diagonal values:", official_summary["diagonal_nonzero_count"])
    print("  symmetry max difference:", official_summary["symmetry_max_abs_difference"])
    print("  save/reload max difference:", save_max_abs_difference)

    print("\nImportant:")
    print("  The existing adj.npz was NOT overwritten.")
    print("  The new official graph is stored separately " "as adj_official_stgcn.npz.")


if __name__ == "__main__":
    main()
