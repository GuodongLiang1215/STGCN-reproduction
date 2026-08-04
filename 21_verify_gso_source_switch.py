import torch

from paper_main import prepare_graph


def main():
    """
    Verify that the official adjacency matrix can be used
    with both the current PyTorch GSO construction and the
    original TensorFlow STGCN scaled-Laplacian construction.
    """

    device = torch.device("cpu")

    results = {}

    for gso_source in ("current", "original"):
        (
            gso,
            n_vertex,
            graph_info,
        ) = prepare_graph(
            dataset="pemsd7-m",
            graph_source="official",
            device=device,
            gso_source=gso_source,
        )

        results[gso_source] = {
            "gso": gso,
            "n_vertex": n_vertex,
            "graph_info": graph_info,
        }

        print("=" * 72)
        print(f"GSO source: {gso_source}")
        print("=" * 72)
        print("Graph source:", graph_info["source"])
        print("Graph path:", graph_info["path"])
        print("Nodes:", n_vertex)
        print(
            "Adjacency non-zero values:",
            graph_info["nonzero_count"],
        )
        print(
            "Isolated-node count:",
            graph_info["isolated_node_count"],
        )
        print(
            "Isolated-node indices:",
            graph_info["isolated_node_indices"],
        )
        print("GSO shape:", tuple(gso.shape))
        print("GSO minimum:", gso.min().item())
        print("GSO maximum:", gso.max().item())
        print("GSO finite:", torch.isfinite(gso).all().item())

        if graph_info["lambda_max"] is not None:
            print(
                "Largest eigenvalue:",
                graph_info["lambda_max"],
            )

        isolated_indices = graph_info[
            "isolated_node_indices"
        ]

        isolated_diagonal = torch.diag(gso)[
            isolated_indices
        ]

        print(
            "Isolated-node diagonal values:",
            isolated_diagonal.tolist(),
        )
        print()

    current = results["current"]
    original = results["original"]

    assert current["n_vertex"] == 228
    assert original["n_vertex"] == 228

    assert tuple(current["gso"].shape) == (228, 228)
    assert tuple(original["gso"].shape) == (228, 228)

    assert torch.isfinite(current["gso"]).all()
    assert torch.isfinite(original["gso"]).all()

    assert (
        original["graph_info"]["nonzero_count"]
        == 1664
    )

    assert (
        original["graph_info"]["isolated_node_count"]
        == 3
    )

    isolated_indices = original[
        "graph_info"
    ]["isolated_node_indices"]

    assert isolated_indices == [13, 135, 226]

    current_isolated_diagonal = torch.diag(
        current["gso"]
    )[isolated_indices]

    original_isolated_diagonal = torch.diag(
        original["gso"]
    )[isolated_indices]

    assert torch.allclose(
        current_isolated_diagonal,
        torch.zeros_like(
            current_isolated_diagonal
        ),
        atol=1e-6,
        rtol=0.0,
    )

    assert torch.allclose(
        original_isolated_diagonal,
        -torch.ones_like(
            original_isolated_diagonal
        ),
        atol=1e-6,
        rtol=0.0,
    )

    gso_max_abs_difference = torch.max(
        torch.abs(
            current["gso"]
            - original["gso"]
        )
    ).item()

    print("=" * 72)
    print("Comparison")
    print("=" * 72)
    print(
        "Maximum absolute GSO difference:",
        gso_max_abs_difference,
    )
    print(
        "GSO exactly equal:",
        torch.equal(
            current["gso"],
            original["gso"],
        ),
    )

    if gso_max_abs_difference < 0.9:
        raise AssertionError(
            "The two GSO constructions are more similar "
            "than expected. Check the implementation."
        )

    print()
    print(
        "PASS: official adjacency can use both GSO "
        "constructions, and the original TensorFlow "
        "isolated-node behavior is reproduced."
    )


if __name__ == "__main__":
    main()
