import torch

from paper_main import prepare_graph


def main():
    """
    Verify that paper_main.py can switch between the
    current adjacency matrix and the official STGCN matrix.
    """

    device = torch.device("cpu")

    results = {}

    for graph_source in ("current", "official"):
        gso, n_vertex, graph_info = prepare_graph(
            dataset="pemsd7-m",
            graph_source=graph_source,
            device=device,
        )

        results[graph_source] = {
            "gso": gso,
            "n_vertex": n_vertex,
            "graph_info": graph_info,
        }

        print("=" * 72)
        print(f"Graph source: {graph_source}")
        print("=" * 72)
        print("Path:", graph_info["path"])
        print("Nodes:", n_vertex)
        print("Adjacency non-zero values:", graph_info["nonzero_count"])
        print(
            "Adjacency non-zero diagonal values:",
            graph_info["diagonal_nonzero_count"],
        )
        print("GSO shape:", tuple(gso.shape))
        print("GSO minimum:", gso.min().item())
        print("GSO maximum:", gso.max().item())
        print("GSO finite:", torch.isfinite(gso).all().item())
        print()

    current = results["current"]
    official = results["official"]

    assert current["n_vertex"] == 228
    assert official["n_vertex"] == 228

    assert tuple(current["gso"].shape) == (228, 228)
    assert tuple(official["gso"].shape) == (228, 228)

    assert torch.isfinite(current["gso"]).all()
    assert torch.isfinite(official["gso"]).all()

    assert (
        official["graph_info"]["nonzero_count"]
        == 1664
    )

    assert (
        official["graph_info"][
            "diagonal_nonzero_count"
        ]
        == 0
    )

    gso_max_abs_difference = torch.max(
        torch.abs(
            current["gso"]
            - official["gso"]
        )
    ).item()

    print("=" * 72)
    print("Comparison")
    print("=" * 72)
    print(
        "Current adjacency non-zero values:",
        current["graph_info"]["nonzero_count"],
    )
    print(
        "Official adjacency non-zero values:",
        official["graph_info"]["nonzero_count"],
    )
    print(
        "Maximum absolute GSO difference:",
        gso_max_abs_difference,
    )
    print(
        "GSO exactly equal:",
        torch.equal(
            current["gso"],
            official["gso"],
        ),
    )

    if gso_max_abs_difference == 0:
        raise AssertionError(
            "The two graph sources unexpectedly produced "
            "identical GSOs."
        )

    print()
    print(
        "PASS: paper_main.py can load both graph sources, "
        "and the official graph is correctly isolated."
    )


if __name__ == "__main__":
    main()
