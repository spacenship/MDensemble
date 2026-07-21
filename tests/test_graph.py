import torch

from protein_flow.geometry.features import rbf_encode
from protein_flow.geometry.graph import (
    SEQUENCE_EDGE_BACKWARD,
    SEQUENCE_EDGE_FORWARD,
    build_geometric_graph,
    build_sequence_graph,
)


def test_sequence_graph_forward_backward_edges_and_padding():
    # batch 0: length 5, all valid. batch 1: length 5, last 2 are padding.
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    graph = build_sequence_graph(mask)
    length = 5

    forward_edges = set(
        (int(s), int(d))
        for s, d, t in zip(*graph.edge_index, graph.edge_type)
        if t == SEQUENCE_EDGE_FORWARD
    )
    backward_edges = set(
        (int(s), int(d))
        for s, d, t in zip(*graph.edge_index, graph.edge_type)
        if t == SEQUENCE_EDGE_BACKWARD
    )

    # batch 0 (offset 0): edges 0-1,1-2,2-3,3-4 fully valid
    expected_forward_b0 = {(0, 1), (1, 2), (2, 3), (3, 4)}
    # batch 1 (offset 5): only 5-6,6-7 valid (index 7-8 and 8-9 touch padding)
    expected_forward_b1 = {(5, 6), (6, 7)}

    assert forward_edges == expected_forward_b0 | expected_forward_b1
    assert backward_edges == {(d, s) for s, d in expected_forward_b0 | expected_forward_b1}


def test_sequence_graph_short_sequence_no_crash():
    mask = torch.ones(2, 1, dtype=torch.bool)
    graph = build_sequence_graph(mask)
    assert graph.edge_index.shape == (2, 0)


def test_geometric_graph_knn_correctness_no_ties():
    # Points on a line with irregular spacing to avoid distance ties.
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0], [4.5, 0.0, 0.0], [7.5, 0.0, 0.0]]]
    )
    mask = torch.ones(1, 5, dtype=torch.bool)
    graph = build_geometric_graph(coords, mask, k=2)

    neighbors_of = {i: set() for i in range(5)}
    for s, d in zip(*graph.edge_index):
        neighbors_of[int(d)].add(int(s))

    # node 2 (x=2.5): nearest are node1 (dist1.5) and node3(dist2.0), not node0(2.5) tie-free check
    assert neighbors_of[2] == {1, 3}
    # node 0 (x=0.0): nearest are node1(1.0) and node2(2.5)
    assert neighbors_of[0] == {1, 2}
    # node 4 (x=7.5): nearest are node3(3.0) and node2(5.0)
    assert neighbors_of[4] == {3, 2}


def test_geometric_graph_no_cross_batch_edges_and_no_self_edges():
    torch.manual_seed(0)
    coords = torch.randn(3, 10, 3)
    mask = torch.ones(3, 10, dtype=torch.bool)
    graph = build_geometric_graph(coords, mask, k=4)

    length = 10
    src, dst = graph.edge_index
    assert torch.all(src // length == dst // length)
    assert torch.all(src != dst)


def test_geometric_graph_padding_excluded_from_both_roles():
    torch.manual_seed(1)
    coords = torch.randn(2, 8, 3)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True  # only first 4 residues valid per sample
    graph = build_geometric_graph(coords, mask, k=3)

    length = 8
    src, dst = graph.edge_index
    local_src = src % length
    local_dst = dst % length
    assert torch.all(local_src < 4)
    assert torch.all(local_dst < 4)


def test_geometric_graph_k_larger_than_valid_residues_is_safe():
    torch.manual_seed(2)
    coords = torch.randn(1, 6, 3)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[:, :3] = True  # only 3 valid residues, k requested = 10
    graph = build_geometric_graph(coords, mask, k=10)

    # each of the 3 valid nodes can have at most 2 neighbors (the other 2 valid nodes)
    length = 6
    src, dst = graph.edge_index
    for node in range(3):
        count = int((dst == node).sum())
        assert count == 2
    assert graph.edge_index.shape[1] == 3 * 2


def test_geometric_graph_radius_cutoff_removes_far_edges():
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    graph = build_geometric_graph(coords, mask, k=2, use_radius_cutoff=True, radius_cutoff=5.0)
    # node2 is far from everything; with cutoff=5 no edges should touch it
    src, dst = graph.edge_index
    assert not torch.any(src == 2)
    assert not torch.any(dst == 2)


def test_geometric_graph_relative_vectors_match_definition():
    torch.manual_seed(3)
    coords = torch.randn(2, 7, 3)
    mask = torch.ones(2, 7, dtype=torch.bool)
    graph = build_geometric_graph(coords, mask, k=3)
    coords_flat = coords.reshape(-1, 3)
    src, dst = graph.edge_index
    expected = coords_flat[src] - coords_flat[dst]
    torch.testing.assert_close(graph.relative_vectors, expected)
    expected_dist = expected.norm(dim=-1)
    torch.testing.assert_close(graph.distances, expected_dist, atol=1e-4, rtol=1e-4)


def test_rbf_encode_shape_and_peak_location():
    distances = torch.tensor([0.0, 5.0, 10.0])
    rbf = rbf_encode(distances, num_rbf=11, min_dist=0.0, max_dist=10.0)
    assert rbf.shape == (3, 11)
    # distance 5.0 should peak near the middle basis function (index 5, center=5.0)
    assert torch.argmax(rbf[1]).item() == 5
    assert torch.all(rbf >= 0.0) and torch.all(rbf <= 1.0 + 1e-6)
