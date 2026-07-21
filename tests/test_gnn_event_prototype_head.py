import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_feature_family_gnn_smoke.py"
SPEC = importlib.util.spec_from_file_location("gnn_event_prototype", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_event_prototype_head_supports_independent_multilabel_events():
    head = MODULE.EventPrototypeHead(hidden=8, n_events=3, metric_dim=5)
    logits = head(torch.randn(4, 8))

    assert logits.shape == (4, 3)
    assert torch.isfinite(logits).all()

    # Multiple events can be active for one token; this is not a softmax head.
    targets = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    loss = MODULE.event_loss_from_logits(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert head.prototypes.grad is not None


def test_target_prediction_layout_preserves_graph_and_event_columns():
    node_targets = torch.rand(2, 6)
    event_logits = torch.randn(2, len(MODULE.ALL_EVENT_TARGETS))
    combined = MODULE.combine_target_predictions(node_targets, event_logits)

    assert combined.shape == (2, 6 + len(MODULE.ALL_EVENT_TARGETS))
    assert torch.allclose(combined[:, :6], node_targets)
    assert torch.all((combined[:, 6:] >= 0) & (combined[:, 6:] <= 1))


def test_fused_and_moe_use_fixed_temporal_edges():
    model = MODULE.FusedFamilyGNN({"a": 4, "b": 5}, hidden=8)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.randn(2, 2)
    pair_src = torch.tensor([0])
    pair_dst = torch.tensor([2])
    outputs = model([torch.randn(3, 4), torch.randn(3, 5)], edge_index, edge_attr, pair_src, pair_dst)

    assert not hasattr(model, "edge_gate")
    assert outputs[0].shape == (3, 6)
    assert outputs[1].shape == (3, len(MODULE.ALL_EVENT_TARGETS))
    assert outputs[2].shape == (1, 2)

    moe = MODULE.FusedFamilyMoEGNN({"a": 4, "b": 5}, hidden=8)
    assert not hasattr(moe, "edge_gate")


def test_graph_event_labels_mark_only_top_decile():
    pd = __import__("pandas")
    dates = list(pd.date_range("2025-01-01", periods=21))
    labels = pd.DataFrame({
        "symbol": ["A"] * 21 + ["B"] * 21,
        "date": dates + dates,
        "long_hub": [0.0] * 20 + [100.0] + [100.0] * 20 + [0.0],
        "long_authority": [0.0] * 20 + [100.0] + [100.0] * 20 + [0.0],
        "long_pagerank": [0.0] * 20 + [100.0] + [100.0] * 20 + [0.0],
        "short_hub": [0.0] * 20 + [100.0] + [100.0] * 20 + [0.0],
        "short_authority": [0.0] * 20 + [100.0] + [100.0] * 20 + [0.0],
        "short_pagerank": [0.0] * 20 + [100.0] + [100.0] * 20 + [0.0],
    })
    result = MODULE.add_graph_event_labels(labels)
    assert result.loc[(result.symbol == "A") & (result.long_hub == 100), "is_long_hub_event"].iloc[0] == 1
    assert result.loc[(result.symbol == "B") & (result.long_hub == 0), "is_long_hub_event"].iloc[0] == 0
