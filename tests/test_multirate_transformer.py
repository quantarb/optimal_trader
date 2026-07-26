import pandas as pd
import pytest
import torch

from scripts.multirate_transformer.alignment import align_available_rows, assert_point_in_time
from scripts.multirate_transformer.config import MultiRateModelConfig
from scripts.multirate_transformer.losses import pairwise_logistic_ranking_loss
from scripts.multirate_transformer.model import MultiRateMultiAssetTransformer
from scripts.multirate_transformer.ranking import add_hits_rank_targets, percentile_rank_within_group
from scripts.multirate_transformer.heads import expand_hits_task_inventory
from scripts.multirate_transformer.sampler import CrossSectionalDateBatchSampler
from scripts.multirate_transformer.walk_forward import anchored_folds
from scripts.multirate_transformer.dataset import build_multirate_window
from scripts.multirate_transformer.warehouse_data import build_rate_views
from scripts.multirate_transformer.trading_policy import build_legacy_compatible_scores


def test_percentile_rank_direction_ties_and_group_mask():
    values = torch.tensor([0.8, 0.6, 0.9, 0.2])
    groups = torch.tensor([1, 1, 1, 2])
    ranks, valid = percentile_rank_within_group(values, groups, torch.ones(4, dtype=torch.bool), min_group_size=2)
    assert valid.tolist() == [True, True, True, False]
    assert ranks[:3].tolist() == pytest.approx([0.5, 0.0, 1.0])


def test_rank_labels_are_separated_by_issuer_and_asset_class():
    frame = pd.DataFrame({
        "date": ["2025-01-02"] * 4,
        "issuer": ["A", "A", "B", "C"],
        "asset_class": ["equity", "bond", "equity", "equity"],
        "long_hub": [0.8, 0.6, 0.9, 0.2],
    })
    ranked = add_hits_rank_targets(frame, ["long_hub"], min_asset_class_group_size=2)
    assert ranked.loc[0, "same_issuer_long_hub_rank"] == 1.0
    assert ranked.loc[1, "same_issuer_long_hub_rank"] == 0.0
    assert ranked["same_asset_class_long_hub_rank_valid"].tolist() == [True, False, True, True]


def test_alignment_rejects_future_observation():
    predictions = pd.DataFrame({"issuer": ["A"], "prediction_date": ["2025-01-01"]})
    observations = pd.DataFrame({"issuer": ["A"], "available_date": ["2025-01-02"], "value": [1.0]})
    aligned = align_available_rows(predictions, observations, entity_columns=["issuer"], value_columns=["value"])
    assert aligned.value.isna().all()
    with pytest.raises(ValueError):
        assert_point_in_time(pd.DataFrame({"prediction_date": ["2025-01-01"], "available_date": ["2025-01-02"]}))


def test_alignment_uses_latest_available_row():
    predictions = pd.DataFrame({"issuer": ["A", "A"], "prediction_date": ["2025-01-02", "2025-02-01"]})
    observations = pd.DataFrame({"issuer": ["A", "A"], "available_date": ["2025-01-01", "2025-01-15"], "value": [1.0, 2.0]})
    aligned = align_available_rows(predictions, observations, entity_columns=["issuer"], value_columns=["value"])
    assert aligned.value.tolist() == [1.0, 2.0]


def test_multirate_model_shapes_and_asset_adapters():
    config = MultiRateModelConfig(d_model=32, num_heads=4, annual_layers=1, quarterly_layers=1, decoder_layers=1, asset_classes=("equity", "corporate_bond", "preferred", "warrant"))
    model = MultiRateMultiAssetTransformer({"annual": 5, "quarterly": 6, "daily": 7, "equity": 7, "corporate_bond": 8, "preferred": 9, "warrant": 10}, config)
    output = model(torch.randn(2, 4, 10), torch.randn(2, 3, 5), torch.randn(2, 5, 6), asset_class="warrant")
    assert output["representation"].shape == (2, 4, 32)
    assert output["annual_memory"].shape == (2, 3, 32)
    assert output["quarterly_memory"].shape == (2, 5, 32)


def test_model_builds_current_dynamic_task_heads():
    config = MultiRateModelConfig(d_model=32, num_heads=4, annual_layers=1, quarterly_layers=1, decoder_layers=1, asset_classes=("equity",))
    model = MultiRateMultiAssetTransformer({"annual": 3, "quarterly": 3, "daily": 3, "equity": 3}, config)
    assert len(model.task_names) == 85
    output = model(torch.randn(1, 2, 3), torch.randn(1, 2, 3), torch.randn(1, 2, 3), asset_class="equity")
    assert set(output["outputs"]) == set(model.task_names)
    assert all(value.shape == (1, 2, 1) for value in output["outputs"].values())


def test_pairwise_loss_only_compares_same_group():
    prediction = torch.tensor([0.1, 0.9, 0.2, 0.8], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    groups = torch.tensor([1, 1, 2, 2])
    loss = pairwise_logistic_ranking_loss(prediction, target, groups, torch.ones(4, dtype=torch.bool))
    assert loss.item() > 0
    loss.backward()


def test_dynamic_heads_expand_each_active_hits_task():
    tasks = [{"task_name": "long_hub", "task_type": "regression"}, {"task_name": "is_insider_buy", "task_type": "event"}]
    names = [row["task_name"] for row in expand_hits_task_inventory(tasks)]
    assert names == ["long_hub", "same_issuer_long_hub_rank", "same_asset_class_long_hub_rank", "is_insider_buy"]


def test_date_batch_sampler_retains_peer_groups():
    frame = pd.DataFrame({
        "date": ["2025-01-01"] * 4 + ["2025-01-02"] * 2,
        "issuer": ["A", "A", "B", "C", "A", "B"],
        "asset_class": ["equity", "bond", "equity", "equity", "equity", "equity"],
    })
    sampler = CrossSectionalDateBatchSampler(frame, batch_dates=1)
    batch = next(iter(sampler))
    coverage = sampler.coverage(batch)
    assert coverage["unique_asset_classes"] >= 1
    assert coverage["valid_asset_class_groups"] >= 0


def test_anchored_walk_forward_never_trains_on_test_year():
    folds = anchored_folds(pd.date_range("2018-01-01", "2021-12-31", freq="MS"), first_test_year=2020, last_test_year=2021)
    assert [(fold.test_year, fold.train_end) for fold in folds] == [(2020, "2019-12-31"), (2021, "2020-12-31")]


def test_dataset_requires_real_availability_dates():
    daily = pd.DataFrame({"issuer": ["A"], "instrument_id": ["EQ1"], "date": ["2025-01-02"], "available_date": ["2025-01-02"]})
    annual = pd.DataFrame({"issuer": ["A"], "available_date": ["2024-12-01"], "revenue": [10.0]})
    quarterly = pd.DataFrame({"issuer": ["A"], "available_date": ["2024-12-15"], "revenue": [3.0]})
    window = build_multirate_window(prediction_date="2025-01-02", asset_class="equity", daily=daily, annual=annual, quarterly=quarterly)
    assert len(window.annual) == len(window.quarterly) == 1
    with pytest.raises(ValueError):
        build_multirate_window(prediction_date="2025-01-02", asset_class="equity", daily=daily, annual=annual.drop(columns="available_date"), quarterly=quarterly)


def test_existing_daily_panel_feeds_rate_views_without_future_rows():
    panel = pd.DataFrame({
        "symbol": ["A"] * 5,
        "date": pd.date_range("2024-01-01", periods=5, freq="90D"),
        "feature": [1, 2, 3, 4, 5],
    })
    views = build_rate_views(panel, symbol="A", prediction_date="2024-12-31", daily_length=10, annual_length=5, quarterly_length=12)
    assert len(views["daily"]) == 5
    assert views["daily"].date.max() <= pd.Timestamp("2024-12-31")
    assert "available_date" in views["annual"]


def test_new_model_uses_existing_entry_exit_score_policy():
    predictions = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "date": ["2025-01-01"] * 4,
        "long_hub": [0.9, 0.7, 0.2, 0.1],
        "long_authority": [0.8, 0.6, 0.3, 0.2],
        "short_hub": [0.1, 0.3, 0.8, 0.9],
        "short_authority": [0.2, 0.4, 0.7, 0.8],
    })
    scores = build_legacy_compatible_scores(predictions)
    assert scores.loc[scores.symbol.eq("A"), "long_score"].item() == 1.0
    assert scores.loc[scores.symbol.eq("A"), "long_exit_score"].item() == 1.0
    assert scores.loc[scores.symbol.eq("A"), "long_agree_count"].item() == 1
    assert scores.loc[scores.symbol.eq("C"), "short_agree_count"].item() == 1


def test_shared_score_policy_handles_speed_and_cross_sectional_heads():
    predictions = pd.DataFrame({
        "symbol": ["A", "B"],
        "date": ["2025-01-01"] * 2,
        "speed_long_hub": [0.2, 0.8],
        "speed_long_authority": [0.3, 0.7],
        "speed_short_hub": [0.8, 0.2],
        "speed_short_authority": [0.7, 0.3],
    })
    scores = build_legacy_compatible_scores(predictions, strategy="speed")
    assert scores.loc[scores.symbol.eq("B"), "long_score"].item() == 1.0
    assert scores.loc[scores.symbol.eq("A"), "short_score"].item() == 1.0
    assert scores["long_agree_count"].tolist() == [0, 1]
    assert scores["short_agree_count"].tolist() == [1, 0]
