import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_symbol_year_transformer_mtl.py"
SPEC = importlib.util.spec_from_file_location("symbol_year_transformer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_causal_mask_blocks_future_tokens_only():
    mask = MODULE.causal_mask(4)
    assert torch.isfinite(mask[0, 0])
    assert torch.isneginf(mask[0, 1])
    assert torch.isfinite(mask[3, 0])
    assert torch.isfinite(mask[3, 3])


def test_transformer_mtl_shapes():
    model = MODULE.TransformerMTL(6, {"sector_target": 3, "industry_target": 4, "year_target": 2})
    graph, speed, events, aux, macro = model(torch.randn(2, 5, 6), torch.zeros(2, 5, dtype=torch.bool))
    assert graph.shape == (2, 5, 6)
    assert speed.shape == (2, 5, len(MODULE.SPEED_TARGET_COLS))
    assert events.shape[0:2] == (2, 5)
    assert set(aux) == {"sector_target", "industry_target", "year_target"}
    assert macro is None


def test_two_tower_transformer_issuer_conditions_instruments():
    model = MODULE.TwoTowerTransformer(issuer_dim=6, instrument_dim=4, instrument_types=2)
    issuer = torch.randn(2, 5, 6)
    instruments = torch.randn(2, 2, 5, 4)
    issuer_padding = torch.zeros(2, 5, dtype=torch.bool)
    instrument_padding = torch.zeros(2, 2, 5, dtype=torch.bool)
    issuer_h, instrument_h = model(
        issuer, instruments, issuer_padding, instrument_padding,
        torch.tensor([[0, 1], [0, 1]]),
    )
    assert issuer_h.shape == (2, 5, MODULE.HIDDEN)
    assert instrument_h.shape == (2, 2, 5, MODULE.HIDDEN)
