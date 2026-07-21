import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "macro_event_prototype.py"
SPEC = importlib.util.spec_from_file_location("macro_event_prototype", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_shared_macro_response_head_uses_one_prototype_space():
    head = MODULE.MacroEventResponsePrototypeHead(company_dim=8, event_dim=6, metric_dim=5)
    logits = head(torch.randn(7, 8), torch.randn(7, 6))

    assert logits.shape == (7, len(MODULE.MACRO_RESPONSE_CLASSES))
    assert torch.isfinite(logits).all()
    loss = MODULE.macro_response_loss(logits, torch.tensor([0, 1, 2, 3, 4, 2, 1]))
    loss.backward()
    assert head.prototypes.grad is not None
