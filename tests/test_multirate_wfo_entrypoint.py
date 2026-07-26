import importlib.util
import os
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_multirate_transformer_wfo.py"
SPEC = importlib.util.spec_from_file_location("multirate_wfo_entrypoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_multirate_entrypoint_enables_anchored_architecture(monkeypatch):
    monkeypatch.delenv("TRANSFORMER_ISSUER_ENCODER_INSTRUMENT_DECODER", raising=False)
    monkeypatch.delenv("TRANSFORMER_MULTIRATE_ISSUER_STATE", raising=False)
    monkeypatch.delenv("TRANSFORMER_SINGLE_FIT", raising=False)

    MODULE._enable_multirate_defaults()

    assert os.environ["TRANSFORMER_ISSUER_ENCODER_INSTRUMENT_DECODER"] == "1"
    assert os.environ["TRANSFORMER_MULTIRATE_ISSUER_STATE"] == "1"
    assert os.environ["TRANSFORMER_SINGLE_FIT"] == "0"


def test_multirate_entrypoint_preserves_explicit_configuration(monkeypatch):
    monkeypatch.setenv("TRANSFORMER_ISSUER_ENCODER_INSTRUMENT_DECODER", "0")
    monkeypatch.setenv("TRANSFORMER_MULTIRATE_ISSUER_STATE", "0")
    monkeypatch.setenv("TRANSFORMER_SINGLE_FIT", "1")

    MODULE._enable_multirate_defaults()

    assert os.environ["TRANSFORMER_ISSUER_ENCODER_INSTRUMENT_DECODER"] == "0"
    assert os.environ["TRANSFORMER_MULTIRATE_ISSUER_STATE"] == "0"
    assert os.environ["TRANSFORMER_SINGLE_FIT"] == "1"
