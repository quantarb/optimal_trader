from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ml.base import FitSpec


class _FlairWordLimitTokenizer:
    """
    Tokenizer adapter that treats `max_length` as a word-token budget using
    Flair's default SegtokTokenizer, then applies HF tokenization.
    """

    def __init__(self, base_tokenizer):
        self.base = base_tokenizer
        try:
            from flair.tokenization import SegtokTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Flair tokenizer is required for word-level max_length semantics. "
                "Install with `pip install flair`."
            ) from e
        self._flair_tok = SegtokTokenizer()

    def _to_word_limited_text(self, text: Any, max_words: int) -> Any:
        if text is None:
            return text
        if isinstance(text, str):
            toks = self._flair_tok.tokenize(text)
            # Flair tokenizers can return either Token-like objects (with .text)
            # or plain strings depending on version/configuration.
            return " ".join(
                (t.text if hasattr(t, "text") else str(t))
                for t in toks[: int(max_words)]
            )
        if isinstance(text, list):
            return [self._to_word_limited_text(t, max_words) for t in text]
        return text

    def _model_subtoken_cap(self) -> int:
        model_max = getattr(self.base, "model_max_length", 512)
        try:
            model_max = int(model_max)
        except Exception:
            model_max = 512
        # HF uses very large sentinels for "infinite"; keep a safe cap.
        if model_max <= 0 or model_max > 100_000:
            return 512
        return model_max

    def __call__(self, *args, **kwargs):
        max_len = kwargs.get("max_length", None)
        truncation = kwargs.get("truncation", False)
        if truncation and max_len is not None:
            word_budget = int(max_len)
            if len(args) >= 1:
                a = list(args)
                a[0] = self._to_word_limited_text(a[0], word_budget)
                args = tuple(a)
            if "text" in kwargs:
                kwargs["text"] = self._to_word_limited_text(kwargs["text"], word_budget)
            if "text_target" in kwargs:
                kwargs["text_target"] = self._to_word_limited_text(kwargs["text_target"], word_budget)
            # After word truncation, keep model-level truncation bounded by model cap.
            kwargs["max_length"] = self._model_subtoken_cap()
        return self.base(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base, name)


@dataclass(frozen=True)
class Seq2SeqBuildConfig:
    numeric_precision: int = 2
    drop_missing_entry_rows: bool = True
    scientific_for_large_numbers: bool = True
    scientific_threshold: float = 1_000_000.0
    dedupe_source_duplicate_features: bool = True
    compact_feature_names: bool = False


_DEFAULT_TARGET_FIELDS: tuple[str, ...] = (
    "trade_return",
    "trade_duration_days",
)


def _format_num(v: float, precision: int = 2) -> str:
    x = float(v)
    p = int(precision)
    xr = round(x, p)
    if abs(xr) < 10 ** (-max(p, 1)):
        xr = 0.0
    s = f"{xr:.{p}f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def _format_num_text(
    v: float,
    *,
    precision: int,
    scientific_for_large_numbers: bool,
    scientific_threshold: float,
) -> str:
    x = float(v)
    if scientific_for_large_numbers and abs(x) >= float(scientific_threshold):
        # Keep scientific notation compact: 2 decimals by default -> 3 sig digits.
        return f"{x:.{int(precision)}e}"
    return _format_num(x, precision=precision)


def _is_missing_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"", "nan", "none", "null", "na", "n/a", "inf", "-inf", "infinity", "-infinity"}
    if pd.isna(v):
        return True
    if isinstance(v, (np.floating, float, np.integer, int)):
        return not np.isfinite(float(v))
    return False


def _compact_fin_key(key: str) -> str:
    """
    Text-only compact aliases for human-recognizable finance names.
    Keeps semantics while reducing token length.
    """
    k = str(key)

    # Exact aliases (common technical + macro + metadata-like fields)
    exact = {
        "Open": "O",
        "High": "H",
        "Low": "L",
        "Close": "C",
        "Volume": "Vol",
        "EntryPx": "EPx",
        "ExitPx": "XPx",
        "TradeDurationDays": "DurD",
        "TradeReturn": "Ret",
        "FederalFundsRate": "FFR",
        "Unemployment": "Unemp",
        "Inflation": "Infl",
        "MarketCap": "MktCap",
        "EnterpriseValue": "EV",
        "CurrentRatio": "CurrR",
        "QuickRatio": "QuickR",
        "CashRatio": "CashR",
        "DebtServiceCoverageRatio": "DSCR",
        "InterestCoverageRatio": "ICR",
        "DividendYield": "DivYld",
        "DividendYieldPercentage": "DivYldPct",
        "ReturnOnAssets": "ROA",
        "ReturnOnEquity": "ROE",
        "ReturnOnInvestedCapital": "ROIC",
        "ReturnOnCapitalEmployed": "ROCE",
        "OperatingReturnOnAssets": "OpROA",
        "EarningsYield": "EarningsYld",
        "FreeCashFlowYield": "FCFYld",
        "NetDebtToEBITDA": "NetDebt2EBITDA",
        "OperatingCashFlowRatio": "OCFR",
        "OperatingCashFlowSalesRatio": "OCF2Sales",
        "PriceToEarningsRatio": "PE",
        "PriceToBookRatio": "PB",
        "PriceToSalesRatio": "PS",
        "PriceToFreeCashFlowRatio": "PFCF",
        "PriceToOperatingCashFlowRatio": "POCF",
    }
    if k in exact:
        return exact[k]

    # Pattern aliases for technical features
    m = re.match(r"^Ret(\d+)d$", k)
    if m:
        return f"R{m.group(1)}D"
    m = re.match(r"^CumRet(\d+)d$", k)
    if m:
        return f"CR{m.group(1)}D"
    m = re.match(r"^DistSMA(\d+)$", k)
    if m:
        return f"SMADev{m.group(1)}"
    m = re.match(r"^SMASlope(\d+)$", k)
    if m:
        return f"SMASlp{m.group(1)}"
    m = re.match(r"^DistEMA(\d+)$", k)
    if m:
        return f"EMADev{m.group(1)}"
    m = re.match(r"^ZClose(\d+)$", k)
    if m:
        return f"ZC{m.group(1)}"
    m = re.match(r"^BBPos(\d+)$", k)
    if m:
        return f"BBP{m.group(1)}"
    m = re.match(r"^ATRPct(\d+)$", k)
    if m:
        return f"ATRP{m.group(1)}"
    m = re.match(r"^VolRegimeZ(\d+)$", k)
    if m:
        return f"VolRZ{m.group(1)}"
    m = re.match(r"^BreakoutUp(\d+)$", k)
    if m:
        return f"BrkUp{m.group(1)}"
    m = re.match(r"^BreakoutDn(\d+)$", k)
    if m:
        return f"BrkDn{m.group(1)}"
    m = re.match(r"^PosInChannel(\d+)$", k)
    if m:
        return f"ChPos{m.group(1)}"
    m = re.match(r"^DistHh(\d+)$", k)
    if m:
        return f"DHH{m.group(1)}"
    m = re.match(r"^DistLl(\d+)$", k)
    if m:
        return f"DLL{m.group(1)}"
    m = re.match(r"^VolZ(\d+)$", k)
    if m:
        return f"VZ{m.group(1)}"
    m = re.match(r"^USTMonth(\d+)$", k)
    if m:
        return f"UST{m.group(1)}M"
    m = re.match(r"^USTYear(\d+)$", k)
    if m:
        return f"UST{m.group(1)}Y"

    # Generic compressions for long fundamentals
    repl = [
        ("OperatingCashFlow", "OCF"),
        ("FreeCashFlow", "FCF"),
        ("CashFlow", "CF"),
        ("EnterpriseValue", "EV"),
        ("MarketCap", "MktCap"),
        ("ReturnOn", "RO"),
        ("PriceTo", "P2"),
        ("DebtTo", "D2"),
        ("LongTerm", "LT"),
        ("ShortTerm", "ST"),
        ("WorkingCapital", "WC"),
        ("InvestedCapital", "IC"),
        ("CapitalExpenditure", "CapEx"),
        ("Dividend", "Div"),
        ("PerShare", "PS"),
        ("Coverage", "Cov"),
        ("Turnover", "TO"),
        ("Outstanding", "Out"),
        ("Inventory", "Inv"),
        ("Receivables", "AR"),
        ("Payables", "AP"),
        ("Revenue", "Rev"),
        ("Income", "Inc"),
        ("Assets", "Ast"),
        ("Equity", "Eq"),
        ("Yield", "Yld"),
        ("Margin", "Mgn"),
        ("EffectiveTaxRate", "ETR"),
    ]
    out = k
    for a, b in repl:
        out = out.replace(a, b)
    return out


def _ensure_feature_panel(final_df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(final_df.index, pd.MultiIndex) and {"date", "symbol"}.issubset(set(final_df.index.names)):
        out = final_df.copy()
    elif {"date", "symbol"}.issubset(set(final_df.columns)):
        out = final_df.set_index(["date", "symbol"])
    else:
        raise ValueError("final_df must have MultiIndex (date, symbol) or columns ['date', 'symbol'].")

    # Normalize index dtypes for stable key lookup.
    out = out.copy()
    date_vals = pd.to_datetime(out.index.get_level_values("date"), errors="coerce")
    sym_vals = out.index.get_level_values("symbol").astype(str)
    out.index = pd.MultiIndex.from_arrays([date_vals, sym_vals], names=["date", "symbol"])
    out = out.sort_index()
    return out


def _ensure_trades_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    out = trades_df.copy()
    # Prefer index-derived keys when present.
    if "symbol" not in out.columns:
        if isinstance(out.index, pd.MultiIndex) and "symbol" in (out.index.names or []):
            out["symbol"] = out.index.get_level_values("symbol")
        elif out.index.name == "symbol":
            out = out.reset_index()
        else:
            raise ValueError("trades_df must provide 'symbol' as a column or index level.")

    # If entry_date is absent but index carries a date, use it as entry_date.
    if "entry_date" not in out.columns:
        if isinstance(out.index, pd.MultiIndex) and "date" in (out.index.names or []):
            out["entry_date"] = out.index.get_level_values("date")
        elif out.index.name == "date":
            out = out.reset_index().rename(columns={"date": "entry_date"})
        elif "date" in out.columns:
            out["entry_date"] = out["date"]
        else:
            raise ValueError("trades_df must provide 'entry_date' (or a date index level/column).")

    if "exit_date" not in out.columns:
        raise ValueError("trades_df must include 'exit_date'.")

    out["symbol"] = out["symbol"].astype(str)
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce")
    out["exit_date"] = pd.to_datetime(out["exit_date"], errors="coerce")
    out = out.dropna(subset=["entry_date", "exit_date", "symbol"])
    out = out[out["exit_date"] > out["entry_date"]]
    out = out.sort_values(["symbol", "entry_date"]).reset_index(drop=True)
    return out


def _resolve_feature_cols(
    panel: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
) -> list[str]:
    if feature_cols is not None:
        cols = [c for c in list(feature_cols) if c in panel.columns]
        if not cols:
            raise ValueError("None of feature_cols are present in final_df.")
        return cols

    # Default: numeric feature columns only.
    return panel.select_dtypes(include=[np.number]).columns.tolist()


def _dedupe_source_duplicate_feature_cols(cols: Sequence[str]) -> list[str]:
    """
    Text-only dedupe for source-collision columns created upstream.
    Example: keep 'CurrentRatio', drop 'CurrentRatioRt'/'CurrentRatioKm' if base exists.
    """
    out: list[str] = []
    all_cols = {str(c) for c in cols}
    for c in cols:
        name = str(c)
        if name.endswith("Rt") and name[:-2] in all_cols:
            continue
        if name.endswith("Km") and name[:-2] in all_cols:
            continue
        out.append(name)
    return out


def _build_input_text(
    row: pd.Series,
    *,
    symbol: str,
    entry_date: pd.Timestamp,
    feature_cols: Sequence[str],
    cfg: Seq2SeqBuildConfig,
) -> str:
    parts = [
        f"Symbol={symbol}",
        f"EntryDate={entry_date.strftime('%Y-%m-%d')}",
    ]

    for c in list(feature_cols):
        v = row.get(c, np.nan)
        if _is_missing_value(v):
            continue
        key = _compact_fin_key(str(c)) if cfg.compact_feature_names else str(c)
        if isinstance(v, (np.floating, float, np.integer, int)):
            parts.append(
                f"{key}={_format_num_text(float(v), precision=cfg.numeric_precision, scientific_for_large_numbers=cfg.scientific_for_large_numbers, scientific_threshold=cfg.scientific_threshold)}"
            )
        else:
            parts.append(f"{key}={v}")
    return " ".join(parts)


def _build_target_text(
    *,
    tr: pd.Series,
    exit_row: pd.Series,
    symbol: str,
    exit_date: pd.Timestamp,
    feature_cols: Sequence[str],
    cfg: Seq2SeqBuildConfig,
    target_fields: Sequence[str],
) -> str:
    # Exit world state: symbol/date + exit-day features.
    parts = [
        f"Symbol={symbol}",
        f"ExitDate={exit_date.strftime('%Y-%m-%d')}",
    ]

    for c in list(feature_cols):
        v = exit_row.get(c, np.nan)
        if _is_missing_value(v):
            continue
        key = _compact_fin_key(str(c)) if cfg.compact_feature_names else str(c)
        if isinstance(v, (np.floating, float, np.integer, int)):
            parts.append(
                f"{key}={_format_num_text(float(v), precision=cfg.numeric_precision, scientific_for_large_numbers=cfg.scientific_for_large_numbers, scientific_threshold=cfg.scientific_threshold)}"
            )
        else:
            parts.append(f"{key}={v}")

    # Trade outcomes appended to target.
    for c in target_fields:
        if c not in tr.index:
            continue
        v = tr[c]
        if _is_missing_value(v):
            continue
        # Trade fields come from snake_case labels/trades; normalize these keys for text.
        raw = str(c).strip().replace("__", "_")
        key = "".join(tok[:1].upper() + tok[1:] for tok in raw.split("_") if tok)
        if c in {"trade_duration_days"}:
            parts.append(f"{key}={int(v)}")
        elif isinstance(v, (np.floating, float, np.integer, int)):
            parts.append(
                f"{key}={_format_num_text(float(v), precision=cfg.numeric_precision, scientific_for_large_numbers=cfg.scientific_for_large_numbers, scientific_threshold=cfg.scientific_threshold)}"
            )
        else:
            parts.append(f"{key}={v}")
    return " ".join(parts)


def prepare_entry2exit_dataset(
    *,
    final_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    numeric_precision: int = 2,
    scientific_for_large_numbers: bool = True,
    scientific_threshold: float = 1_000_000.0,
    dedupe_source_duplicate_features: bool = True,
    compact_feature_names: bool = False,
    target_fields: Sequence[str] = _DEFAULT_TARGET_FIELDS,
    drop_missing_entry_rows: bool = True,
) -> pd.DataFrame:
    """
    Build entry->exit seq2seq rows from features + optimal trades.

    Returns a dataframe with original trade columns plus:
      - input_text  (entry world state)
      - target_text (exit world state + trade outcome fields)
    """
    cfg = Seq2SeqBuildConfig(
        numeric_precision=int(numeric_precision),
        drop_missing_entry_rows=bool(drop_missing_entry_rows),
        scientific_for_large_numbers=bool(scientific_for_large_numbers),
        scientific_threshold=float(scientific_threshold),
        dedupe_source_duplicate_features=bool(dedupe_source_duplicate_features),
        compact_feature_names=bool(compact_feature_names),
    )

    panel = _ensure_feature_panel(final_df)
    trades = _ensure_trades_df(trades_df)
    feat_cols = _resolve_feature_cols(panel, feature_cols)
    if cfg.dedupe_source_duplicate_features:
        feat_cols = _dedupe_source_duplicate_feature_cols(feat_cols)

    rows: list[dict[str, Any]] = []
    missing_entry = 0
    missing_exit = 0
    for _, tr in trades.iterrows():
        symbol = str(tr["symbol"])
        entry_date = pd.Timestamp(tr["entry_date"]).normalize()
        exit_date = pd.Timestamp(tr["exit_date"]).normalize()
        entry_key = (entry_date, symbol)
        exit_key = (exit_date, symbol)

        if entry_key not in panel.index:
            missing_entry += 1
            if cfg.drop_missing_entry_rows:
                continue
            entry_row = pd.Series(index=feat_cols, dtype=float)
        else:
            entry_row = panel.loc[entry_key]
            if isinstance(entry_row, pd.DataFrame):
                entry_row = entry_row.iloc[0]

        if exit_key not in panel.index:
            missing_exit += 1
            exit_row = pd.Series(index=feat_cols, dtype=float)
        else:
            exit_row = panel.loc[exit_key]
            if isinstance(exit_row, pd.DataFrame):
                exit_row = exit_row.iloc[0]

        input_text = _build_input_text(
            entry_row,
            symbol=symbol,
            entry_date=entry_date,
            feature_cols=feat_cols,
            cfg=cfg,
        )
        target_text = _build_target_text(
            tr=tr,
            exit_row=exit_row,
            symbol=symbol,
            exit_date=exit_date,
            feature_cols=feat_cols,
            cfg=cfg,
            target_fields=target_fields,
        )

        rec = tr.to_dict()
        rec["input_text"] = input_text
        rec["target_text"] = target_text
        rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=list(trades.columns) + ["input_text", "target_text"])

    out = out.reset_index(drop=True)
    out.attrs["missing_entry_rows"] = int(missing_entry)
    out.attrs["missing_exit_rows"] = int(missing_exit)
    out.attrs["n_feature_cols_used"] = int(len(feat_cols))
    return out


def train_seq2seq_model(
    *,
    seq2seq_df: pd.DataFrame,
    model_name: str = "google/flan-t5-small",
    max_length: int = 512,
    num_train_epochs: float = 1.0,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    spec: Optional[FitSpec] = None,
) -> dict[str, Any]:
    """
    Minimal seq2seq trainer (HF Transformers).
    """
    if seq2seq_df.empty:
        raise ValueError("seq2seq_df is empty.")

    seq_cfg = spec.sequence if spec is not None else None
    input_col = seq_cfg.input_col if seq_cfg is not None else "input_text"
    target_col = seq_cfg.target_col if seq_cfg is not None else "target_text"
    source_len = int(seq_cfg.max_source_length) if seq_cfg is not None else int(max_length)
    target_len = int(seq_cfg.max_target_length) if seq_cfg is not None else int(max_length)
    padding = seq_cfg.padding if seq_cfg is not None else "max_length"

    if not {input_col, target_col}.issubset(set(seq2seq_df.columns)):
        raise ValueError(f"seq2seq_df must include '{input_col}' and '{target_col}' columns.")

    from datasets import Dataset
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    ds = Dataset.from_pandas(seq2seq_df[[input_col, target_col]].reset_index(drop=True))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = _FlairWordLimitTokenizer(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Training with KV cache increases memory; disable it for seq2seq finetuning.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    def _tokenize(batch):
        model_inputs = tokenizer(
            batch[input_col],
            max_length=source_len,
            truncation=True,
            padding=padding,
        )
        labels = tokenizer(
            text_target=batch[target_col],
            max_length=target_len,
            truncation=True,
            padding=padding,
        )
        # Ignore padded label tokens in the loss; otherwise the model overfits
        # to predicting padding and can degenerate into repetitive outputs.
        pad_id = getattr(tokenizer, "pad_token_id", None)
        label_ids = labels["input_ids"]
        if pad_id is not None:
            label_ids = [
                [(tok if tok != pad_id else -100) for tok in seq]
                for seq in label_ids
            ]
        model_inputs["labels"] = label_ids
        return model_inputs

    ds_tok = ds.map(_tokenize, batched=True, remove_columns=list(ds.column_names))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Apple MPS can OOM with moderate per-device batches. Use micro-batches and
    # recover effective batch size via gradient accumulation.
    is_mps = bool(torch.backends.mps.is_available())
    req_batch = max(1, int(batch_size))
    if is_mps:
        per_device_batch = min(req_batch, 2)
    else:
        per_device_batch = req_batch
    grad_accum = int(math.ceil(req_batch / per_device_batch))

    if is_mps and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    args = Seq2SeqTrainingArguments(
        output_dir="/tmp/seq2seq_artifacts",
        per_device_train_batch_size=int(per_device_batch),
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(learning_rate),
        warmup_ratio=float(warmup_ratio),
        lr_scheduler_type=str(lr_scheduler_type),
        num_train_epochs=float(num_train_epochs),
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        optim="adafactor",
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Ensure downstream callers receive a fully materialized model on a stable
    # device for immediate inference (notebook inputs default to CPU tensors).
    model = trainer.model
    try:
        model = model.to("cpu")
        model.eval()
    except Exception:
        # If device transfer is unavailable for any reason, keep trained model.
        pass

    return {"trainer": trainer, "model": model, "tokenizer": tokenizer, "train_dataset": ds_tok}
