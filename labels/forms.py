from __future__ import annotations

from django import forms
from typing import Optional, Sequence


PRICE_COL_CHOICES = (
    ("adj_open", "Adjust Open"),
    ("adj_high", "Adjust High"),
    ("adj_low", "Adjust Low"),
    ("adj_close", "Adjust Close"),
)

TRADE_DEDUP_CHOICES = (
    ("exact", "Exact Trade (symbol/side/entry/exit)"),
    ("entry_date", "Same Entry Date (keep highest return)"),
    ("none", "No Trade Deduplication"),
)

MODEL_TARGET_CHOICES = (
    ("label", "Label"),
    ("market_position", "Market Position"),
    ("trade_return", "Trade Return"),
)


class LabelingConfigForm(forms.Form):
    k_w_list = forms.CharField(required=False, initial="", label="k list for W", help_text="Comma-separated list, e.g. 2,3,4")
    k_m_list = forms.CharField(required=False, initial="", label="k list for M", help_text="Comma-separated list, e.g. 1,2")
    k_qe_list = forms.CharField(required=False, initial="", label="k list for QE", help_text="Comma-separated list, e.g. 1,2")
    k_ye_list = forms.CharField(required=False, initial="1, 2, 4, 8", label="k list for YE", help_text="Comma-separated list, e.g. 1")
    min_profit_pct = forms.FloatField(
        required=False,
        min_value=0.0,
        initial=1.0,
        label="Min Profit %",
        help_text="Percent points. Example: 10 means 10%.",
    )

    buy_execution = forms.ChoiceField(required=True, choices=PRICE_COL_CHOICES, initial="adj_high", label="Buy Price Execution")
    sell_execution = forms.ChoiceField(required=True, choices=PRICE_COL_CHOICES, initial="adj_low", label="Sell Price Execution")
    short_execution = forms.ChoiceField(required=True, choices=PRICE_COL_CHOICES, initial="adj_low", label="Short Price Execution")
    cover_execution = forms.ChoiceField(required=True, choices=PRICE_COL_CHOICES, initial="adj_high", label="Cover Price Execution")
    fee_bps = forms.FloatField(required=False, min_value=0.0, initial=10.0)
    slippage_bps = forms.FloatField(required=False, min_value=0.0, initial=10.0)
    trade_dedup_mode = forms.ChoiceField(
        required=True,
        choices=TRADE_DEDUP_CHOICES,
        initial="exact",
        label="Trade Deduplication",
        help_text="Choose how to deduplicate overlapping generated trades.",
    )
    model_target_col = forms.ChoiceField(
        required=True,
        choices=MODEL_TARGET_CHOICES,
        initial="label",
        label="Model Target Label",
        help_text="Target column to carry into the model forms.",
    )

    def clean(self):
        cleaned = super().clean()
        raw_lists = [
            str(cleaned.get("k_w_list") or "").strip(),
            str(cleaned.get("k_m_list") or "").strip(),
            str(cleaned.get("k_qe_list") or "").strip(),
            str(cleaned.get("k_ye_list") or "").strip(),
        ]
        if not any(raw_lists):
            raise forms.ValidationError("Enter at least one k value (W, M, QE, or YE) before building config.")
        return cleaned

    def __init__(
        self,
        *args,
        symbol_choices: Optional[Sequence[tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not self.is_bound:
            self.fields["buy_execution"].initial = "adj_high"
            self.fields["sell_execution"].initial = "adj_low"
            self.fields["short_execution"].initial = "adj_low"
            self.fields["cover_execution"].initial = "adj_high"
