from __future__ import annotations

from typing import Optional, Sequence

from django import forms


class FeaturePreviewForm(forms.Form):
    symbol = forms.ChoiceField(required=True, choices=())
    include_price_technicals = forms.BooleanField(required=False, initial=True)
    include_fundamental_change = forms.BooleanField(required=False, initial=True)
    include_statement_quality = forms.BooleanField(required=False, initial=True)
    include_event_features = forms.BooleanField(required=False, initial=True)
    include_ownership_features = forms.BooleanField(required=False, initial=True)
    include_economic_indicators = forms.BooleanField(required=False, initial=True)
    include_treasury_rates = forms.BooleanField(required=False, initial=True)
    preview_rows = forms.IntegerField(required=False, min_value=10, max_value=1000, initial=100)

    def __init__(
        self,
        *args,
        symbol_choices: Optional[Sequence[tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if symbol_choices:
            self.fields["symbol"].choices = list(symbol_choices)
        if not self.is_bound:
            if symbol_choices:
                for candidate in ("AAPL", "MSFT", "NVDA"):
                    if any(candidate == c[0] for c in symbol_choices):
                        self.fields["symbol"].initial = candidate
                        break
