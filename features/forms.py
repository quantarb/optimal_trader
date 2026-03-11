from __future__ import annotations

from typing import Optional, Sequence

from django import forms


class FeaturePreviewForm(forms.Form):
    universe_artifact_id = forms.ChoiceField(
        required=True,
        choices=(),
        label="Universe Artifact",
        help_text="Select the universe to load symbols from.",
    )
    job_name = forms.CharField(
        required=False,
        label="Job Name",
        help_text="Optional. Used when saving a feature engineering job.",
    )
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
        universe_artifact_choices: Optional[Sequence[tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if universe_artifact_choices:
            self.fields["universe_artifact_id"].choices = list(universe_artifact_choices)
        if not self.is_bound:
            if universe_artifact_choices:
                self.fields["universe_artifact_id"].initial = universe_artifact_choices[0][0]
