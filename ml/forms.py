from __future__ import annotations

import json
from typing import Sequence

from django import forms

from .feature_families import FEATURE_FAMILY_CHOICES


FRAMEWORK_CHOICES = (
    ("sklearn", "scikit-learn"),
    ("torch", "PyTorch"),
    ("autogluon", "AutoGluon"),
)

TASK_TYPE_CHOICES = (
    ("classification", "Classification"),
    ("regression", "Regression"),
    ("embedding", "Embedding"),
    ("seq2seq", "Sequence"),
)

ALGORITHM_CHOICES = (
    ("random_forest_classifier", "Random Forest Classifier"),
    ("random_forest_regressor", "Random Forest Regressor"),
    ("autoencoder", "Autoencoder"),
    ("ppo", "PPO"),
    ("a2c", "A2C"),
)


class ModelTrainingForm(forms.Form):
    symbol = forms.ChoiceField(
        required=True,
        choices=(),
        help_text="The symbol used to build the training dataset for this job.",
    )
    name = forms.CharField(
        max_length=255,
        help_text="Human-readable name for this training run.",
    )
    framework = forms.ChoiceField(choices=FRAMEWORK_CHOICES, initial="sklearn")
    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES, initial="random_forest_classifier")
    task_type = forms.ChoiceField(choices=TASK_TYPE_CHOICES, initial="classification")
    target_col = forms.CharField(
        max_length=128,
        initial="label",
        help_text="The target column in your training dataset.",
    )
    feature_families = forms.MultipleChoiceField(
        choices=FEATURE_FAMILY_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        initial=[
            "prices_div_adj",
            "key_metrics",
            "ratios",
            "income_statement",
            "income_statement_growth",
        ],
        help_text="Choose which feature families should be built for training.",
    )
    split_ratio = forms.FloatField(
        min_value=0.1,
        max_value=0.99,
        initial=0.8,
        help_text="Fraction of rows used for training.",
    )
    params_json = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 6}),
        initial='{"n_estimators": 200, "max_depth": 12}',
        label="Hyperparameters (JSON)",
        help_text="Optional JSON object of model hyperparameters.",
    )
    notes = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 4}),
        help_text="Optional notes about data source, filtering, or intent.",
    )

    def __init__(self, *args, symbol_choices: Sequence[tuple[str, str]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if symbol_choices:
            self.fields["symbol"].choices = list(symbol_choices)
            if not self.is_bound:
                for candidate in ("AAPL", "MSFT", "NVDA"):
                    if any(candidate == code for code, _label in symbol_choices):
                        self.fields["symbol"].initial = candidate
                        break

    def clean_feature_families(self) -> list[str]:
        feature_families = list(self.cleaned_data["feature_families"])
        if not feature_families:
            raise forms.ValidationError("Select at least one feature family.")
        return feature_families

    def clean_params_json(self) -> dict:
        raw_value = (self.cleaned_data.get("params_json") or "").strip()
        if not raw_value:
            return {}
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise forms.ValidationError("Enter a valid JSON object.") from exc
        if not isinstance(parsed, dict):
            raise forms.ValidationError("Hyperparameters must be a JSON object.")
        return parsed
