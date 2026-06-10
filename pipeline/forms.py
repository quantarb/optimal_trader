from __future__ import annotations

import json

from django import forms

from .models import StrategyDefinition

MODEL_ALGORITHM_CHOICES = (
    ("random_forest_classifier", "Random Forest Classifier"),
    ("moe_random_forest_classifier", "Feature Family MoE Random Forest"),
    ("sector_moe_random_forest_classifier", "Sector MoE Random Forest"),
    ("industry_moe_random_forest_classifier", "Industry MoE Random Forest"),
    ("random_forest_regressor", "Random Forest Regressor"),
)

CLASSIFIER_ALGORITHM_CHOICES = tuple(
    choice for choice in MODEL_ALGORITHM_CHOICES if choice[0] != "random_forest_regressor"
)


class FitModelPipelineForm(forms.Form):
    RESEARCH_SCOPE_CHOICES = (
        ("single_regime", "Single Regime / Small Universe"),
        ("broad_universe", "Broad Universe"),
        ("long_history", "Long History"),
        ("broad_universe_long_history", "Broad Universe + Long History"),
    )
    SAMPLE_WEIGHT_CHOICES = (
        ("trade_return_abs", "Weight Bigger Oracle Trades More"),
        ("uniform", "Uniform"),
    )

    name = forms.CharField(max_length=255)
    job_type = forms.ChoiceField(
        choices=(
            ("fit_classifier", "Fit Classifier"),
            ("fit_regressor", "Fit Regressor"),
            ("fit_mtl", "Fit Multi-Task"),
        )
    )
    algorithm = forms.ChoiceField(
        choices=CLASSIFIER_ALGORITHM_CHOICES,
        required=False,
        initial="random_forest_classifier",
        help_text="Used by Fit Classifier. Sector and industry MoE routes use FMP profile classifications.",
    )
    feature_artifact_id = forms.TypedChoiceField(coerce=int, choices=(), empty_value=0, label="Feature Artifact")
    label_artifact_id = forms.TypedChoiceField(coerce=int, choices=(), empty_value=0, label="Label Artifact")
    prediction_artifact_ids = forms.MultipleChoiceField(
        required=False,
        choices=(),
        widget=forms.CheckboxSelectMultiple,
        label="Extra State Panels",
    )
    target_col = forms.CharField(max_length=128, required=False)
    split_ratio = forms.FloatField(min_value=0.1, max_value=0.99, initial=0.8)
    research_scope = forms.ChoiceField(choices=RESEARCH_SCOPE_CHOICES, required=False, initial="broad_universe_long_history")
    min_abs_trade_return_pct = forms.FloatField(min_value=0.0, required=False, initial=8.0, label="Min Abs Trade Return %")
    max_hold_days = forms.IntegerField(min_value=1, required=False, initial=90)
    sample_weight_mode = forms.ChoiceField(choices=SAMPLE_WEIGHT_CHOICES, required=False, initial="trade_return_abs")
    params_json = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows": 4}), initial="{}")

    def __init__(self, *args, feature_choices=None, label_choices=None, prediction_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["feature_artifact_id"].choices = [(0, "Select a FEATURES artifact")] + list(feature_choices or [])
        self.fields["label_artifact_id"].choices = [(0, "Select a LABELS artifact")] + list(label_choices or [])
        self.fields["prediction_artifact_ids"].choices = list(prediction_choices or [])

    def clean_feature_artifact_id(self) -> int:
        value = int(self.cleaned_data.get("feature_artifact_id") or 0)
        if value <= 0:
            raise forms.ValidationError("Select a feature artifact.")
        return value

    def clean_label_artifact_id(self) -> int:
        value = int(self.cleaned_data.get("label_artifact_id") or 0)
        if value <= 0:
            raise forms.ValidationError("Select a label artifact.")
        return value

    def clean_prediction_artifact_ids(self) -> list[int]:
        out: list[int] = []
        for value in list(self.cleaned_data.get("prediction_artifact_ids") or []):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed > 0:
                out.append(parsed)
        return out

    def clean_params_json(self) -> dict:
        raw = str(self.cleaned_data.get("params_json") or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise forms.ValidationError("Enter a valid JSON object.") from exc
        if not isinstance(payload, dict):
            raise forms.ValidationError("Hyperparameters must be a JSON object.")
        return payload


class ScoreModelPipelineForm(forms.Form):
    name = forms.CharField(max_length=255)
    job_type = forms.ChoiceField(
        choices=(
            ("score_classifier", "Score Classifier"),
            ("score_regressor", "Score Regressor"),
            ("score_mtl", "Score Multi-Task"),
        )
    )
    model_artifact_id = forms.TypedChoiceField(coerce=int, choices=(), empty_value=0, label="Model Artifact")
    feature_artifact_id = forms.TypedChoiceField(coerce=int, choices=(), empty_value=0, label="Feature Artifact")
    label_artifact_id = forms.TypedChoiceField(coerce=int, required=False, choices=(), empty_value=0, label="Optional Label Artifact")
    prediction_artifact_ids = forms.MultipleChoiceField(
        required=False,
        choices=(),
        widget=forms.CheckboxSelectMultiple,
        label="Extra State Panels",
    )

    def __init__(self, *args, model_choices=None, feature_choices=None, label_choices=None, prediction_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model_artifact_id"].choices = [(0, "Select a model artifact")] + list(model_choices or [])
        self.fields["feature_artifact_id"].choices = [(0, "Select a FEATURES artifact")] + list(feature_choices or [])
        self.fields["label_artifact_id"].choices = [(0, "No label artifact")] + list(label_choices or [])
        self.fields["prediction_artifact_ids"].choices = list(prediction_choices or [])

    def clean_prediction_artifact_ids(self) -> list[int]:
        out: list[int] = []
        for value in list(self.cleaned_data.get("prediction_artifact_ids") or []):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed > 0:
                out.append(parsed)
        return out


class StrategyDatasetPipelineForm(forms.Form):
    name = forms.CharField(max_length=255)
    strategy_definition_id = forms.TypedChoiceField(coerce=int, choices=(), empty_value=0, label="Strategy Definition")
    feature_artifact_id = forms.TypedChoiceField(coerce=int, choices=(), empty_value=0, label="Feature Artifact")
    label_artifact_id = forms.TypedChoiceField(coerce=int, required=False, choices=(), empty_value=0, label="Optional Label Artifact")
    prediction_artifact_ids = forms.MultipleChoiceField(
        required=False,
        choices=(),
        widget=forms.CheckboxSelectMultiple,
        label="State Panels",
    )

    def __init__(self, *args, strategy_definition_choices=None, feature_choices=None, label_choices=None, prediction_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["strategy_definition_id"].choices = [(0, "Select a strategy definition")] + list(strategy_definition_choices or [])
        self.fields["feature_artifact_id"].choices = [(0, "Select a FEATURES artifact")] + list(feature_choices or [])
        self.fields["label_artifact_id"].choices = [(0, "No label artifact")] + list(label_choices or [])
        self.fields["prediction_artifact_ids"].choices = list(prediction_choices or [])

    def clean_strategy_definition_id(self) -> int:
        value = int(self.cleaned_data.get("strategy_definition_id") or 0)
        if value <= 0:
            raise forms.ValidationError("Select a strategy definition.")
        return value

    def clean_prediction_artifact_ids(self) -> list[int]:
        out: list[int] = []
        for value in list(self.cleaned_data.get("prediction_artifact_ids") or []):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed > 0:
                out.append(parsed)
        return out


class BacktestPipelineForm(forms.Form):
    name = forms.CharField(max_length=255)
    strategy_dataset_artifact_id = forms.TypedChoiceField(coerce=int, choices=(), empty_value=0, label="Strategy Dataset Artifact")
    transaction_cost_bps = forms.FloatField(min_value=0.0, initial=0.0)

    def __init__(self, *args, strategy_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["strategy_dataset_artifact_id"].choices = [(0, "Select a strategy dataset artifact")] + list(strategy_choices or [])


class OptimalTradeResearchForm(forms.Form):
    UNIVERSE_MODE_CHOICES = (
        ("mag7", "MAG7"),
        ("us_market_cap_screen", "US Market Cap Screen"),
    )
    MARKET_CAP_TIER_CHOICES = (
        ("", "Custom / None"),
        ("1t", ">= $1T"),
        ("100b", ">= $100B"),
        ("10b", ">= $10B"),
    )

    name = forms.CharField(max_length=255, initial="optimal_trade_research")
    universe_mode = forms.ChoiceField(choices=UNIVERSE_MODE_CHOICES, initial="mag7", label="Universe Scope")
    market_cap_tier = forms.ChoiceField(choices=MARKET_CAP_TIER_CHOICES, required=False, initial="", label="Market Cap Tier")
    min_market_cap = forms.FloatField(required=False, min_value=0.0, initial=None, label="Min Market Cap ($)")
    country = forms.CharField(required=False, max_length=16, initial="US")
    exchanges_csv = forms.CharField(required=False, initial="NASDAQ,NYSE,AMEX", label="Exchanges")
    max_symbols = forms.IntegerField(required=False, min_value=1, initial=None, label="Max Symbols")
    feature_artifact_id = forms.TypedChoiceField(coerce=int, required=False, choices=(), empty_value=0, label="Optional Feature Artifact")
    label_artifact_id = forms.TypedChoiceField(coerce=int, required=False, choices=(), empty_value=0, label="Optional Label Artifact")
    profile_name = forms.ChoiceField(choices=(), initial="broad_universe_long_history", label="Research Profile")
    test_start_year = forms.IntegerField(min_value=2000, initial=2023)
    test_end_year = forms.IntegerField(min_value=2000, initial=2025)
    min_profit_pct = forms.FloatField(min_value=0.0, initial=12.0, label="Oracle Min Profit %")
    transaction_cost_bps = forms.FloatField(min_value=0.0, initial=10.0)
    resume_existing = forms.BooleanField(required=False, initial=True, label="Reuse Completed Folds And Suites")

    def __init__(self, *args, feature_choices=None, label_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        from .research_suite import research_profile_names

        self.fields["feature_artifact_id"].choices = [(0, "Build features inside suite")] + list(feature_choices or [])
        self.fields["label_artifact_id"].choices = [(0, "Build labels inside suite")] + list(label_choices or [])
        self.fields["profile_name"].choices = [(name, name) for name in research_profile_names()]

    def clean(self):
        cleaned = super().clean()
        start = int(cleaned.get("test_start_year") or 0)
        end = int(cleaned.get("test_end_year") or 0)
        if start and end and end < start:
            raise forms.ValidationError("Test end year must be greater than or equal to test start year.")
        universe_mode = str(cleaned.get("universe_mode") or "mag7").strip()
        market_cap_tier = str(cleaned.get("market_cap_tier") or "").strip().lower()
        if universe_mode == "us_market_cap_screen":
            if market_cap_tier == "1t":
                cleaned["min_market_cap"] = 1_000_000_000_000.0
            elif market_cap_tier == "100b":
                cleaned["min_market_cap"] = 100_000_000_000.0
            elif market_cap_tier == "10b":
                cleaned["min_market_cap"] = 10_000_000_000.0
            if not str(cleaned.get("country") or "").strip():
                raise forms.ValidationError("Country is required for screened universes.")
            if not str(cleaned.get("exchanges_csv") or "").strip():
                raise forms.ValidationError("At least one exchange is required for screened universes.")
        return cleaned


class StrategyDefinitionForm(forms.ModelForm):
    advanced_config_json = forms.CharField(widget=forms.Textarea(attrs={"rows": 8}), label="Advanced JSON Overrides", required=False, initial="{}")

    class Meta:
        model = StrategyDefinition
        fields = [
            "name",
            "slug",
            "strategy_type",
            "description",
            "gate_quantile",
            "top_k",
            "rebalance_freq",
            "gross_exposure",
            "selection_side",
            "signal_combination",
            "action_source_field",
            "action_threshold",
            "is_active",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["advanced_config_json"].initial = json.dumps(dict(self.instance.config or {}), indent=2, sort_keys=True) if self.instance.pk else json.dumps({}, indent=2)

    def clean_advanced_config_json(self) -> dict:
        raw = str(self.cleaned_data.get("advanced_config_json") or "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise forms.ValidationError("Enter a valid JSON object.") from exc
        if not isinstance(payload, dict):
            raise forms.ValidationError("Strategy config must be a JSON object.")
        return payload

    def save(self, commit=True):
        instance = super().save(commit=False)
        config = dict(self.cleaned_data.get("advanced_config_json") or {})
        config.update(
            {
                "gate_quantile": float(self.cleaned_data.get("gate_quantile") or instance.gate_quantile),
                "top_k": int(self.cleaned_data.get("top_k") or instance.top_k),
                "rebalance_freq": str(self.cleaned_data.get("rebalance_freq") or instance.rebalance_freq),
                "gross_exposure": float(self.cleaned_data.get("gross_exposure") or instance.gross_exposure),
                "selection_side": str(self.cleaned_data.get("selection_side") or instance.selection_side),
                "signal_combination": str(self.cleaned_data.get("signal_combination") or instance.signal_combination),
                "action_source_field": str(self.cleaned_data.get("action_source_field") or instance.action_source_field or ""),
                "action_threshold": float(self.cleaned_data.get("action_threshold") or instance.action_threshold),
            }
        )
        instance.config = config
        if commit:
            instance.save()
        return instance
