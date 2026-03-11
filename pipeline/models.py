from __future__ import annotations

from django.db import models


class PipelineRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    class Mode(models.TextChoices):
        STRICT = "strict", "Strict"
        AUTO_BUILD_MISSING = "auto_build_missing", "Auto Build Missing"

    name = models.CharField(max_length=255, blank=True, default="")
    requested_job = models.CharField(max_length=64, db_index=True)
    mode = models.CharField(max_length=32, choices=Mode.choices, default=Mode.STRICT)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    config = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"pipeline-run:{self.pk}:{self.requested_job}:{self.status}"


class JobRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"

    pipeline_run = models.ForeignKey(PipelineRun, on_delete=models.CASCADE, related_name="job_runs")
    job_type = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    config = models.JSONField(default=dict, blank=True)
    input_artifacts = models.ManyToManyField("Artifact", blank=True, related_name="consumed_by")
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at", "id"]

    def __str__(self) -> str:
        return f"job-run:{self.pk}:{self.job_type}:{self.status}"


class Artifact(models.Model):
    pipeline_run = models.ForeignKey(PipelineRun, on_delete=models.CASCADE, related_name="artifacts")
    producer_job = models.ForeignKey(
        JobRun,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="produced_artifacts",
    )
    artifact_type = models.CharField(max_length=64, db_index=True)
    key = models.CharField(max_length=64, unique=True)
    uri = models.CharField(max_length=1024, blank=True, default="")
    content = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    payload_hash = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "id"]
        indexes = [models.Index(fields=["artifact_type", "-created_at"])]

    def __str__(self) -> str:
        return f"artifact:{self.pk}:{self.artifact_type}"


class StrategyDefinition(models.Model):
    class StrategyType(models.TextChoices):
        NOTEBOOK_TOPK_V1 = "notebook_topk_v1", "Notebook Top-K"
        RL_POLICY_V1 = "rl_policy_v1", "RL Policy"

    class RebalanceFreq(models.TextChoices):
        DAILY = "D", "Daily"
        WEEKLY = "W", "Weekly"
        MONTHLY = "M", "Monthly"

    class SelectionSide(models.TextChoices):
        LONG_ONLY = "long_only", "Long Only"
        LONG_SHORT = "long_short", "Long / Short"

    class SignalCombination(models.TextChoices):
        MULTIPLY = "multiply", "Multiply Components"
        MEAN = "mean", "Mean Components"
        DIRECT = "direct", "Direct Signal"

    name = models.CharField(max_length=255, db_index=True)
    slug = models.SlugField(max_length=255, unique=True)
    strategy_type = models.CharField(max_length=64, db_index=True, choices=StrategyType.choices, default=StrategyType.NOTEBOOK_TOPK_V1)
    description = models.TextField(blank=True, default="")
    gate_quantile = models.FloatField(default=0.5)
    top_k = models.PositiveIntegerField(default=20)
    rebalance_freq = models.CharField(max_length=4, choices=RebalanceFreq.choices, default=RebalanceFreq.WEEKLY)
    gross_exposure = models.FloatField(default=0.8)
    selection_side = models.CharField(max_length=32, choices=SelectionSide.choices, default=SelectionSide.LONG_ONLY)
    signal_combination = models.CharField(max_length=32, choices=SignalCombination.choices, default=SignalCombination.MULTIPLY)
    action_source_field = models.CharField(max_length=255, blank=True, default="")
    action_threshold = models.FloatField(default=0.0)
    config = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name", "id"]

    def __str__(self) -> str:
        return f"strategy-definition:{self.pk}:{self.slug}"
