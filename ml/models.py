from __future__ import annotations

import pickle
from typing import Any

from django.db import models as django_models

from .feature_families import FEATURE_FAMILY_LABELS


class ModelArtifact(django_models.Model):
    name = django_models.CharField(max_length=255, db_index=True)
    version = django_models.PositiveIntegerField(default=1)
    framework = django_models.CharField(max_length=64, blank=True, default="")
    task_type = django_models.CharField(max_length=64, blank=True, default="")
    target_col = django_models.CharField(max_length=128, blank=True, default="")
    feature_cols = django_models.JSONField(default=list, blank=True)
    metrics = django_models.JSONField(default=dict, blank=True)
    params = django_models.JSONField(default=dict, blank=True)
    metadata = django_models.JSONField(default=dict, blank=True)
    artifact_format = django_models.CharField(max_length=32, default="pickle")
    artifact_blob = django_models.BinaryField()
    artifact_size_bytes = django_models.PositiveBigIntegerField(default=0)
    is_active = django_models.BooleanField(default=True)
    created_at = django_models.DateTimeField(auto_now_add=True)
    updated_at = django_models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (("name", "version"),)
        ordering = ["name", "-version"]

    def __str__(self) -> str:
        return f"{self.name}:v{self.version}"

    def set_artifact(self, model_obj: Any) -> None:
        payload = pickle.dumps(model_obj)
        self.artifact_blob = payload
        self.artifact_size_bytes = len(payload)

    def get_artifact(self) -> Any:
        if not self.artifact_blob:
            raise ValueError("No serialized artifact is stored on this record.")
        return pickle.loads(bytes(self.artifact_blob))


# Existing non-Django ML utilities remain available via `ml.legacy`.


class ModelTrainingJob(django_models.Model):
    STATUS_CHOICES = (
        ("pending", "Pending"),
        ("running", "Running"),
        ("succeeded", "Succeeded"),
        ("failed", "Failed"),
    )

    name = django_models.CharField(max_length=255, db_index=True)
    framework = django_models.CharField(max_length=64)
    algorithm = django_models.CharField(max_length=128)
    task_type = django_models.CharField(max_length=64)
    target_col = django_models.CharField(max_length=128)
    feature_cols = django_models.JSONField(default=list, blank=True)
    split_ratio = django_models.FloatField(default=0.8)
    params = django_models.JSONField(default=dict, blank=True)
    notes = django_models.TextField(blank=True, default="")
    status = django_models.CharField(max_length=16, choices=STATUS_CHOICES, default="pending")
    latest_artifact = django_models.ForeignKey(
        ModelArtifact,
        null=True,
        blank=True,
        on_delete=django_models.SET_NULL,
        related_name="training_jobs",
    )
    created_at = django_models.DateTimeField(auto_now_add=True)
    updated_at = django_models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.status})"

    @property
    def feature_families(self) -> list[str]:
        return list(self.feature_cols or [])

    @feature_families.setter
    def feature_families(self, value) -> None:
        self.feature_cols = list(value or [])

    def feature_family_labels(self) -> list[str]:
        return [FEATURE_FAMILY_LABELS.get(key, key) for key in self.feature_families]

    @property
    def training_symbol(self) -> str:
        context = dict(self.params or {}).get("__job_context__")
        if isinstance(context, dict):
            raw = context.get("symbol")
            if raw:
                return str(raw).strip().upper()
        return ""
