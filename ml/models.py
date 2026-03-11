from __future__ import annotations

import pickle
from typing import Any

from django.db import models as django_models


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
