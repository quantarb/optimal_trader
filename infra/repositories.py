from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ml.models import ModelArtifact
from pipeline.models import Artifact


@dataclass
class DjangoArtifactRepository:
    """Django-backed artifact repository used by workflows."""

    def get_pipeline_artifact(self, artifact_id: int, *, artifact_type: str | None = None):
        queryset = Artifact.objects.filter(pk=int(artifact_id))
        if artifact_type:
            queryset = queryset.filter(artifact_type=str(artifact_type))
        return queryset.first()

    def list_pipeline_artifacts(self, artifact_ids: Sequence[int], *, artifact_types: Sequence[str] | None = None) -> list[Artifact]:
        queryset = Artifact.objects.filter(id__in=[int(value) for value in list(artifact_ids or [])])
        if artifact_types:
            queryset = queryset.filter(artifact_type__in=[str(value) for value in artifact_types])
        return list(queryset.order_by("id"))

    def get_saved_model(self, saved_model_id: int):
        return ModelArtifact.objects.filter(pk=int(saved_model_id)).first()
