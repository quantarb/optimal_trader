from django.contrib import admin

from .models import Artifact, JobRun, PipelineRun


@admin.register(PipelineRun)
class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ("id", "requested_job", "mode", "status", "created_at", "started_at", "finished_at")
    list_filter = ("status", "mode", "requested_job")
    search_fields = ("id", "requested_job", "name")


@admin.register(JobRun)
class JobRunAdmin(admin.ModelAdmin):
    list_display = ("id", "pipeline_run", "job_type", "status", "created_at", "started_at", "finished_at")
    list_filter = ("status", "job_type")
    search_fields = ("id", "job_type", "pipeline_run__id")


@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ("id", "artifact_type", "pipeline_run", "producer_job", "created_at")
    list_filter = ("artifact_type",)
    search_fields = ("id", "artifact_type", "key", "uri")
