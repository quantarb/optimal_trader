from django.contrib import admin

from .models import ModelArtifact


@admin.register(ModelArtifact)
class ModelArtifactAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "version",
        "framework",
        "task_type",
        "artifact_size_bytes",
        "is_active",
        "created_at",
    )
    list_filter = ("framework", "task_type", "is_active")
    search_fields = ("name", "target_col")
    ordering = ("name", "-version")
