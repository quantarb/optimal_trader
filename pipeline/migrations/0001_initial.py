from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="PipelineRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(blank=True, default="", max_length=255)),
                ("requested_job", models.CharField(db_index=True, max_length=64)),
                (
                    "mode",
                    models.CharField(
                        choices=[("strict", "Strict"), ("auto_build_missing", "Auto Build Missing")],
                        default="strict",
                        max_length=32,
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[("pending", "Pending"), ("running", "Running"), ("succeeded", "Succeeded"), ("failed", "Failed")],
                        db_index=True,
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("config", models.JSONField(blank=True, default=dict)),
                ("error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="JobRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("job_type", models.CharField(db_index=True, max_length=64)),
                (
                    "status",
                    models.CharField(
                        choices=[("pending", "Pending"), ("running", "Running"), ("succeeded", "Succeeded"), ("failed", "Failed")],
                        db_index=True,
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("config", models.JSONField(blank=True, default=dict)),
                ("error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "pipeline_run",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="job_runs", to="pipeline.pipelinerun"),
                ),
            ],
            options={"ordering": ["created_at", "id"]},
        ),
        migrations.CreateModel(
            name="Artifact",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("artifact_type", models.CharField(db_index=True, max_length=64)),
                ("key", models.CharField(max_length=64, unique=True)),
                ("uri", models.CharField(blank=True, default="", max_length=1024)),
                ("content", models.JSONField(blank=True, default=dict)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("payload_hash", models.CharField(blank=True, default="", max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "pipeline_run",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="artifacts", to="pipeline.pipelinerun"),
                ),
                (
                    "producer_job",
                    models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="produced_artifacts", to="pipeline.jobrun"),
                ),
            ],
            options={"ordering": ["-created_at", "id"]},
        ),
        migrations.AddField(
            model_name="jobrun",
            name="input_artifacts",
            field=models.ManyToManyField(blank=True, related_name="consumed_by", to="pipeline.artifact"),
        ),
        migrations.AddIndex(
            model_name="artifact",
            index=models.Index(fields=["artifact_type", "-created_at"], name="pipeline_art_artifac_e5319f_idx"),
        ),
    ]
