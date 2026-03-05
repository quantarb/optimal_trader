from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ModelArtifact",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(db_index=True, max_length=255)),
                ("version", models.PositiveIntegerField(default=1)),
                ("framework", models.CharField(blank=True, default="", max_length=64)),
                ("task_type", models.CharField(blank=True, default="", max_length=64)),
                ("target_col", models.CharField(blank=True, default="", max_length=128)),
                ("feature_cols", models.JSONField(blank=True, default=list)),
                ("metrics", models.JSONField(blank=True, default=dict)),
                ("params", models.JSONField(blank=True, default=dict)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("artifact_format", models.CharField(default="pickle", max_length=32)),
                ("artifact_blob", models.BinaryField()),
                ("artifact_size_bytes", models.PositiveBigIntegerField(default=0)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["name", "-version"],
                "unique_together": {("name", "version")},
            },
        ),
        migrations.CreateModel(
            name="ModelTrainingJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(db_index=True, max_length=255)),
                ("framework", models.CharField(max_length=64)),
                ("algorithm", models.CharField(max_length=128)),
                ("task_type", models.CharField(max_length=64)),
                ("target_col", models.CharField(max_length=128)),
                ("feature_cols", models.JSONField(blank=True, default=list)),
                ("split_ratio", models.FloatField(default=0.8)),
                ("params", models.JSONField(blank=True, default=dict)),
                ("notes", models.TextField(blank=True, default="")),
                ("status", models.CharField(choices=[("pending", "Pending"), ("running", "Running"), ("succeeded", "Succeeded"), ("failed", "Failed")], default="pending", max_length=16)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("latest_artifact", models.ForeignKey(blank=True, null=True, on_delete=models.deletion.SET_NULL, related_name="training_jobs", to="ml.modelartifact")),
            ],
            options={
                "ordering": ["-created_at", "name"],
            },
        ),
    ]
