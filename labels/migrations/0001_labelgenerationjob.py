from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="LabelGenerationJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("status", models.CharField(choices=[("pending", "Pending"), ("running", "Running"), ("completed", "Completed"), ("completed_with_errors", "Completed With Errors"), ("failed", "Failed")], db_index=True, default="pending", max_length=32)),
                ("symbols", models.JSONField(blank=True, default=list)),
                ("total", models.IntegerField(default=0)),
                ("completed", models.IntegerField(default=0)),
                ("generated_labels_count", models.IntegerField(default=0)),
                ("current_symbol", models.CharField(blank=True, default="", max_length=32)),
                ("errors", models.JSONField(blank=True, default=list)),
                ("config", models.JSONField(blank=True, default=dict)),
                ("celery_task_id", models.CharField(blank=True, default="", max_length=255)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
