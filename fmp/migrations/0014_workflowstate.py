from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("fmp", "0013_universedownloadjob_metrics"),
    ]

    operations = [
        migrations.CreateModel(
            name="WorkflowState",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("key", models.CharField(default="default", max_length=64, unique=True)),
                ("universe_symbols", models.JSONField(blank=True, default=list)),
                ("universe_filters", models.JSONField(blank=True, default=dict)),
                ("labels_config", models.JSONField(blank=True, default=dict)),
                ("label_target_col", models.CharField(blank=True, default="label", max_length=128)),
                ("labels_generated_count", models.IntegerField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["-updated_at"],
            },
        ),
    ]
