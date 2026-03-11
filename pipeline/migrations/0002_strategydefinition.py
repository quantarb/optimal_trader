from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("pipeline", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="StrategyDefinition",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(db_index=True, max_length=255)),
                ("slug", models.SlugField(max_length=255, unique=True)),
                ("strategy_type", models.CharField(db_index=True, max_length=64)),
                ("description", models.TextField(blank=True, default="")),
                ("config", models.JSONField(blank=True, default=dict)),
                ("is_active", models.BooleanField(db_index=True, default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "ordering": ["name", "id"],
            },
        ),
    ]
