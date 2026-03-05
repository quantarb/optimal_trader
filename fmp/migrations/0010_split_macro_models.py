from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0009_macro_value_required"),
    ]

    operations = [
        migrations.CreateModel(
            name="EconomicIndicatorSeries",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("code", models.CharField(db_index=True, max_length=128, unique=True)),
                ("display_name", models.CharField(blank=True, default="", max_length=255)),
                ("last_fetched_at", models.DateTimeField(blank=True, null=True)),
                ("min_date", models.DateField(blank=True, null=True)),
                ("max_date", models.DateField(blank=True, null=True)),
                ("last_updated", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["code"]},
        ),
        migrations.CreateModel(
            name="TreasuryRateSeries",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("code", models.CharField(db_index=True, max_length=128, unique=True)),
                ("display_name", models.CharField(blank=True, default="", max_length=255)),
                ("last_fetched_at", models.DateTimeField(blank=True, null=True)),
                ("min_date", models.DateField(blank=True, null=True)),
                ("max_date", models.DateField(blank=True, null=True)),
                ("last_updated", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["code"]},
        ),
        migrations.CreateModel(
            name="EconomicIndicatorObservation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("observation_date", models.DateField()),
                ("value", models.FloatField()),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("series", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="observations", to="fmp.economicindicatorseries")),
            ],
            options={"ordering": ["series__code", "-observation_date"]},
        ),
        migrations.CreateModel(
            name="TreasuryRateObservation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("observation_date", models.DateField()),
                ("value", models.FloatField()),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("series", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="observations", to="fmp.treasuryrateseries")),
            ],
            options={"ordering": ["series__code", "-observation_date"]},
        ),
        migrations.AddIndex(
            model_name="economicindicatorobservation",
            index=models.Index(fields=["series", "-observation_date"], name="fmp_econo_series__9a85ae_idx"),
        ),
        migrations.AddIndex(
            model_name="treasuryrateobservation",
            index=models.Index(fields=["series", "-observation_date"], name="fmp_treas_series__648ad3_idx"),
        ),
        migrations.AlterUniqueTogether(
            name="economicindicatorobservation",
            unique_together={("series", "observation_date")},
        ),
        migrations.AlterUniqueTogether(
            name="treasuryrateobservation",
            unique_together={("series", "observation_date")},
        ),
    ]
