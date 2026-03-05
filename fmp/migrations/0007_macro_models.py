from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0006_remove_symbolsectionhistorical_fmp_symbolsectionhistorical_symbol_section_record_key_uniq_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="MacroSeries",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("code", models.CharField(db_index=True, max_length=128, unique=True)),
                ("display_name", models.CharField(blank=True, default="", max_length=255)),
                ("category", models.CharField(blank=True, default="economic", max_length=32)),
                ("last_fetched_at", models.DateTimeField(blank=True, null=True)),
                ("min_date", models.DateField(blank=True, null=True)),
                ("max_date", models.DateField(blank=True, null=True)),
                ("last_updated", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["code"]},
        ),
        migrations.CreateModel(
            name="MacroObservation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("observation_date", models.DateField()),
                ("value", models.FloatField(blank=True, null=True)),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("series", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="observations", to="fmp.macroseries")),
            ],
            options={"ordering": ["series__code", "-observation_date"]},
        ),
        migrations.AddIndex(
            model_name="macroobservation",
            index=models.Index(fields=["series", "-observation_date"], name="fmp_macroob_series__7b6ff4_idx"),
        ),
        migrations.AlterUniqueTogether(
            name="macroobservation",
            unique_together={("series", "observation_date")},
        ),
    ]
