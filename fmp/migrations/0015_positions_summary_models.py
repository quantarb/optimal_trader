from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0014_workflowstate"),
    ]

    operations = [
        migrations.CreateModel(
            name="PositionSummarySeries",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("last_fetched_at", models.DateTimeField(blank=True, null=True)),
                ("min_report_date", models.DateField(blank=True, null=True)),
                ("max_report_date", models.DateField(blank=True, null=True)),
                ("last_year", models.IntegerField(blank=True, null=True)),
                ("last_quarter", models.IntegerField(blank=True, null=True)),
                ("report_count", models.IntegerField(default=0)),
                ("last_updated", models.DateTimeField(auto_now=True)),
                (
                    "symbol",
                    models.OneToOneField(
                        on_delete=models.deletion.CASCADE,
                        related_name="position_summary_series",
                        to="fmp.symbol",
                    ),
                ),
            ],
            options={"ordering": ["symbol__symbol"]},
        ),
        migrations.CreateModel(
            name="PositionSummaryObservation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("report_year", models.IntegerField()),
                ("report_quarter", models.IntegerField()),
                ("report_date", models.DateField(blank=True, null=True)),
                ("investor_count", models.IntegerField(blank=True, null=True)),
                ("shares_held", models.FloatField(blank=True, null=True)),
                ("investment_value", models.FloatField(blank=True, null=True)),
                ("ownership_pct", models.FloatField(blank=True, null=True)),
                ("shares_change", models.FloatField(blank=True, null=True)),
                ("investment_change", models.FloatField(blank=True, null=True)),
                ("ownership_pct_change", models.FloatField(blank=True, null=True)),
                ("put_call_ratio", models.FloatField(blank=True, null=True)),
                ("call_count", models.IntegerField(blank=True, null=True)),
                ("put_count", models.IntegerField(blank=True, null=True)),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "series",
                    models.ForeignKey(
                        on_delete=models.deletion.CASCADE,
                        related_name="observations",
                        to="fmp.positionsummaryseries",
                    ),
                ),
            ],
            options={
                "ordering": ["-report_date", "-report_year", "-report_quarter"],
            },
        ),
        migrations.AddIndex(
            model_name="positionsummaryobservation",
            index=models.Index(fields=["series", "-report_year", "-report_quarter"], name="fmp_possu_series__c5b7c7_idx"),
        ),
        migrations.AddIndex(
            model_name="positionsummaryobservation",
            index=models.Index(fields=["series", "-report_date"], name="fmp_possu_series__0d4d77_idx"),
        ),
        migrations.AlterUniqueTogether(
            name="positionsummaryobservation",
            unique_together={("series", "report_year", "report_quarter")},
        ),
    ]
