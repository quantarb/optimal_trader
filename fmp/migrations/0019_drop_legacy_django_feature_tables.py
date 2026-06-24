from __future__ import annotations

from django.db import migrations


def drop_legacy_feature_tables(apps, schema_editor):
    model_names = [
        ("fmp", "SymbolSectionHistorical"),
        ("fmp", "SymbolSectionSnapshot"),
        ("fmp", "SymbolSectionState"),
        ("fmp", "MacroObservation"),
        ("fmp", "MacroSeries"),
        ("fmp", "EconomicIndicatorObservation"),
        ("fmp", "EconomicIndicatorSeries"),
        ("fmp", "TreasuryRateObservation"),
        ("fmp", "TreasuryRateSeries"),
        ("fmp", "PositionSummaryObservation"),
        ("fmp", "PositionSummarySeries"),
        ("fmp", "UniverseDownloadJob"),
        ("fmp", "WorkflowState"),
    ]

    for app_label, model_name in model_names:
        model = apps.get_model(app_label, model_name)
        schema_editor.delete_model(model)


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0018_purge_remaining_django_feature_data"),
    ]

    operations = [
        migrations.RunPython(drop_legacy_feature_tables, migrations.RunPython.noop),
    ]
