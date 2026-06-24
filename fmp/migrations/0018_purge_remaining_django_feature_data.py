from __future__ import annotations

from django.db import migrations


def purge_remaining_django_feature_data(apps, schema_editor):
    SymbolSectionHistorical = apps.get_model("fmp", "SymbolSectionHistorical")
    SymbolSectionSnapshot = apps.get_model("fmp", "SymbolSectionSnapshot")
    SymbolSectionState = apps.get_model("fmp", "SymbolSectionState")
    MacroObservation = apps.get_model("fmp", "MacroObservation")
    MacroSeries = apps.get_model("fmp", "MacroSeries")
    EconomicIndicatorObservation = apps.get_model("fmp", "EconomicIndicatorObservation")
    EconomicIndicatorSeries = apps.get_model("fmp", "EconomicIndicatorSeries")
    TreasuryRateObservation = apps.get_model("fmp", "TreasuryRateObservation")
    TreasuryRateSeries = apps.get_model("fmp", "TreasuryRateSeries")
    PositionSummaryObservation = apps.get_model("fmp", "PositionSummaryObservation")
    PositionSummarySeries = apps.get_model("fmp", "PositionSummarySeries")
    UniverseDownloadJob = apps.get_model("fmp", "UniverseDownloadJob")
    WorkflowState = apps.get_model("fmp", "WorkflowState")

    SymbolSectionHistorical.objects.all().delete()
    SymbolSectionSnapshot.objects.all().delete()
    SymbolSectionState.objects.all().delete()

    MacroObservation.objects.all().delete()
    MacroSeries.objects.all().delete()

    EconomicIndicatorObservation.objects.all().delete()
    EconomicIndicatorSeries.objects.all().delete()

    TreasuryRateObservation.objects.all().delete()
    TreasuryRateSeries.objects.all().delete()

    PositionSummaryObservation.objects.all().delete()
    PositionSummarySeries.objects.all().delete()

    UniverseDownloadJob.objects.all().delete()
    WorkflowState.objects.all().delete()


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0017_purge_legacy_feature_data"),
    ]

    operations = [
        migrations.RunPython(purge_remaining_django_feature_data, migrations.RunPython.noop),
    ]
