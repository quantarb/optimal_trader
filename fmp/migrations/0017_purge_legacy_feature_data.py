from __future__ import annotations

from django.db import migrations


LEGACY_SECTION_KEYS = (
    "prices_div_adj",
    "prices_unadjusted",
    "income_statement_ttm",
    "balance_sheet_ttm",
    "cash_flow_ttm",
    "key_metrics_ttm",
    "ratios_ttm",
)

LEGACY_MACRO_PREFIXES = (
    "sector_perf__",
    "industry_perf__",
    "sector_pe__",
    "industry_pe__",
)


def purge_legacy_feature_data(apps, schema_editor):
    Symbol = apps.get_model("fmp", "Symbol")
    SymbolSectionHistorical = apps.get_model("fmp", "SymbolSectionHistorical")
    SymbolSectionSnapshot = apps.get_model("fmp", "SymbolSectionSnapshot")
    SymbolSectionState = apps.get_model("fmp", "SymbolSectionState")
    MacroSeries = apps.get_model("fmp", "MacroSeries")

    SymbolSectionHistorical.objects.filter(section_key__in=LEGACY_SECTION_KEYS).delete()
    SymbolSectionSnapshot.objects.filter(section_key__in=LEGACY_SECTION_KEYS).delete()
    SymbolSectionState.objects.filter(section_key__in=LEGACY_SECTION_KEYS).delete()

    legacy_macro_series = MacroSeries.objects.none()
    for prefix in LEGACY_MACRO_PREFIXES:
        legacy_macro_series = legacy_macro_series | MacroSeries.objects.filter(code__startswith=prefix)
    legacy_macro_series.delete()

    for symbol in Symbol.objects.only("id", "historical_date_ranges").iterator():
        ranges = symbol.historical_date_ranges or {}
        if not isinstance(ranges, dict):
            continue
        changed = False
        for section_key in LEGACY_SECTION_KEYS:
            if section_key in ranges:
                ranges.pop(section_key, None)
                changed = True
        if changed:
            symbol.historical_date_ranges = ranges
            symbol.save(update_fields=["historical_date_ranges"])


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0016_remove_django_price_history"),
    ]

    operations = [
        migrations.RunPython(purge_legacy_feature_data, migrations.RunPython.noop),
    ]
