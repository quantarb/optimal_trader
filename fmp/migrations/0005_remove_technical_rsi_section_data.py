from django.db import migrations


def purge_technical_rsi_section(apps, schema_editor):
    Symbol = apps.get_model("fmp", "Symbol")
    SymbolSectionHistorical = apps.get_model("fmp", "SymbolSectionHistorical")
    SymbolSectionSnapshot = apps.get_model("fmp", "SymbolSectionSnapshot")
    SymbolSectionState = apps.get_model("fmp", "SymbolSectionState")

    section_key = "technical_rsi"

    SymbolSectionHistorical.objects.filter(section_key=section_key).delete()
    SymbolSectionSnapshot.objects.filter(section_key=section_key).delete()
    SymbolSectionState.objects.filter(section_key=section_key).delete()

    for sym in Symbol.objects.all().only("id", "historical_date_ranges"):
        ranges = sym.historical_date_ranges or {}
        if isinstance(ranges, dict) and section_key in ranges:
            ranges.pop(section_key, None)
            sym.historical_date_ranges = ranges
            sym.save(update_fields=["historical_date_ranges"])


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0004_symbol_historical_date_ranges"),
    ]

    operations = [
        migrations.RunPython(purge_technical_rsi_section, migrations.RunPython.noop),
    ]

