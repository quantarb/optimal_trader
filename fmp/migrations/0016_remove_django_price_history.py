from django.db import migrations


PRICE_SECTION_KEYS = ("prices_div_adj", "prices_unadjusted")


def remove_django_price_history(apps, schema_editor):
    Symbol = apps.get_model("fmp", "Symbol")
    SymbolSectionHistorical = apps.get_model("fmp", "SymbolSectionHistorical")
    SymbolSectionState = apps.get_model("fmp", "SymbolSectionState")

    SymbolSectionHistorical.objects.filter(section_key__in=PRICE_SECTION_KEYS).delete()
    SymbolSectionState.objects.filter(section_key__in=PRICE_SECTION_KEYS).delete()

    for symbol in Symbol.objects.exclude(historical_date_ranges={}).only("id", "historical_date_ranges").iterator():
        ranges = dict(symbol.historical_date_ranges or {})
        changed = False
        for section_key in PRICE_SECTION_KEYS:
            if section_key in ranges:
                ranges.pop(section_key, None)
                changed = True
        if changed:
            symbol.historical_date_ranges = ranges
            symbol.save(update_fields=["historical_date_ranges"])


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0015_positions_summary_models"),
    ]

    operations = [
        migrations.RunPython(remove_django_price_history, reverse_code=migrations.RunPython.noop),
    ]
