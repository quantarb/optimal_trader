from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0003_symbol_section_cache"),
    ]

    operations = [
        migrations.AddField(
            model_name="symbol",
            name="historical_date_ranges",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
