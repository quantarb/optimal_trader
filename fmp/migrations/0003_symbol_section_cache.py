from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0002_lookup_models"),
    ]

    operations = [
        migrations.CreateModel(
            name="SymbolSectionHistorical",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("section_key", models.CharField(max_length=100)),
                ("record_key", models.CharField(max_length=64)),
                ("record_date", models.DateField(blank=True, null=True)),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("symbol", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="section_history", to="fmp.symbol")),
            ],
            options={
                "ordering": ["section_key", "-record_date", "-updated_at"],
            },
        ),
        migrations.CreateModel(
            name="SymbolSectionSnapshot",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("section_key", models.CharField(max_length=100)),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("symbol", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="section_snapshots", to="fmp.symbol")),
            ],
            options={
                "ordering": ["section_key"],
            },
        ),
        migrations.CreateModel(
            name="SymbolSectionState",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("section_key", models.CharField(max_length=100)),
                ("kind", models.CharField(choices=[("snapshot", "snapshot"), ("historical", "historical")], max_length=16)),
                ("last_fetched_at", models.DateTimeField(blank=True, null=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("symbol", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="section_states", to="fmp.symbol")),
            ],
            options={
                "ordering": ["section_key"],
            },
        ),
        migrations.AddConstraint(
            model_name="symbolsectionstate",
            constraint=models.UniqueConstraint(fields=("symbol", "section_key"), name="fmp_symbolsectionstate_symbol_section_key_uniq"),
        ),
        migrations.AddConstraint(
            model_name="symbolsectionsnapshot",
            constraint=models.UniqueConstraint(fields=("symbol", "section_key"), name="fmp_symbolsectionsnapshot_symbol_section_key_uniq"),
        ),
        migrations.AddConstraint(
            model_name="symbolsectionhistorical",
            constraint=models.UniqueConstraint(fields=("symbol", "section_key", "record_key"), name="fmp_symbolsectionhistorical_symbol_section_record_key_uniq"),
        ),
        migrations.AddIndex(
            model_name="symbolsectionhistorical",
            index=models.Index(fields=["symbol", "section_key", "-record_date"], name="fmp_hist_sym_sect_date_idx"),
        ),
    ]
