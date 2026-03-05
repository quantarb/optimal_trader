from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("fmp", "0008_rename_fmp_macroob_series__7b6ff4_idx_fmp_macroob_series__673b11_idx"),
    ]

    operations = [
        migrations.AlterField(
            model_name="macroobservation",
            name="value",
            field=models.FloatField(),
        ),
    ]
