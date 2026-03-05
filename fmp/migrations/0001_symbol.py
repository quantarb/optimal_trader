from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Symbol",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("symbol", models.CharField(db_index=True, max_length=32, unique=True)),
                ("company_name", models.CharField(blank=True, default="", max_length=255)),
                ("exchange", models.CharField(blank=True, default="", max_length=64)),
                ("country", models.CharField(blank=True, default="", max_length=8)),
                ("sector", models.CharField(blank=True, default="", max_length=128)),
                ("industry", models.CharField(blank=True, default="", max_length=255)),
                ("market_cap", models.FloatField(blank=True, null=True)),
                ("price", models.FloatField(blank=True, null=True)),
                ("beta", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("dividend", models.FloatField(blank=True, null=True)),
                ("dividend_yield", models.FloatField(blank=True, null=True)),
                ("payload", models.JSONField(blank=True, default=dict)),
                ("last_date_updated", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ["symbol"]},
        ),
    ]
