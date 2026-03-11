from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("pipeline", "0002_strategydefinition"),
    ]

    operations = [
        migrations.AddField(
            model_name="strategydefinition",
            name="action_source_field",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
        migrations.AddField(
            model_name="strategydefinition",
            name="action_threshold",
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name="strategydefinition",
            name="gate_quantile",
            field=models.FloatField(default=0.5),
        ),
        migrations.AddField(
            model_name="strategydefinition",
            name="gross_exposure",
            field=models.FloatField(default=0.8),
        ),
        migrations.AddField(
            model_name="strategydefinition",
            name="rebalance_freq",
            field=models.CharField(choices=[("D", "Daily"), ("W", "Weekly"), ("M", "Monthly")], default="W", max_length=4),
        ),
        migrations.AddField(
            model_name="strategydefinition",
            name="selection_side",
            field=models.CharField(choices=[("long_only", "Long Only"), ("long_short", "Long / Short")], default="long_only", max_length=32),
        ),
        migrations.AddField(
            model_name="strategydefinition",
            name="signal_combination",
            field=models.CharField(choices=[("multiply", "Multiply Components"), ("mean", "Mean Components"), ("direct", "Direct Signal")], default="multiply", max_length=32),
        ),
        migrations.AddField(
            model_name="strategydefinition",
            name="top_k",
            field=models.PositiveIntegerField(default=20),
        ),
        migrations.AlterField(
            model_name="strategydefinition",
            name="strategy_type",
            field=models.CharField(choices=[("notebook_topk_v1", "Notebook Top-K"), ("rl_policy_v1", "RL Policy")], db_index=True, default="notebook_topk_v1", max_length=64),
        ),
    ]
