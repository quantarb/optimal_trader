from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ml", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="modeltrainingjob",
            name="celery_task_id",
            field=models.CharField(blank=True, default="", max_length=255),
        ),
        migrations.AddField(
            model_name="modeltrainingjob",
            name="current_symbol",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
        migrations.AddField(
            model_name="modeltrainingjob",
            name="errors",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="modeltrainingjob",
            name="finished_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="modeltrainingjob",
            name="progress_completed",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="modeltrainingjob",
            name="progress_total",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="modeltrainingjob",
            name="started_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
