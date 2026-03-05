from django.test import TestCase
from django.urls import reverse

from fmp.models import Symbol

from .models import ModelTrainingJob
from .store import load_model_artifact, save_model_artifact


class ModelArtifactStoreTests(TestCase):
    def test_save_and_load_round_trip(self):
        record = save_model_artifact(
            name="rf_classifier",
            model_obj={"weights": [1, 2, 3]},
            framework="sklearn",
            task_type="classification",
            feature_cols=["a", "b"],
            metrics={"accuracy": 0.91},
        )

        self.assertEqual(record.version, 1)
        self.assertGreater(record.artifact_size_bytes, 0)
        self.assertEqual(load_model_artifact(name="rf_classifier"), {"weights": [1, 2, 3]})


class ModelTrainingViewTests(TestCase):
    def test_post_creates_training_job(self):
        Symbol.objects.create(symbol="AAPL")
        response = self.client.post(
            reverse("train_model"),
            {
                "name": "baseline_rf",
                "symbol": "AAPL",
                "framework": "sklearn",
                "algorithm": "random_forest_classifier",
                "task_type": "classification",
                "target_col": "label",
                "feature_families": ["prices_div_adj", "earnings"],
                "split_ratio": "0.8",
                "params_json": '{"n_estimators": 100}',
                "notes": "Initial baseline",
            },
        )

        self.assertEqual(response.status_code, 200)
        job = ModelTrainingJob.objects.get(name="baseline_rf")
        self.assertEqual(job.status, "pending")
        self.assertEqual(job.feature_families, ["prices_div_adj", "earnings"])
        self.assertEqual(job.params["n_estimators"], 100)
        self.assertEqual(job.training_symbol, "AAPL")

    def test_training_page_can_delete_job(self):
        job = ModelTrainingJob.objects.create(
            name="delete_me",
            framework="sklearn",
            algorithm="random_forest_classifier",
            task_type="classification",
            target_col="label",
            feature_cols=["prices_div_adj"],
            split_ratio=0.8,
            params={"__job_context__": {"symbol": "AAPL"}},
            status="pending",
        )

        response = self.client.post(
            reverse("train_model"),
            {"delete_job_id": str(job.id)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Training job deleted.")
        self.assertFalse(ModelTrainingJob.objects.filter(pk=job.pk).exists())

    def test_model_artifact_detail_view_renders_metrics(self):
        artifact = save_model_artifact(
            name="rf_classifier",
            model_obj={"weights": [1, 2, 3]},
            framework="sklearn",
            task_type="classification",
            feature_cols=["feature_a", "feature_b"],
            metrics={
                "accuracy": 0.91,
                "confusion_matrix": [[10, 2], [1, 9]],
            },
            params={"n_estimators": 100},
            metadata={
                "symbol": "AAPL",
                "model_summary": "TOP 10 FEATURES:\n- feature_a: 0.7300\n- feature_b: 0.2700",
            },
        )
        job = ModelTrainingJob.objects.create(
            name="artifact_job",
            framework="sklearn",
            algorithm="random_forest_classifier",
            task_type="classification",
            target_col="label",
            feature_cols=["prices_div_adj"],
            split_ratio=0.8,
            params={"__job_context__": {"symbol": "AAPL"}},
            status="succeeded",
            latest_artifact=artifact,
        )

        response = self.client.get(reverse("model_artifact_detail", args=[artifact.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "rf_classifier")
        self.assertContains(response, "&quot;accuracy&quot;: 0.91")
        self.assertContains(response, "Model Summary")
        self.assertContains(response, "feature_a")
        self.assertContains(response, "artifact_job")

    def test_model_artifact_detail_view_can_delete_artifact(self):
        artifact = save_model_artifact(
            name="rf_classifier",
            model_obj={"weights": [1, 2, 3]},
            framework="sklearn",
            task_type="classification",
            feature_cols=["feature_a"],
        )
        job = ModelTrainingJob.objects.create(
            name="artifact_job",
            framework="sklearn",
            algorithm="random_forest_classifier",
            task_type="classification",
            target_col="label",
            feature_cols=["prices_div_adj"],
            split_ratio=0.8,
            params={"__job_context__": {"symbol": "AAPL"}},
            status="succeeded",
            latest_artifact=artifact,
        )

        response = self.client.post(
            reverse("model_artifact_detail", args=[artifact.id]),
            {"delete_artifact": "1"},
        )

        self.assertRedirects(response, reverse("train_model"))
        self.assertFalse(job.__class__.objects.filter(pk=job.pk, latest_artifact__isnull=False).exists())
        self.assertFalse(type(artifact).objects.filter(pk=artifact.pk).exists())
