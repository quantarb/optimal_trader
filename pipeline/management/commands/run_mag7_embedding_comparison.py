from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from features.pipeline_builders import (
    REPRESENTATION_EMBEDDING_MODEL_VERSION,
    _append_representation_embedding_columns,
)
from ml.execution import _dedupe_label_frame, infer_feature_family_columns, load_artifact_csv_frame, train_model_from_artifact_inputs
from pipeline.management.commands.run_mag7_backtest import MAG7_SYMBOLS
from pipeline.models import Artifact, PipelineRun
from pipeline.service_runtime import (
    artifact_payload_hash,
    json_safe_value,
    write_frame_artifact,
    write_payload_artifact,
)
from pipeline.services import execute_pipeline_run
from settings import BASE_DIR


LABEL_KS = [1, 2, 4, 8]


class Command(BaseCommand):
    help = "Compare MAG7 numeric features against grouped representation embeddings."

    def add_arguments(self, parser):
        parser.add_argument("--universe-artifact-id", type=int, default=0)
        parser.add_argument("--label-artifact-id", type=int, default=0)
        parser.add_argument("--feature-artifact-id", type=int, default=0)
        parser.add_argument("--min-profit-pct", type=float, default=10.0)
        parser.add_argument("--split-ratio", type=float, default=0.8)
        parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
        parser.add_argument("--embedding-model-version", default=REPRESENTATION_EMBEDDING_MODEL_VERSION)
        parser.add_argument("--embedding-store-dir", default=str(Path(BASE_DIR) / "data" / "embedding_store"))
        parser.add_argument("--embedding-device", default="")
        parser.add_argument("--allow-model-download", action="store_true")
        parser.add_argument("--output-basename", default="mag7_embedding_comparison")

    def handle(self, *args, **options):
        universe_artifact = self._resolve_or_build_artifact(
            artifact_id=int(options["universe_artifact_id"] or 0),
            artifact_type="UNIVERSE",
            name="mag7-embedding-comparison-universe",
            requested_job="universe",
            config={"symbols": MAG7_SYMBOLS},
            input_ids=[],
        )
        label_artifact = self._resolve_or_build_artifact(
            artifact_id=int(options["label_artifact_id"] or 0),
            artifact_type="LABELS",
            name="mag7-embedding-comparison-labels",
            requested_job="labels",
            config={"k_params": {"YE": list(LABEL_KS)}, "min_profit_pct": float(options["min_profit_pct"])},
            input_ids=[int(universe_artifact.id)],
        )
        feature_artifact = self._resolve_or_build_artifact(
            artifact_id=int(options["feature_artifact_id"] or 0),
            artifact_type="FEATURES",
            name="mag7-embedding-comparison-features",
            requested_job="features",
            config={},
            input_ids=[int(universe_artifact.id)],
        )

        embedding_feature_artifact = self._build_embedding_feature_artifact(
            feature_artifact=feature_artifact,
            label_artifact=label_artifact,
            model_name=str(options["embedding_model_name"]).strip() or "sentence-transformers/all-MiniLM-L6-v2",
            model_version=str(options["embedding_model_version"]).strip() or REPRESENTATION_EMBEDDING_MODEL_VERSION,
            store_dir=str(options["embedding_store_dir"]).strip() or str(Path(BASE_DIR) / "data" / "embedding_store"),
            device=str(options["embedding_device"]).strip() or None,
            local_files_only=not bool(options["allow_model_download"]),
        )

        numeric_family_map = infer_feature_family_columns(self._feature_columns_from_artifact(feature_artifact))
        numeric_feature_families = [family for family in numeric_family_map if family != "representation_embedding"]

        numeric_results = self._train_suite(
            feature_artifact=feature_artifact,
            label_artifact=label_artifact,
            feature_families=numeric_feature_families,
            split_ratio=float(options["split_ratio"]),
            model_prefix="mag7_numeric_all_families",
        )
        embedding_results = self._train_suite(
            feature_artifact=embedding_feature_artifact,
            label_artifact=label_artifact,
            feature_families=["representation_embedding"],
            split_ratio=float(options["split_ratio"]),
            model_prefix="mag7_grouped_embedding",
        )

        summary = {
            "artifacts": {
                "universe": int(universe_artifact.id),
                "labels": int(label_artifact.id),
                "numeric_features": int(feature_artifact.id),
                "embedding_features": int(embedding_feature_artifact.id),
            },
            "config": {
                "symbols": list(MAG7_SYMBOLS),
                "label_k_params": {"YE": list(LABEL_KS)},
                "min_profit_pct": float(options["min_profit_pct"]),
                "split_ratio": float(options["split_ratio"]),
                "embedding_model_name": str(options["embedding_model_name"]),
                "embedding_model_version": str(options["embedding_model_version"]),
                "embedding_store_dir": str(options["embedding_store_dir"]),
                "embedding_local_files_only": not bool(options["allow_model_download"]),
            },
            "numeric_all_families": numeric_results,
            "grouped_embedding": embedding_results,
            "comparison": self._comparison_summary(numeric_results, embedding_results),
        }
        summary_uri = write_payload_artifact(
            str(options["output_basename"]).strip() or "mag7_embedding_comparison",
            summary,
        ).uri
        self.stdout.write(self.style.SUCCESS(json.dumps(summary, indent=2, sort_keys=True)))
        self.stdout.write(self.style.SUCCESS(f"Summary written to {summary_uri}"))

    def _resolve_or_build_artifact(
        self,
        *,
        artifact_id: int,
        artifact_type: str,
        name: str,
        requested_job: str,
        config: dict[str, Any],
        input_ids: list[int],
    ) -> Artifact:
        if artifact_id > 0:
            artifact = Artifact.objects.filter(pk=artifact_id, artifact_type=artifact_type).first()
            if artifact is None:
                raise CommandError(f"{artifact_type} artifact #{artifact_id} was not found.")
            return artifact
        pipeline_run = PipelineRun.objects.create(
            name=name,
            requested_job=requested_job,
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.PENDING,
            config=dict(config),
        )
        return execute_pipeline_run(
            pipeline_run=pipeline_run,
            target_job=requested_job,
            mode="strict",
            config=dict(config),
            input_artifact_ids=list(input_ids),
        )

    def _build_embedding_feature_artifact(
        self,
        *,
        feature_artifact: Artifact,
        label_artifact: Artifact,
        model_name: str,
        model_version: str,
        store_dir: str,
        device: str | None,
        local_files_only: bool,
    ) -> Artifact:
        feature_df = load_artifact_csv_frame(feature_artifact)
        label_df = load_artifact_csv_frame(label_artifact)
        if feature_df.empty:
            raise CommandError(f"Feature artifact #{feature_artifact.id} is empty.")
        if label_df.empty:
            raise CommandError(f"Label artifact #{label_artifact.id} is empty.")

        if "k" in label_df.columns:
            label_df = label_df[pd.to_numeric(label_df["k"], errors="coerce").isin(LABEL_KS)].copy()
        label_df = _dedupe_label_frame(label_df)
        key_df = label_df[["date", "symbol"]].drop_duplicates().copy()
        labeled_feature_df = feature_df.merge(key_df, on=["date", "symbol"], how="inner")
        if labeled_feature_df.empty:
            raise CommandError("No overlapping labeled rows were found for embedding generation.")
        labeled_feature_df = labeled_feature_df.sort_values(["date", "symbol"]).reset_index(drop=True)

        grouped_feature_columns = infer_feature_family_columns(self._feature_columns_from_frame(feature_df))
        grouped_feature_columns["representation_embedding"] = list(grouped_feature_columns.get("representation_embedding") or [])
        augmented_df, embedding_columns, embedding_meta = _append_representation_embedding_columns(
            labeled_feature_df,
            grouped_feature_columns,
            config={
                "enabled": True,
                "model_name": model_name,
                "model_version": model_version,
                "store_dir": store_dir,
                "column_prefix": "embedding_",
                "local_files_only": bool(local_files_only),
                "device": device,
            },
        )
        if not embedding_columns:
            raise CommandError("Representation embedding generation produced no embedding columns.")

        embedding_df = augmented_df[["date", "symbol", *embedding_columns]].copy()
        fieldnames = ["date", "symbol", *embedding_columns]
        stored = write_frame_artifact(
            f"features_{uuid.uuid4().hex}",
            frame=embedding_df,
            fieldnames=fieldnames,
        )
        pipeline_run = PipelineRun.objects.create(
            name="mag7-embedding-comparison-embedding-features",
            requested_job="features",
            mode=PipelineRun.Mode.STRICT,
            status=PipelineRun.Status.SUCCEEDED,
            config={
                "source_feature_artifact_id": int(feature_artifact.id),
                "source_label_artifact_id": int(label_artifact.id),
                "representation_embedding_enabled": True,
                "representation_embedding_model_name": model_name,
                "representation_embedding_model_version": model_version,
            },
        )
        content = {
            "rows": int(len(embedding_df)),
            "symbols": int(embedding_df["symbol"].astype(str).nunique()),
            "feature_column_count": int(len(embedding_columns)),
        }
        metadata = {
            "source_feature_artifact_id": int(feature_artifact.id),
            "source_label_artifact_id": int(label_artifact.id),
            "feature_family_columns": {"representation_embedding": list(embedding_columns)},
            "feature_column_count": int(len(embedding_columns)),
            "symbols_processed": int(embedding_df["symbol"].astype(str).nunique()),
            "representation_embedding_enabled": True,
            "representation_embedding_columns": list(embedding_columns),
            "representation_embedding_dimension": int(len(embedding_columns)),
            "representation_embedding_model_name": str(embedding_meta.get("model_name") or model_name),
            "representation_embedding_model_version": str(embedding_meta.get("model_version") or model_version),
            "representation_embedding_store_dir": str(embedding_meta.get("store_dir") or store_dir),
            "representation_embedding_family_groups": dict(embedding_meta.get("family_groups") or {}),
            "labeled_row_count": int(len(embedding_df)),
            "selected_label_ks": list(LABEL_KS),
            **stored.storage_metadata(),
        }
        safe_content = json_safe_value(content)
        return Artifact.objects.create(
            pipeline_run=pipeline_run,
            artifact_type="FEATURES",
            key=uuid.uuid4().hex,
            uri=stored.uri,
            content=safe_content,
            metadata=json_safe_value(metadata),
            payload_hash=artifact_payload_hash(safe_content, stored.uri),
        )

    def _train_suite(
        self,
        *,
        feature_artifact: Artifact,
        label_artifact: Artifact,
        feature_families: list[str],
        split_ratio: float,
        model_prefix: str,
    ) -> dict[str, Any]:
        return {
            "feature_artifact_id": int(feature_artifact.id),
            "feature_families": list(feature_families),
            "feature_count": len(self._selected_feature_columns(feature_artifact, feature_families)),
            "classifier": self._train_one(
                name=f"{model_prefix}_classifier",
                algorithm="random_forest_classifier",
                task_type="classification",
                target_col="label",
                feature_artifact=feature_artifact,
                label_artifact=label_artifact,
                feature_families=feature_families,
                split_ratio=split_ratio,
            ),
            "regressor": self._train_one(
                name=f"{model_prefix}_regressor",
                algorithm="random_forest_regressor",
                task_type="regression",
                target_col="trade_return",
                feature_artifact=feature_artifact,
                label_artifact=label_artifact,
                feature_families=feature_families,
                split_ratio=split_ratio,
            ),
            "autoencoder": self._train_one(
                name=f"{model_prefix}_autoencoder",
                algorithm="autoencoder",
                task_type="reconstruction",
                target_col="trade_return",
                feature_artifact=feature_artifact,
                label_artifact=label_artifact,
                feature_families=feature_families,
                split_ratio=split_ratio,
            ),
        }

    def _train_one(
        self,
        *,
        name: str,
        algorithm: str,
        task_type: str,
        target_col: str,
        feature_artifact: Artifact,
        label_artifact: Artifact,
        feature_families: list[str],
        split_ratio: float,
    ) -> dict[str, Any]:
        saved_model = train_model_from_artifact_inputs(
            name=name,
            algorithm=algorithm,
            task_type=task_type,
            target_col=target_col,
            feature_artifact=feature_artifact,
            label_artifact=label_artifact,
            framework="torch" if algorithm == "autoencoder" else "sklearn",
            split_ratio=float(split_ratio),
            params={},
            feature_families=list(feature_families),
            label_ks=list(LABEL_KS),
        )
        metadata = dict(saved_model.metadata or {})
        return {
            "saved_model_artifact_id": int(saved_model.id),
            "algorithm": str(algorithm),
            "task_type": str(task_type),
            "feature_count": int(len(saved_model.feature_cols or [])),
            "joined_rows": int(metadata.get("joined_rows") or 0),
            "symbols_count": int(metadata.get("symbols_count") or 0),
            "dataset_build_seconds": float(metadata.get("dataset_build_seconds") or 0.0),
            "fit_seconds": float(metadata.get("fit_seconds") or 0.0),
            "metrics": dict(saved_model.metrics or {}),
        }

    def _comparison_summary(self, numeric_results: dict[str, Any], embedding_results: dict[str, Any]) -> dict[str, Any]:
        return {
            "classifier": {
                "numeric_accuracy": self._metric(numeric_results, "classifier", "accuracy"),
                "embedding_accuracy": self._metric(embedding_results, "classifier", "accuracy"),
                "numeric_roc_auc": self._metric(numeric_results, "classifier", "roc_auc"),
                "embedding_roc_auc": self._metric(embedding_results, "classifier", "roc_auc"),
                "embedding_minus_numeric_accuracy": self._delta(
                    self._metric(embedding_results, "classifier", "accuracy"),
                    self._metric(numeric_results, "classifier", "accuracy"),
                ),
                "embedding_minus_numeric_roc_auc": self._delta(
                    self._metric(embedding_results, "classifier", "roc_auc"),
                    self._metric(numeric_results, "classifier", "roc_auc"),
                ),
            },
            "regressor": {
                "numeric_r2": self._metric(numeric_results, "regressor", "r2"),
                "embedding_r2": self._metric(embedding_results, "regressor", "r2"),
                "numeric_mae": self._metric(numeric_results, "regressor", "mae"),
                "embedding_mae": self._metric(embedding_results, "regressor", "mae"),
                "numeric_mse": self._metric(numeric_results, "regressor", "mse"),
                "embedding_mse": self._metric(embedding_results, "regressor", "mse"),
                "embedding_minus_numeric_r2": self._delta(
                    self._metric(embedding_results, "regressor", "r2"),
                    self._metric(numeric_results, "regressor", "r2"),
                ),
                "embedding_minus_numeric_mae": self._delta(
                    self._metric(embedding_results, "regressor", "mae"),
                    self._metric(numeric_results, "regressor", "mae"),
                ),
                "embedding_minus_numeric_mse": self._delta(
                    self._metric(embedding_results, "regressor", "mse"),
                    self._metric(numeric_results, "regressor", "mse"),
                ),
            },
            "autoencoder": {
                "numeric_recon_error_mean": self._metric(numeric_results, "autoencoder", "recon_error_mean"),
                "embedding_recon_error_mean": self._metric(embedding_results, "autoencoder", "recon_error_mean"),
                "numeric_recon_error_p95": self._metric(numeric_results, "autoencoder", "recon_error_p95"),
                "embedding_recon_error_p95": self._metric(embedding_results, "autoencoder", "recon_error_p95"),
                "embedding_minus_numeric_recon_error_mean": self._delta(
                    self._metric(embedding_results, "autoencoder", "recon_error_mean"),
                    self._metric(numeric_results, "autoencoder", "recon_error_mean"),
                ),
                "embedding_minus_numeric_recon_error_p95": self._delta(
                    self._metric(embedding_results, "autoencoder", "recon_error_p95"),
                    self._metric(numeric_results, "autoencoder", "recon_error_p95"),
                ),
            },
        }

    @staticmethod
    def _metric(results: dict[str, Any], model_key: str, metric_name: str) -> float | None:
        try:
            value = ((results.get(model_key) or {}).get("metrics") or {}).get(metric_name)
            return None if value in (None, "") else float(value)
        except Exception:
            return None

    @staticmethod
    def _delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return float(left) - float(right)

    @staticmethod
    def _feature_columns_from_artifact(feature_artifact: Artifact) -> list[str]:
        return Command._feature_columns_from_frame(load_artifact_csv_frame(feature_artifact))

    @staticmethod
    def _feature_columns_from_frame(feature_df: pd.DataFrame) -> list[str]:
        return [str(col) for col in feature_df.columns if str(col) not in {"date", "symbol"}]

    def _selected_feature_columns(self, feature_artifact: Artifact, feature_families: list[str]) -> list[str]:
        family_map = infer_feature_family_columns(self._feature_columns_from_artifact(feature_artifact))
        if not feature_families:
            return self._feature_columns_from_artifact(feature_artifact)
        selected: list[str] = []
        for family_name in list(feature_families):
            selected.extend(list(family_map.get(family_name) or []))
        return list(dict.fromkeys(selected))
