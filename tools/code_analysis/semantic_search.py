from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False

from .repository import ClassRecord, FunctionRecord, RepositoryInventory, build_repository_inventory
from .utils import DEFAULT_EMBEDDING_MODEL, slugify_query


FALLBACK_HASH_DIM = 1024


@dataclass
class CodeChunk:
    chunk_id: str
    kind: str
    module: str
    qualname: str
    path: str
    lineno: int
    end_lineno: int
    text: str
    preview: str
    is_test: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "kind": self.kind,
            "module": self.module,
            "qualname": self.qualname,
            "path": self.path,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "text": self.text,
            "preview": self.preview,
            "is_test": self.is_test,
        }


@dataclass
class SemanticIndexReport:
    backend: str
    model_name: str
    dimension: int
    chunk_count: int
    index_path: str
    chunks_path: str
    embeddings_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_count": self.chunk_count,
            "index_path": self.index_path,
            "chunks_path": self.chunks_path,
            "embeddings_path": self.embeddings_path,
        }


def extract_code_chunks(inventory: RepositoryInventory, *, include_tests: bool = False) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for module_record in inventory.modules.values():
        module_is_test = ".tests" in module_record.module or module_record.module.endswith("_tests")
        if module_is_test and not include_tests:
            continue
        for function_record in module_record.functions:
            is_test = module_is_test or function_record.name.startswith("test_")
            if is_test and not include_tests:
                continue
            text = _chunk_text_for_function(function_record)
            chunks.append(
                CodeChunk(
                    chunk_id=function_record.full_name,
                    kind="function",
                    module=function_record.module,
                    qualname=function_record.qualname,
                    path=function_record.path,
                    lineno=function_record.lineno,
                    end_lineno=function_record.end_lineno,
                    text=text,
                    preview=_preview_text(text),
                    is_test=is_test,
                )
            )
        for class_record in module_record.class_records:
            is_test = module_is_test or class_record.name.startswith("Test")
            if is_test and not include_tests:
                continue
            text = _chunk_text_for_class(class_record)
            chunks.append(
                CodeChunk(
                    chunk_id=class_record.full_name,
                    kind="class",
                    module=class_record.module,
                    qualname=class_record.qualname,
                    path=class_record.path,
                    lineno=class_record.lineno,
                    end_lineno=class_record.end_lineno,
                    text=text,
                    preview=_preview_text(text),
                    is_test=is_test,
                )
            )
    return chunks


def build_semantic_index(
    *,
    root: Path,
    output_dir: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    include_tests: bool = False,
) -> tuple[SemanticIndexReport, list[CodeChunk], np.ndarray]:
    inventory = build_repository_inventory(root)
    chunks = extract_code_chunks(inventory, include_tests=include_tests)
    backend, resolved_model_name, embeddings = _encode_texts(
        [chunk.text for chunk in chunks],
        requested_model_name=model_name,
    )
    index_path = output_dir / "semantic_index.faiss"
    chunks_path = output_dir / "semantic_chunks.json"
    embeddings_path = output_dir / "semantic_embeddings.npy"
    effective_backend = backend if _HAS_FAISS else backend.replace("+faiss", "+numpy")
    if _HAS_FAISS:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)
    chunks_path.write_text(
        json.dumps(
            {
                "backend": effective_backend,
                "model_name": resolved_model_name,
                "dimension": int(embeddings.shape[1]),
                "chunk_count": len(chunks),
                "chunks": [chunk.to_dict() for chunk in chunks],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = SemanticIndexReport(
        backend=effective_backend,
        model_name=resolved_model_name,
        dimension=int(embeddings.shape[1]),
        chunk_count=len(chunks),
        index_path=str(index_path),
        chunks_path=str(chunks_path),
        embeddings_path=str(embeddings_path),
    )
    return report, chunks, embeddings


def load_semantic_index(output_dir: Path) -> tuple[SemanticIndexReport, list[CodeChunk], np.ndarray, Any | None]:
    chunks_payload = json.loads((output_dir / "semantic_chunks.json").read_text(encoding="utf-8"))
    report = SemanticIndexReport(
        backend=str(chunks_payload.get("backend") or "sentence_transformers+faiss"),
        model_name=str(chunks_payload.get("model_name") or DEFAULT_EMBEDDING_MODEL),
        dimension=int(chunks_payload.get("dimension") or 0),
        chunk_count=int(chunks_payload.get("chunk_count") or 0),
        index_path=str(output_dir / "semantic_index.faiss"),
        chunks_path=str(output_dir / "semantic_chunks.json"),
        embeddings_path=str(output_dir / "semantic_embeddings.npy"),
    )
    chunks = [CodeChunk(**row) for row in list(chunks_payload.get("chunks") or [])]
    embeddings = np.load(output_dir / "semantic_embeddings.npy")
    index = None
    if _HAS_FAISS and (output_dir / "semantic_index.faiss").exists():
        index = faiss.read_index(str(output_dir / "semantic_index.faiss"))
    return report, chunks, embeddings, index


def search_semantic_index(
    *,
    query: str,
    output_dir: Path,
    model_name: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    report, chunks, _embeddings, index = load_semantic_index(output_dir)
    backend, resolved_model_name, query_embedding = _encode_texts(
        [query],
        requested_model_name=model_name or report.model_name,
        backend_hint=report.backend,
    )
    if index is not None:
        scores, indices = index.search(query_embedding, max(int(top_k), 1))
    else:
        similarity = np.asarray(embeddings @ query_embedding.T, dtype="float32").reshape(-1)
        top_k = max(int(top_k), 1)
        top_indices = np.argsort(-similarity)[:top_k]
        scores = np.asarray([similarity[top_indices]], dtype="float32")
        indices = np.asarray([top_indices], dtype="int64")
    rows: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if int(idx) < 0 or int(idx) >= len(chunks):
            continue
        chunk = chunks[int(idx)]
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "kind": chunk.kind,
                "module": chunk.module,
                "qualname": chunk.qualname,
                "path": chunk.path,
                "lineno": chunk.lineno,
                "end_lineno": chunk.end_lineno,
                "score": round(float(score), 6),
                "preview": chunk.preview,
            }
        )
    return {
        "query": query,
        "query_slug": slugify_query(query),
        "backend": backend,
        "model_name": resolved_model_name,
        "results": rows,
    }


def semantic_index_markdown(report: SemanticIndexReport) -> str:
    return "\n".join(
        [
            "# Semantic Index",
            "",
            f"- backend: {report.backend}",
            f"- model: `{report.model_name}`",
            f"- dimension: {report.dimension}",
            f"- chunks: {report.chunk_count}",
            f"- index: `{report.index_path}`",
            f"- metadata: `{report.chunks_path}`",
        ]
    )


def search_results_markdown(payload: dict[str, Any]) -> str:
    sections = [
        "# Search Results",
        "",
        f"- query: `{payload['query']}`",
        f"- backend: {payload['backend']}",
        f"- model: `{payload['model_name']}`",
        "",
        "## Matches",
    ]
    sections.extend(
        f"- `{row['chunk_id']}` ({row['kind']}, score {row['score']:.3f})"
        for row in payload["results"]
    )
    return "\n".join(sections)


def _chunk_text_for_function(record: FunctionRecord) -> str:
    header = f"Function {record.full_name}\nModule: {record.module}\n"
    return header + record.source


def _chunk_text_for_class(record: ClassRecord) -> str:
    header = f"Class {record.full_name}\nModule: {record.module}\nMethods: {', '.join(record.methods)}\n"
    return header + record.source


def _preview_text(text: str, *, limit: int = 280) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _encode_texts(
    texts: list[str],
    *,
    requested_model_name: str,
    backend_hint: str | None = None,
) -> tuple[str, str, np.ndarray]:
    if not texts:
        return _fallback_backend_name(), _fallback_model_name(), np.zeros((0, FALLBACK_HASH_DIM), dtype="float32")
    if _should_use_sentence_transformers(requested_model_name=requested_model_name, backend_hint=backend_hint):
        try:
            model = _load_embedding_model(requested_model_name)
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return _backend_name("sentence_transformers"), requested_model_name, np.asarray(embeddings, dtype="float32")
        except Exception:
            if backend_hint == "sentence_transformers+faiss" and os.environ.get("CODE_ANALYSIS_REQUIRE_SENTENCE_TRANSFORMERS") == "1":
                raise
    vectorizer = HashingVectorizer(
        n_features=FALLBACK_HASH_DIM,
        alternate_sign=False,
        norm=None,
        ngram_range=(1, 2),
    )
    embeddings = vectorizer.transform(texts).toarray().astype("float32")
    embeddings = normalize(embeddings, norm="l2", copy=False)
    return _fallback_backend_name(), _fallback_model_name(), embeddings


def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=True)


def _fallback_model_name() -> str:
    return f"hashing_vectorizer_{FALLBACK_HASH_DIM}"


def _backend_name(prefix: str) -> str:
    return f"{prefix}+{'faiss' if _HAS_FAISS else 'numpy'}"


def _fallback_backend_name() -> str:
    return _backend_name("hashing_vectorizer")


def _should_use_sentence_transformers(*, requested_model_name: str, backend_hint: str | None) -> bool:
    if backend_hint not in {None, "sentence_transformers+faiss"}:
        return False
    if os.environ.get("CODE_ANALYSIS_USE_SENTENCE_TRANSFORMERS") == "1":
        return True
    return Path(requested_model_name).exists()
