"""ONNX Runtime versions of the embedding model and the reranker, for serving.

The web app runs without PyTorch: both models are exported once to ONNX (int8
dynamic quantisation, see onnx_export.py) and run with onnxruntime, tokenised with
the `tokenizers` library from the exported tokenizer.json.

The classes mimic the small part of the sentence-transformers API that
``retrieval.Retriever`` uses (``encode`` and ``predict``), so they can be passed to
it unchanged.

Model directory layout (written by onnx_export.py):
    model.onnx | model.int8.onnx   graph (inputs input_ids, attention_mask[, token_type_ids])
    tokenizer.json                 Hugging Face fast tokenizer
    ragk8s_onnx.json               {"kind", "max_length", "pooling", "normalize", "activation", "source_model"}
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ONNX_DIR = Path("models") / "onnx"
META_FILE = "ragk8s_onnx.json"
# fp32: plain export. int8: dynamic int8 on every supported op (word embeddings included).
# int8-matmul: dynamic int8, per channel, on MatMul weights only (embeddings stay fp32).
VARIANT_FILES = {"fp32": "model.onnx", "int8": "model.int8.onnx", "int8-matmul": "model.int8-matmul.onnx"}
DEFAULT_VARIANT = "int8"


def _session(model_dir: Path, variant: str, threads: int) -> Any:
    import onnxruntime as ort

    if variant not in VARIANT_FILES:
        raise ValueError(f"unknown ONNX variant {variant!r}, expected one of {sorted(VARIANT_FILES)}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = threads  # 1 = one vCPU on Cloud Run
    options.inter_op_num_threads = 1
    path = model_dir / VARIANT_FILES[variant]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m ragk8s onnx export` first")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _tokenizer(model_dir: Path, max_length: int) -> Any:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]") or 0, pad_token="[PAD]")
    return tok


def _feeds(session: Any, encodings: Sequence[Any]) -> dict[str, npt.NDArray[np.int64]]:
    arrays = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }
    return {i.name: arrays[i.name] for i in session.get_inputs()}


class OnnxEmbedder:
    """Mean-pooled, L2-normalised sentence embeddings (all-MiniLM-L6-v2 style)."""

    def __init__(
        self, model_dir: Path = ONNX_DIR / "embedder", variant: str = DEFAULT_VARIANT, threads: int = 1
    ) -> None:
        self.meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        if self.meta.get("pooling", "mean") != "mean":
            raise ValueError(f"unsupported pooling: {self.meta['pooling']}")
        self._session = _session(model_dir, variant, threads)
        self._tok = _tokenizer(model_dir, int(self.meta["max_length"]))

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **_: Any,
    ) -> npt.NDArray[np.float32]:
        out = []
        for start in range(0, len(sentences), batch_size):
            encodings = self._tok.encode_batch(list(sentences[start : start + batch_size]))
            feeds = _feeds(self._session, encodings)
            hidden = self._session.run(None, feeds)[0]
            mask = np.array([e.attention_mask for e in encodings], dtype=np.float32)[..., None]
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            if normalize_embeddings or self.meta.get("normalize", True):
                pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.append(pooled.astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


class OnnxCrossEncoder:
    """Relevance score of (query, passage) pairs (ms-marco cross-encoder style)."""

    def __init__(
        self,
        model_dir: Path = ONNX_DIR / "reranker",
        variant: str = DEFAULT_VARIANT,
        threads: int = 1,
        max_length: int | None = None,
    ) -> None:
        """``max_length`` truncates (query, passage) pairs below the model's own limit, to cut latency."""
        self.meta = json.loads((model_dir / META_FILE).read_text(encoding="utf-8"))
        self._session = _session(model_dir, variant, threads)
        self.max_length = min(max_length or int(self.meta["max_length"]), int(self.meta["max_length"]))
        self._tok = _tokenizer(model_dir, self.max_length)

    def predict(self, pairs: Sequence[tuple[str, str]], batch_size: int = 16, **_: Any) -> npt.NDArray[np.float32]:
        scores = []
        for start in range(0, len(pairs), batch_size):
            batch = [(q, p) for q, p in pairs[start : start + batch_size]]
            encodings = self._tok.encode_batch(batch)
            logits = self._session.run(None, _feeds(self._session, encodings))[0]
            scores.append(np.asarray(logits, dtype=np.float32).reshape(len(batch), -1)[:, 0])
        values = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
        if self.meta.get("activation") == "sigmoid":
            values = 1.0 / (1.0 + np.exp(-values))
        return values.astype(np.float32)
