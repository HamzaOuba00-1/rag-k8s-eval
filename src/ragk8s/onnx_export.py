"""Export the embedding model and the reranker to ONNX, then check the rankings.

Runs where PyTorch and the Hugging Face models are available (the development PC):

    python -m ragk8s onnx export   -> models/onnx/{embedder,reranker}/
    python -m ragk8s onnx check    -> results/onnx_parity.json

Three variants are written per model (see ``onnx_models.VARIANT_FILES``): fp32, int8
on every supported op, and int8 on MatMul weights only.

``check`` answers the question that matters for the app: with an ONNX variant, does
each evaluation question get the same context as with the PyTorch models used for the
evaluation? It compares, per strategy and method, the chunks selected within the
context budget (identical contexts, Jaccard overlap, same first chunk), the
source-page MRR with each back-end, and the query time. PyTorch and onnxruntime run
with the same number of CPU threads, so the times are comparable. PyTorch always keeps
the evaluation settings; ``rerank_max_length`` only shortens the ONNX reranker input,
so the agreement figures show how far a faster app setting drifts from the evaluation.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .onnx_models import META_FILE, ONNX_DIR, VARIANT_FILES, OnnxCrossEncoder, OnnxEmbedder
from .utils import write_json

PARITY_PATH = config.RESULTS_DIR / "onnx_parity.json"
PARITY_METHODS = ("dense", "hybrid", "hybrid_rerank")  # BM25 does not use any model


def _export_graph(model: Any, sample: dict[str, Any], out_dir: Path, output_name: str, per_token: bool) -> None:
    """torch.onnx (dynamo exporter) -> single-file model.onnx -> the two dynamic int8 variants."""
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    names = list(sample)
    axes = {n: {0: "batch", 1: "sequence"} for n in names}
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.onnx"
        torch.onnx.export(
            model.eval(),
            tuple(sample[n] for n in names),
            str(raw),
            input_names=names,
            output_names=[output_name],
            dynamic_axes={**axes, output_name: {0: "batch", 1: "sequence"} if per_token else {0: "batch"}},
            opset_version=17,
            dynamo=True,
        )
        graph = onnx.load(str(raw), load_external_data=True)
        onnx.save_model(graph, str(out_dir / "model.onnx"), save_as_external_data=False)
    fp32 = str(out_dir / VARIANT_FILES["fp32"])
    quantize_dynamic(fp32, str(out_dir / VARIANT_FILES["int8"]), weight_type=QuantType.QInt8)
    quantize_dynamic(
        fp32,
        str(out_dir / VARIANT_FILES["int8-matmul"]),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
    )


def _sample_inputs(tokenizer: Any, texts: Any) -> dict[str, Any]:
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return {k: batch[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch}


def export(out_root: Path = ONNX_DIR) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    out: dict[str, Any] = {}

    st = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    emb_dir = out_root / "embedder"
    emb_dir.mkdir(parents=True, exist_ok=True)
    pooling: list[Any] = [m for m in st if type(m).__name__ == "Pooling"]
    pooling_mode = pooling[0].get_config_dict().get("pooling_mode", "mean") if pooling else "mean"
    if pooling_mode not in ("mean", "mean_tokens"):
        raise ValueError(f"unsupported pooling mode {pooling_mode}")
    _export_graph(
        st[0].auto_model,
        _sample_inputs(st.tokenizer, ["a query", "a longer passage of text"]),
        emb_dir,
        "last_hidden_state",
        True,
    )
    st.tokenizer.save_pretrained(str(emb_dir))
    emb_meta = {
        "kind": "embedder",
        "source_model": config.EMBEDDING_MODEL,
        "max_length": int(st.max_seq_length or 256),
        "pooling": "mean",
        "normalize": any(type(m).__name__ == "Normalize" for m in st),
    }
    write_json(emb_dir / META_FILE, emb_meta)
    out["embedder"] = emb_meta

    ce = CrossEncoder(config.RERANKER_MODEL, device="cpu")
    ce_dir = out_root / "reranker"
    ce_dir.mkdir(parents=True, exist_ok=True)
    pairs = [["a query", "a passage"], ["another query", "another longer passage"]]
    sample = ce.tokenizer(*zip(*pairs, strict=True), padding=True, truncation=True, return_tensors="pt")
    _export_graph(
        ce.model,
        {k: sample[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample},
        ce_dir,
        "logits",
        False,
    )
    ce.tokenizer.save_pretrained(str(ce_dir))
    activation = type(getattr(ce, "activation_fn", None)).__name__.lower()
    ce_meta = {
        "kind": "reranker",
        "source_model": config.RERANKER_MODEL,
        "max_length": int(ce.max_length or 512),
        "activation": "sigmoid" if "sigmoid" in activation else "identity",
    }
    write_json(ce_dir / META_FILE, ce_meta)
    out["reranker"] = ce_meta

    for name in ("embedder", "reranker"):
        d = out_root / name
        out[name]["size_mb"] = {f.name: round(f.stat().st_size / 1e6, 1) for f in sorted(d.glob("*.onnx"))}
        for extra in d.glob("*.onnx.data"):
            extra.unlink()
    return out


# --- parity -------------------------------------------------------------------------


def context_agreement(reference: Sequence[Sequence[str]], candidate: Sequence[Sequence[str]]) -> dict[str, float]:
    """Agreement between two lists of contexts (chunk ids per question)."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("need two non-empty lists of the same length")
    identical = [list(a) == list(b) for a, b in zip(reference, candidate, strict=True)]
    jaccard = [
        len(set(a) & set(b)) / len(set(a) | set(b)) if (a or b) else 1.0
        for a, b in zip(reference, candidate, strict=True)
    ]
    first = [bool(a) and bool(b) and a[0] == b[0] for a, b in zip(reference, candidate, strict=True)]
    return {
        "n": len(reference),
        "identical_context": float(np.mean(identical)),
        "mean_jaccard": float(np.mean(jaccard)),
        "same_first_chunk": float(np.mean(first)),
    }


def _mrr(contexts: Sequence[Sequence[dict[str, Any]]], sources: Sequence[str | None]) -> float:
    values = []
    for hits, source in zip(contexts, sources, strict=True):
        if not source:
            continue
        ranks = [i for i, h in enumerate(hits, start=1) if h["doc_id"] == source]
        values.append(1.0 / ranks[0] if ranks else 0.0)
    return float(np.mean(values)) if values else float("nan")


def check(
    strategies: Sequence[str] = config.STRATEGIES,
    variants: Sequence[str] = tuple(VARIANT_FILES),
    threads: int = 1,
    rerank_max_length: int | None = None,
    out: Path = PARITY_PATH,
) -> dict[str, Any]:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from .pipeline import load_questions
    from .retrieval import Retriever

    torch.set_num_threads(threads)
    questions = [q for q in load_questions() if q["category"] == "factual"]
    texts = [q["question"] for q in questions]
    sources = [q.get("source_doc_id") for q in questions]
    models: dict[str, tuple[Any, Any]] = {
        "torch": (
            SentenceTransformer(config.EMBEDDING_MODEL, device="cpu"),
            CrossEncoder(config.RERANKER_MODEL, device="cpu"),
        )
    }
    for v in variants:
        models[v] = (
            OnnxEmbedder(variant=v, threads=threads),
            OnnxCrossEncoder(variant=v, threads=threads, max_length=rerank_max_length),
        )

    reference = np.asarray(models["torch"][0].encode(texts, normalize_embeddings=True))
    result: dict[str, Any] = {
        "n_questions": len(questions),
        "threads": threads,
        "variants": list(variants),
        # PyTorch keeps the evaluation settings; only the ONNX reranker is truncated.
        "onnx_rerank_max_length": models[variants[0]][1].max_length if variants else None,
        "size_mb": {
            v: {
                kind: round((ONNX_DIR / kind / VARIANT_FILES[v]).stat().st_size / 1e6, 1)
                for kind in ("embedder", "reranker")
            }
            for v in variants
        },
        "query_embedding_cosine": {},
        "configs": {},
    }
    for v in variants:
        cosine = (reference * models[v][0].encode(texts)).sum(axis=1)
        result["query_embedding_cosine"][v] = {"min": float(np.min(cosine)), "mean": float(np.mean(cosine))}

    for strategy in strategies:
        retrievers = {
            name: Retriever(strategy, embed_model=embed, cross_encoder=rerank)  # type: ignore[arg-type]
            for name, (embed, rerank) in models.items()
        }
        for method in PARITY_METHODS:
            contexts: dict[str, list[list[dict[str, Any]]]] = {}
            row: dict[str, Any] = {}
            for name, retriever in retrievers.items():
                started = time.perf_counter()
                contexts[name] = [[h.chunk for h in retriever.search(method, t)] for t in texts]  # type: ignore[arg-type]
                row[name] = {
                    "mrr": _mrr(contexts[name], sources),
                    "ms_per_query": round(1000 * (time.perf_counter() - started) / len(texts), 1),
                }
            ids = {k: [[c["chunk_id"] for c in ctx] for ctx in v] for k, v in contexts.items()}
            for v in variants:
                row[v].update(context_agreement(ids["torch"], ids[v]))
            result["configs"][f"{strategy}_{method}"] = row
    write_json(out, result)
    return result
