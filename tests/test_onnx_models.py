from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
tokenizers = pytest.importorskip("tokenizers")

from onnx import TensorProto, helper  # noqa: E402

from ragk8s.onnx_export import context_agreement  # noqa: E402
from ragk8s.onnx_models import META_FILE, OnnxCrossEncoder, OnnxEmbedder  # noqa: E402

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "pod", "node", "service", "volume"]


def _tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    tok = Tokenizer(models.WordPiece({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tok.save(str(path / "tokenizer.json"))


def _embedding_table() -> np.ndarray:
    table = np.zeros((len(VOCAB), 3), dtype=np.float32)
    table[4] = [1, 0, 0]  # pod
    table[5] = [0, 1, 0]  # node
    table[6] = [0, 0, 1]  # service
    table[7] = [1, 1, 0]  # volume
    return table


def _model(path: Path, kind: str) -> None:
    """Tiny graphs: embedder = token embedding lookup; reranker = mean of embeddings dotted with [1, 1, 1]."""
    table = helper.make_tensor("table", TensorProto.FLOAT, [len(VOCAB), 3], _embedding_table().flatten().tolist())
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])
    nodes = [helper.make_node("Gather", ["table", "input_ids"], ["hidden"])]
    initializers = [table]
    if kind == "embedder":
        nodes.append(helper.make_node("Identity", ["hidden"], ["last_hidden_state"]))
        out = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "s", 3])
    else:
        initializers.append(helper.make_tensor("w", TensorProto.FLOAT, [3, 1], [1.0, 1.0, 1.0]))
        nodes += [
            helper.make_node("ReduceMean", ["hidden"], ["mean"], axes=[1], keepdims=0),
            helper.make_node("MatMul", ["mean", "w"], ["logits"]),
        ]
        out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", 1])
    graph = helper.make_graph(nodes, kind, [ids, mask], [out], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path / "model.onnx"))
    onnx.save(model, str(path / "model.int8.onnx"))
    _tokenizer(path)
    meta = {"kind": kind, "max_length": 16, "pooling": "mean", "normalize": True, "activation": "identity"}
    (path / META_FILE).write_text(json.dumps(meta), encoding="utf-8")


def test_embedder_mean_pools_over_real_tokens_and_normalises(tmp_path: Path) -> None:
    _model(tmp_path / "emb", "embedder")
    emb = OnnxEmbedder(tmp_path / "emb")
    vectors = emb.encode(["pod", "pod node node", "service"], batch_size=2)
    assert vectors.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(vectors[0], [1, 0, 0], atol=1e-6)
    # [CLS]/[SEP] have zero vectors; padding must not dilute the mean of a short sentence.
    np.testing.assert_allclose(vectors[1], np.array([1, 2, 0]) / np.sqrt(5), atol=1e-6)


def test_cross_encoder_scores_pairs(tmp_path: Path) -> None:
    _model(tmp_path / "ce", "reranker")
    ce = OnnxCrossEncoder(tmp_path / "ce")
    scores = ce.predict([("pod", "volume volume"), ("pod", "pod")], batch_size=1)
    assert scores.shape == (2,) and scores[0] > scores[1]


def test_unsupported_pooling(tmp_path: Path) -> None:
    _model(tmp_path / "emb", "embedder")
    meta = json.loads((tmp_path / "emb" / META_FILE).read_text())
    (tmp_path / "emb" / META_FILE).write_text(json.dumps({**meta, "pooling": "cls"}))
    with pytest.raises(ValueError):
        OnnxEmbedder(tmp_path / "emb")


def test_context_agreement() -> None:
    a = [["c1", "c2"], ["c3"], ["c4", "c5"]]
    b = [["c1", "c2"], ["c9"], ["c5", "c4"]]
    agr = context_agreement(a, b)
    assert agr["identical_context"] == pytest.approx(1 / 3)
    assert agr["same_first_chunk"] == pytest.approx(1 / 3)
    assert agr["mean_jaccard"] == pytest.approx((1 + 0 + 1) / 3)
    with pytest.raises(ValueError):
        context_agreement([], [])
