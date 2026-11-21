"""
Étape 7b — Évaluation RAGAS d'un run produit par run_pipeline.py.

Sur les questions "factual" : calcule faithfulness, answer_relevancy,
context_precision, context_recall via RAGAS (le LLM local Ollama sert de
"juge" pour ces métriques).

Sur les questions "impossible" : RAGAS n'a pas de sens ici (pas de vraie
réponse à comparer). On calcule à la place un taux de refus correct :
le système a-t-il dit "je ne sais pas" plutôt que d'halluciner ?

Usage :
  python src/evaluate_ragas.py run_structure_hybrid_rerank
  (le nom correspond au fichier results/<nom>.jsonl généré par run_pipeline.py)
"""

import json
import sys
import types
from pathlib import Path

# --- Correctif de compatibilité --------------------------------------------
# ragas importe en dur `langchain_community.chat_models.vertexai`, un module
# qui a été supprimé dans les versions récentes de langchain-community
# (déplacé vers le package séparé langchain-google-vertexai). Comme ce
# projet n'utilise jamais Google VertexAI, on simule ce module avec un stub
# vide AVANT d'importer ragas, pour éviter le crash à l'import.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _fake_module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # stub : jamais réellement instancié dans ce projet
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "Stub de compatibilité ragas : VertexAI n'est pas utilisé ici."
            )

    _fake_module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_module
# ----------------------------------------------------------------------------

from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

RESULTS_DIR = Path("results")
# Le juge 3B produisait des sorties JSON mal formées sur certaines questions
# (schéma non respecté -> crash de parsing). Le 7B est plus lent (tourne en
# partie sur CPU faute de VRAM) mais nettement plus fiable pour respecter le
# format de sortie structuré attendu par RAGAS. On accepte la lenteur plutôt
# que des évaluations qui plantent ou des scores non fiables.
JUDGE_MODEL = "qwen2.5:7b-instruct"

REFUSAL_MARKERS = [
    "ne trouve pas cette information",
    "je ne sais pas",
    "not answerable",
    "cannot find this information",
    "n'est pas mentionné",
]


def load_run(run_name):
    path = RESULTS_DIR / f"{run_name}.jsonl"
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def evaluate_factual(records):
    factual = [r for r in records if r["category"] == "factual"]
    if not factual:
        print("Aucune question 'factual' dans ce run.")
        return None

    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=JUDGE_MODEL, temperature=0, num_ctx=8192)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    run_config = RunConfig(timeout=300, max_workers=1)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"Évaluation RAGAS sur {len(factual)} questions factuelles, UNE PAR UNE "
          f"(plus lent qu'un batch, mais une question qui casse RAGAS ne fait pas "
          f"perdre les autres)...")

    rows = []
    n_failed = 0
    for i, r in enumerate(factual):
        single = Dataset.from_list([{
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }])
        try:
            result = evaluate(
                single, metrics=metrics, llm=judge_llm,
                embeddings=judge_embeddings, run_config=run_config,
                raise_exceptions=True,  # on la catch nous-mêmes juste après
            )
            row = result.to_pandas().iloc[0].to_dict()
        except Exception as e:
            print(f"  [{i + 1}/{len(factual)}] ÉCHEC (question ignorée pour les métriques) : "
                  f"{type(e).__name__}")
            row = {col: float("nan") for col in metric_cols}
            n_failed += 1
        else:
            print(f"  [{i + 1}/{len(factual)}] OK")

        row["question"] = r["question"]
        rows.append(row)

    if n_failed:
        print(f"\n{n_failed}/{len(factual)} questions n'ont pas pu être évaluées "
              f"(échec de parsing du juge local) — à mentionner comme limite dans le rapport.")

    return pd.DataFrame(rows)


def evaluate_impossible(records):
    impossible = [r for r in records if r["category"] == "impossible"]
    if not impossible:
        return None

    correct_refusals = 0
    for r in impossible:
        answer_lower = r["answer"].lower()
        if any(marker in answer_lower for marker in REFUSAL_MARKERS):
            correct_refusals += 1

    rate = correct_refusals / len(impossible)
    print(f"\nTaux de refus correct sur questions 'impossible' : "
          f"{correct_refusals}/{len(impossible)} ({rate:.1%})")
    return rate


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "run_structure_hybrid_rerank"
    records = load_run(run_name)
    print(f"{len(records)} enregistrements chargés depuis {run_name}.jsonl")

    df = evaluate_factual(records)

    summary_data = {
        "run_name": run_name,
        "n_factual": len([r for r in records if r["category"] == "factual"]),
        "n_impossible": len([r for r in records if r["category"] == "impossible"]),
    }

    if df is not None:
        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        means = df[metric_cols].mean()  # .mean() ignore les NaN par défaut (skipna=True)
        n_nan = df[metric_cols].isna().sum()

        print("\n=== Scores moyens (questions factuelles) ===")
        print(means.to_string())
        print("\n=== Questions non évaluables (échec de parsing du juge) ===")
        print(n_nan.to_string())

        summary_data.update(means.to_dict())
        summary_data["n_parsing_failures"] = n_nan.to_dict()

        detail_path = RESULTS_DIR / f"scores_{run_name}.csv"
        df.to_csv(detail_path, index=False)
        print(f"\nDétail par question -> {detail_path}")

    refusal_rate = evaluate_impossible(records)
    summary_data["refusal_rate"] = refusal_rate

    summary_path = RESULTS_DIR / f"summary_{run_name}.json"
    summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRésumé global -> {summary_path}")


if __name__ == "__main__":
    main()