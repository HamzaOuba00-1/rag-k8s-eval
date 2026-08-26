"""
Étape 8 — Agrégation et comparaison des résultats.

Rassemble tous les fichiers results/summary_*.json (produits par
evaluate_ragas.py) en un seul tableau comparatif, trié par fidélité
(faithfulness) décroissante — la métrique la plus importante pour ce
projet (mesure directe de l'hallucination).

Gère les runs en mode "fast" (2 métriques) et "full" (4 métriques) sans
planter : les colonnes manquantes apparaissent simplement en vide.

Produit :
  results/comparison.csv  (pour retravailler les chiffres, Excel...)
  results/comparison.md   (tableau Markdown prêt à coller dans le rapport)

Usage :
  python src/compare_results.py
"""

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_summaries():
    summaries = []
    for path in sorted(RESULTS_DIR.glob("summary_run_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(data)
    return summaries


def parse_run_name(run_name):
    """
    run_structure_hybrid_rerank -> ("structure", "hybrid_rerank")
    run_structure_bm25          -> ("structure", "bm25")
    """
    name = run_name.replace("run_", "", 1)
    for strategy in ("fixed", "structure", "semantic"):
        if name.startswith(strategy + "_"):
            method = name[len(strategy) + 1:]
            return strategy, method
    return name, ""


def main():
    summaries = load_summaries()
    if not summaries:
        print("Aucun fichier results/summary_run_*.json trouvé. "
              "Lance d'abord evaluate_ragas.py sur au moins un run.")
        return

    rows = []
    for s in summaries:
        strategy, method = parse_run_name(s["run_name"])
        row = {
            "chunking": strategy,
            "retrieval": method,
            "mode": s.get("mode", "?"),
            "n_factual": s.get("n_factual"),
            "n_impossible": s.get("n_impossible"),
        }
        for col in METRIC_COLS:
            row[col] = s.get(col)
        row["refusal_rate"] = s.get("refusal_rate")
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(by="faithfulness", ascending=False, na_position="last")

    # Arrondi pour la lisibilité (les CSV/JSON bruts gardent la pleine précision)
    display_cols = ["chunking", "retrieval", "mode"] + METRIC_COLS + ["refusal_rate"]
    df_display = df[display_cols].copy()
    for col in METRIC_COLS + ["refusal_rate"]:
        df_display[col] = df_display[col].apply(lambda x: round(x, 3) if pd.notna(x) else "")

    print("\n=== Tableau comparatif (trié par fidélité décroissante) ===\n")
    print(df_display.to_string(index=False))

    csv_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV complet -> {csv_path}")

    md_path = RESULTS_DIR / "comparison.md"
    md_path.write_text(df_display.to_markdown(index=False), encoding="utf-8")
    print(f"Tableau Markdown (pour le rapport) -> {md_path}")

    if df["faithfulness"].notna().any():
        best = df.iloc[0]
        print(f"\n>>> Meilleure combinaison (fidélité) : "
              f"chunking={best['chunking']}, retrieval={best['retrieval']} "
              f"(faithfulness={best['faithfulness']:.3f})")
        print(">>> C'est celle-ci qu'il faudra réévaluer en mode --full "
              "sur un échantillon plus large pour le chiffre final.")


if __name__ == "__main__":
    main()