"""
Étape 1 — Ingestion du corpus.

Ce script :
1. Clone (en sparse-checkout, donc SANS télécharger tout le repo Kubernetes)
   uniquement les dossiers de doc qui nous intéressent.
2. Parcourt les fichiers Markdown, retire les métadonnées YAML ("front matter")
   qui ne sont pas du contenu utile pour le RAG.
3. Sauvegarde chaque document nettoyé dans data/raw/, accompagné d'un JSON
   de métadonnées (chemin d'origine + URL reconstruite vers la doc officielle,
   utile plus tard pour les CITATIONS).

Pourquoi le sparse-checkout ? Le repo kubernetes/website fait plusieurs
centaines de Mo avec l'historique + toutes les traductions. Le sparse-checkout
permet de ne récupérer QUE les dossiers Markdown anglais dont on a besoin,
ce qui est beaucoup plus rapide et léger.
"""

import subprocess
import shutil
import json
import os
import stat
from pathlib import Path

import frontmatter
from tqdm import tqdm

# --- Configuration ---------------------------------------------------------

REPO_URL = "https://github.com/kubernetes/website.git"
CLONE_DIR = Path("data/_k8s_website_clone")  # dossier temporaire de clone
RAW_DIR = Path("data/raw")

# Dossiers de doc qu'on veut garder (relatifs à la racine du repo cloné)
SPARSE_PATHS = [
    "content/en/docs/concepts",
    "content/en/docs/tasks",
    "content/en/docs/tutorials",
]

# Base URL pour reconstruire un lien vers la doc officielle en ligne
BASE_DOC_URL = "https://kubernetes.io/docs"


# --- Étape 1a : cloner le repo en sparse-checkout ---------------------------

def clone_sparse():
    if CLONE_DIR.exists():
        print(f"'{CLONE_DIR}' existe déjà, on saute le clonage.")
        return

    print("Clonage sparse du repo kubernetes/website (peut prendre 1-2 min)...")

    # --depth 1 : on ne prend que le dernier commit, pas tout l'historique
    # --filter=blob:none : ne télécharge pas les fichiers tant qu'on n'a
    #   pas précisé lesquels avec sparse-checkout (encore plus rapide)
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--filter=blob:none",
            "--sparse", REPO_URL, str(CLONE_DIR),
        ],
        check=True,
    )

    subprocess.run(
        ["git", "sparse-checkout", "set", *SPARSE_PATHS],
        cwd=CLONE_DIR,
        check=True,
    )

    print("Clonage terminé.")


# --- Étape 1b : nettoyer et copier les fichiers Markdown -------------------

def build_doc_url(relative_path: Path) -> str:
    """
    Reconstruit une URL vers la doc officielle à partir du chemin du fichier.
    Ex: content/en/docs/concepts/overview/what-is-kubernetes.md
        -> https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/
    """
    parts = relative_path.parts
    # on enlève "content", "en" et l'extension .md
    idx = parts.index("docs")
    doc_parts = parts[idx:]  # à partir de "docs/..."
    slug = "/".join(doc_parts)
    slug = slug.replace(".md", "").replace("_index", "")
    url = f"https://kubernetes.io/{slug}/"
    return url


def clean_and_export():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    md_files = []
    for sparse_path in SPARSE_PATHS:
        md_files.extend((CLONE_DIR / sparse_path).rglob("*.md"))

    print(f"{len(md_files)} fichiers Markdown trouvés. Nettoyage en cours...")

    manifest = []

    for i, md_path in enumerate(tqdm(md_files)):
        post = frontmatter.load(md_path)  # sépare le YAML front matter du contenu
        content = post.content.strip()

        if len(content) < 200:
            # fichiers quasi-vides (redirections, index sans contenu...) : on ignore
            continue

        relative_path = md_path.relative_to(CLONE_DIR)
        doc_id = f"doc_{i:04d}"
        out_path = RAW_DIR / f"{doc_id}.md"
        out_path.write_text(content, encoding="utf-8")

        manifest.append({
            "doc_id": doc_id,
            "title": post.get("title", relative_path.stem),
            "source_path": str(relative_path),
            "source_url": build_doc_url(relative_path),
            "n_chars": len(content),
        })

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(manifest)} documents exportés dans {RAW_DIR}/")
    print(f"Manifeste des métadonnées : {manifest_path}")
    total_chars = sum(d["n_chars"] for d in manifest)
    print(f"Volume total : ~{total_chars:,} caractères (~{total_chars // 5:,} mots)")


def _remove_readonly(func, path, exc_info):
    """
    Callback pour shutil.rmtree : sous Windows, Git marque certains fichiers
    internes (.git/objects/...) en lecture seule, ce qui fait planter la
    suppression normale. On lève ce flag puis on réessaie.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup_clone():
    """Supprime le clone git temporaire (on n'a plus besoin que de data/raw/)."""
    if CLONE_DIR.exists():
        shutil.rmtree(CLONE_DIR, onerror=_remove_readonly)
        print(f"Dossier temporaire {CLONE_DIR} supprimé.")


if __name__ == "__main__":
    clone_sparse()
    clean_and_export()
    cleanup_clone()
