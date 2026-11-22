"""Round trip with Kaggle notebooks (GPU work that the local machine is too slow for).

export: package what a notebook needs into kaggle/export/<name>.zip, to upload as
        a Kaggle dataset: the stored contexts (results/retrieval/...), the gold
        questions and a manifest with the git commit of the code. The notebook
        installs ``ragk8s`` from GitHub at exactly that commit, so the prompt and
        the answer cache keys are identical to local runs.
import: copy what a notebook produced (the "Output" zip downloaded from Kaggle, or
        its extracted folder) back into the project: results/runs/** and the
        answer cache (.cache/generations/*.json). Existing files are never
        overwritten with different content.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .pipeline import GENERATION_CACHE, QUESTIONS_PATH, RUNS_DIR, RunConfig, retrieval_path

EXPORT_DIR = Path("kaggle") / "export"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def current_commit(require_pushed: bool = True) -> str:
    """HEAD commit; refuses a dirty tree or (optionally) a commit not on origin."""
    commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=no", "--", "src", "pyproject.toml"):
        raise RuntimeError("src/ or pyproject.toml has uncommitted changes: commit them first")
    if require_pushed and not _git("branch", "-r", "--contains", commit):
        raise RuntimeError(f"commit {commit[:7]} is not on GitHub yet: run `git push origin main` first")
    return commit


def export(name: str, configs: Sequence[RunConfig], commit: str | None = None, out_dir: Path = EXPORT_DIR) -> Path:
    commit = commit or current_commit()
    files: list[tuple[Path, str]] = []
    for cfg in configs:
        src = retrieval_path(cfg)
        if not src.exists():
            raise FileNotFoundError(f"{src} missing: run `python -m ragk8s retrieve` first")
        files += [
            (src, f"results/retrieval/{src.name}"),
            (src.with_suffix(".meta.json"), f"results/retrieval/{cfg.name}.meta.json"),
        ]
    files.append((QUESTIONS_PATH, "eval/questions.jsonl"))
    manifest = {
        "repo": "https://github.com/HamzaOuba00-1/rag-k8s-eval",
        "commit": commit,
        "configs": [c.name for c in configs],
        "ragk8s_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path, arcname in files:
            z.write(path, arcname)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return out


def _source_root(path: Path, tmp: Path) -> Path:
    if path.is_dir():
        return path
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    return tmp


def import_outputs(path: Path, runs_dir: Path = RUNS_DIR, cache_dir: Path = GENERATION_CACHE) -> dict[str, Any]:
    """Copy runs and cached answers from a Kaggle output (zip or folder)."""
    with tempfile.TemporaryDirectory() as tmp_name:
        root = _source_root(path, Path(tmp_name))
        runs = [p for p in root.rglob("*") if p.is_file() and "runs" in p.parts and p.suffix in (".jsonl", ".json")]
        cached = [p for p in root.rglob("*.json") if p.parent.name == "generations"]
        if not runs and not cached:
            raise FileNotFoundError(f"no results/runs/** or .cache/generations/*.json found in {path}")
        copied, identical, conflicts = 0, 0, []
        targets = [(p, runs_dir / Path(*p.parts[p.parts.index("runs") + 1 :])) for p in runs]
        targets += [(p, cache_dir / p.name) for p in cached]
        for src, dst in targets:
            if dst.exists() and dst.read_bytes() == src.read_bytes():
                identical += 1
                continue
            if dst.exists() and dst.parent == cache_dir:
                conflicts.append(str(dst))  # a cached answer must never change silently
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)  # run files are assembled from the cache: replacing them is safe
            copied += 1
    return {"copied": copied, "identical": identical, "conflicts": conflicts, "runs": [str(p.name) for p in runs]}


def default_name(configs: Sequence[RunConfig]) -> str:
    return "contexts-" + "-".join(c.name for c in configs) if len(configs) == 1 else f"contexts-{len(configs)}-configs"
