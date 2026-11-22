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

from . import __version__, config
from .judge import JUDGE_CACHE
from .pipeline import QUESTIONS_PATH, RUNS_DIR, RunConfig, retrieval_path

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


def export(
    name: str,
    configs: Sequence[RunConfig],
    commit: str | None = None,
    out_dir: Path = EXPORT_DIR,
    run_models: Sequence[str] = (),
    include_judge_cache: bool = False,
) -> Path:
    """Archive for a Kaggle dataset.

    Generation notebook: the stored contexts of ``configs``. Judge notebook: also the
    complete runs of ``run_models`` (e.g. ``qwen3.8-27b``) for ``configs`` and, to
    resume across sessions, the judge cache built so far.
    """
    commit = commit or current_commit()
    files: list[tuple[Path, str]] = []
    included_runs: list[str] = []
    for cfg in configs:
        src = retrieval_path(cfg)
        if not src.exists():
            raise FileNotFoundError(f"{src} missing: run `python -m ragk8s retrieve` first")
        files.append((src, f"results/retrieval/{src.name}"))
        files.append((src.with_suffix(".meta.json"), f"results/retrieval/{cfg.name}.meta.json"))
        for model in run_models:
            run = RUNS_DIR / model / f"run_{cfg.name}.jsonl"
            meta = run.with_suffix(".meta.json")
            # Only complete runs are judged; the others are simply not exported yet.
            if run.exists() and meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("complete"):
                files += [(run, f"results/runs/{model}/{run.name}"), (meta, f"results/runs/{model}/{meta.name}")]
                included_runs.append(f"{model}/{cfg.name}")
    for model in run_models:
        if not any(r.startswith(model + "/") for r in included_runs):
            raise FileNotFoundError(f"no complete run of {model} for these configurations")
    if include_judge_cache and JUDGE_CACHE.exists():
        files += [(p, f".cache/judge/{p.name}") for p in sorted(JUDGE_CACHE.glob("*.json"))]
    files.append((QUESTIONS_PATH, "eval/questions.jsonl"))
    manifest = {
        "repo": "https://github.com/HamzaOuba00-1/rag-k8s-eval",
        "commit": commit,
        "configs": [c.name for c in configs],
        "runs": included_runs,
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


def import_outputs(
    path: Path, results_dir: Path = config.RESULTS_DIR, cache_root: Path = config.CACHE_DIR
) -> dict[str, Any]:
    """Copy results/** and .cache/{generations,judge}/*.json from a Kaggle output (zip or folder).

    Result files are rebuilt from the caches, so they may be replaced; a cached answer
    or score is never overwritten with different content (reported as a conflict).
    """
    with tempfile.TemporaryDirectory() as tmp_name:
        root = _source_root(path, Path(tmp_name))
        targets: list[tuple[Path, Path, bool]] = []
        for src in sorted(p for p in root.rglob("*") if p.is_file()):
            parts = src.relative_to(root).parts
            if ".cache" in parts and len(parts) > parts.index(".cache") + 2:
                sub = parts[parts.index(".cache") + 1]
                if sub in ("generations", "judge") and src.suffix == ".json":
                    targets.append((src, cache_root / sub / src.name, True))
            elif "results" in parts and parts[parts.index("results") + 1 : parts.index("results") + 2] in (
                ("runs",),
                ("judge",),
            ):
                targets.append((src, results_dir / Path(*parts[parts.index("results") + 1 :]), False))
        if not targets:
            raise FileNotFoundError(f"no results/runs, results/judge or .cache/... files found in {path}")
        copied, identical, conflicts, outputs = 0, 0, [], []
        for src, dst, immutable in targets:
            if dst.exists() and dst.read_bytes() == src.read_bytes():
                identical += 1
                continue
            if dst.exists() and immutable:
                conflicts.append(str(dst))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1
            if not immutable:
                outputs.append(dst.name)
    return {"copied": copied, "identical": identical, "conflicts": conflicts, "runs": outputs}


def default_name(configs: Sequence[RunConfig]) -> str:
    return "contexts-" + "-".join(c.name for c in configs) if len(configs) == 1 else f"contexts-{len(configs)}-configs"
