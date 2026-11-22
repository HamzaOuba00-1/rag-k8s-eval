"""Turn Hugo/Docsy markup from the Kubernetes docs into plain readable text.

The raw Markdown contains ~3,500 Hugo shortcodes ({{< note >}},
{{< glossary_tooltip text="Pod" ... >}}, {{% heading "whatsnext" %}}, ...) and
HTML comments used as layout markers (<!-- overview -->). Left as-is they add
lexical noise for BM25, dilute embeddings and waste the generator's context.

``clean_markup`` approximates what the website renders: shortcodes that carry
text are replaced by that text, layout-only shortcodes are dropped. Version
placeholders are resolved with the values of the pinned upstream commit
(hugo.toml and data/releases/schedule.yaml at config.K8S_WEBSITE_COMMIT).
The frozen corpus in data/raw/ is never modified; cleaning happens at
chunking time.
"""

from __future__ import annotations

import re
from typing import Final

# Values of the website at the pinned commit.
SITE_VERSION: Final = "v1.36"
SKEW_VERSIONS: Final = {"currentVersion": "1.36", "latestVersion": "1.36", "currentPatchVersion": "1.36.2"}

# i18n/en/en.toml "<name>_heading" strings.
HEADINGS: Final = {
    "prerequisites": "Before you begin",
    "whatsnext": "What's next",
    "objectives": "Objectives",
    "cleanup": "Cleaning up",
    "seealso": "See Also",
    "synopsis": "Synopsis",
    "options": "Options",
    "examples": "Examples",
}

_ADMONITIONS: Final = {"note": "Note", "caution": "Caution", "warning": "Warning"}

_SHORTCODE_RE = re.compile(r"\{\{([<%])\s*(/?)\s*([\w/-]+)(.*?)\s*[>%]\}\}", re.S)
_ARG_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_POSITIONAL_RE = re.compile(r'^\s*"([^"]*)"|^\s*(\w+)')
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_HIDDEN_BLOCK_RE = re.compile(r"\{\{<\s*comment\s*>\}\}.*?\{\{<\s*/comment\s*>\}\}", re.S)
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _args(raw: str) -> dict[str, str]:
    return dict(_ARG_RE.findall(raw))


def _positional(raw: str) -> str:
    match = _POSITIONAL_RE.match(raw)
    if not match:
        return ""
    return match.group(1) if match.group(1) is not None else match.group(2)


def _render(match: re.Match[str]) -> str:
    closing, name, raw = match.group(2), match.group(3), match.group(4)
    if closing:
        return ""
    args = _args(raw)
    if name == "glossary_tooltip":
        return args.get("text", args.get("term_id", ""))
    if name == "heading":
        key = _positional(raw)
        return HEADINGS.get(key, key.replace("_", " ").capitalize())
    if name in _ADMONITIONS:
        return f"{_ADMONITIONS[name]}:"
    if name == "alert":
        return f"{args['title']}:" if "title" in args else ""
    if name in ("tab", "details"):
        label = args.get("name") or args.get("summary") or ""
        return f"{label}:" if label else ""
    if name == "feature-state":
        version, state = args.get("for_k8s_version", ""), args.get("state", "")
        return f"FEATURE STATE: Kubernetes {version} [{state}]".replace("  ", " ")
    if name in ("code_sample", "example", "codenew"):
        return f"(example file: {args['file']})" if "file" in args else ""
    if name == "api-reference":
        return args.get("text", args.get("page", ""))
    if name == "figure":
        return args.get("alt", args.get("caption", ""))
    if name == "table":
        return args.get("caption", "")
    if name == "glossary_definition":
        return args.get("prepend", "")
    if name == "param":
        return SITE_VERSION if _positional(raw) == "version" else ""
    if name == "skew":
        return SKEW_VERSIONS.get(_positional(raw), "")
    # Layout-only or unresolvable: include, version-check, tabs, mermaid,
    # highlight, thirdparty-content, tutorials/modules, ...
    return ""


def clean_markup(text: str) -> str:
    """Return ``text`` with HTML comments removed and Hugo shortcodes rendered as text."""
    text = _HIDDEN_BLOCK_RE.sub("", text)
    text = _COMMENT_RE.sub("", text)
    text = _SHORTCODE_RE.sub(_render, text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()
