from __future__ import annotations

from ragk8s.textclean import clean_markup


def test_layout_comments_and_layout_shortcodes_are_removed() -> None:
    text = '<!-- overview -->\nIntro.\n\n{{< include "task-tutorial-prereqs.md" >}} {{< version-check >}}\nEnd.'
    assert clean_markup(text) == "Intro.\n\nEnd."


def test_text_bearing_shortcodes_are_rendered() -> None:
    text = (
        '## {{% heading "prerequisites" %}}\n'
        'A {{< glossary_tooltip text="Pod" term_id="pod" >}} runs.\n'
        '{{< feature-state for_k8s_version="v1.25" state="stable" >}}\n'
        "{{< note >}}\nBe careful.\n{{< /note >}}\n"
        'Version {{< skew currentVersion >}} ({{< param "version" >}}).'
    )
    assert clean_markup(text) == (
        "## Before you begin\nA Pod runs.\nFEATURE STATE: Kubernetes v1.25 [stable]\nNote:\nBe careful.\n\n"
        "Version 1.36 (v1.36)."
    )


def test_hidden_comment_blocks_are_dropped() -> None:
    assert clean_markup("Keep.{{< comment >}}editor note{{< /comment >}} Keep too.") == "Keep. Keep too."


def test_go_templates_in_code_are_left_alone() -> None:
    text = "```yaml\nimage: {{ .Values.image }}\n```"
    assert clean_markup(text) == text
