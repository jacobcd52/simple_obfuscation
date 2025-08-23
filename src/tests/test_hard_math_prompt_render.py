from __future__ import annotations

from pathlib import Path

from src.reward.judge_reward import JUDGE_PROMPT_TEMPLATES


def test_render_hard_math_prompt_to_file(tmp_path):
    # Prepare render kwargs
    kwargs = {
        "prompt": "What is 2+2?",
        "response": "### Answer\n\\boxed{4}\n\n### Explanation\nBecause 2+2=4.",
        "hint": "",
    }

    # Render the template
    template = JUDGE_PROMPT_TEMPLATES["hard_math"]
    rendered = template.format(**kwargs)

    # Write to repo data directory for manual inspection
    out_path = Path("data") / "hard_math_prompt_rendered.txt"
    out_path.write_text(rendered, encoding="utf-8")

    # Basic sanity checks
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "{prompt}" not in content
    assert "{response}" not in content



