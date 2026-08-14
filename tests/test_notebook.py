from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_notebook_runs_the_real_pipeline_before_the_offline_control() -> None:
    notebook = json.loads((PROJECT_ROOT / "evals.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
    )

    assert 'os.environ.get("ANTHROPIC_API_KEY")' in source
    assert "from inspect_ai import eval_async" in source
    assert "property_content_eval()" in source
    assert 'log_dir=str(NOTEBOOK_LOG_DIR)' in source
    assert "score.metadata[\"generation_result\"][\"content\"]" in source
    assert "semantic_grounding" in source
    assert source.index("await eval_async") < source.index("await build_offline_summary")
    assert "read_eval_log" not in source


def test_notebook_is_committed_without_execution_outputs() -> None:
    notebook = json.loads((PROJECT_ROOT / "evals.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
