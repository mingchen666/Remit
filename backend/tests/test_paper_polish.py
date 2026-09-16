from pathlib import Path

from app.utils.paper_polish import enforce_em_dash_budget, polish_markdown


def test_enforce_em_dash_budget_normalizes_excess() -> None:
    assert enforce_em_dash_budget("甲—乙—丙—丁") == "甲—乙—丙，丁"


def test_polish_markdown_applies_em_dash_budget(tmp_path: Path) -> None:
    polished = polish_markdown("# 标题\n\n甲—乙—丙—丁", tmp_path)
    assert polished.count("—") == 2
    assert "丙，丁" in polished
