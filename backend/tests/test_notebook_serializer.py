"""Notebook 断点追加与安全加载回归测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nbformat
from nbformat import v4 as nbf

from app.tools.notebook_serializer import NotebookSerializer


class NotebookSerializerTests(unittest.TestCase):
    def test_resume_appends_without_overwriting_existing_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notebook.ipynb"
            existing = nbf.new_notebook(
                cells=[
                    nbf.new_markdown_cell("old section"),
                    nbf.new_code_cell(
                        "print('old result')",
                        outputs=[
                            nbf.new_output(
                                output_type="stream",
                                name="stdout",
                                text="old result\n",
                            )
                        ],
                    ),
                ]
            )
            path.write_text(nbformat.writes(existing), encoding="utf-8")

            serializer = NotebookSerializer(work_dir=tmp)
            serializer.add_markdown_segmentation_to_notebook(
                "resumed work",
                "ques1-repair",
            )

            saved = nbformat.reads(path.read_text(encoding="utf-8"), as_version=4)
            self.assertEqual(len(saved.cells), 3)
            self.assertEqual(saved.cells[0].source, "old section")
            self.assertEqual(saved.cells[1].outputs[0].text, "old result\n")
            self.assertIn("ques1-repair", saved.cells[2].source)

    def test_invalid_existing_notebook_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notebook.ipynb"
            path.write_text("not-json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "格式损坏"):
                NotebookSerializer(work_dir=tmp)

            self.assertEqual(path.read_text(encoding="utf-8"), "not-json")


if __name__ == "__main__":
    unittest.main()
