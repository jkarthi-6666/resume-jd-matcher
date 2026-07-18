"""Unit tests for PDF parser. Mocks PyMuPDF to avoid needing real PDFs."""
import pytest
from unittest.mock import patch, MagicMock
from src.parser import extract_text, ImageOnlyPDF, _clean


class TestClean:
    def test_ligature_fix(self):
        assert "fi" in _clean("ﬁle")
        assert "fl" in _clean("ﬂow")

    def test_collapses_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        cleaned = _clean(text)
        assert "\n\n\n" not in cleaned


class TestExtractText:
    def _mock_doc(self, text: str):
        page = MagicMock()
        page.get_text.return_value = text
        doc = MagicMock()
        doc.__iter__ = MagicMock(return_value=iter([page]))
        doc.close = MagicMock()
        return doc

    def test_extracts_text(self):
        long_text = "A" * 300 + " experienced software engineer"
        with patch("fitz.open", return_value=self._mock_doc(long_text)):
            result = extract_text(b"fake_pdf")
        assert len(result) >= 200

    def test_raises_image_only_for_short_text(self):
        with patch("fitz.open", return_value=self._mock_doc("tiny")):
            with pytest.raises(ImageOnlyPDF):
                extract_text(b"fake_pdf")
