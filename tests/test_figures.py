# tests/test_figures.py
"""Figures: crop from GROBID's box. The crop is embedded as an image, never described."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from research_assistant.shared import figures as fg
from research_assistant.shared.extract import Figure


def _fig(**kw):
    base = dict(id="fig_0", kind="figure", label="Figure 1", caption="Photoconductivity dynamics.",
                page=2, bbox=(50.0, 379.0, 542.0, 643.0), refs=[])
    base.update(kw)
    return Figure(**base)


class TestCropName(unittest.TestCase):
    def test_name_is_document_page_and_index(self):
        self.assertEqual(fg.crop_name("doi_10.1_x.pdf", _fig(page=4), 2), "doi_10.1_x.pdf_p4_f2.png")


class TestCrop(unittest.TestCase):
    def _fake_fitz(self, saved):
        class _Pix:
            def save(self, path):
                saved.append(path)

        class _Page:
            rect = types.SimpleNamespace(width=600.0, height=800.0)

            def get_pixmap(self, clip=None, dpi=72):
                saved.append(("clip", clip.x0, clip.y0, clip.x1, clip.y1))
                return _Pix()

        fitz = types.ModuleType("fitz")
        fitz.Rect = lambda x0, y0, x1, y1: types.SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1)
        fitz.open = lambda path: [_Page(), _Page(), _Page()]
        return patch.dict(sys.modules, {"fitz": fitz})

    def test_crops_with_padding_clamped_to_the_page(self):
        saved = []
        with self._fake_fitz(saved), tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "x.png")
            ok = fg.crop_figure("/x/p.pdf", _fig(bbox=(590.0, 5.0, 620.0, 100.0)), out, dpi=72)
        self.assertTrue(ok)
        self.assertEqual(saved[0], ("clip", 580.0, 0.0, 600.0, 110.0))     # 10pt pad, clamped
        self.assertEqual(saved[1], out)

    def test_no_bbox_means_no_crop(self):
        with self._fake_fitz([]):
            self.assertFalse(fg.crop_figure("/x/p.pdf", _fig(bbox=None), "/x/out.png", dpi=72))

    def test_page_out_of_range_means_no_crop(self):
        with self._fake_fitz([]):
            self.assertFalse(fg.crop_figure("/x/p.pdf", _fig(page=7), "/x/out.png", dpi=72))


class TestImageEntryConstants(unittest.TestCase):
    def test_prefix_names_the_figure(self):
        self.assertEqual(fg.FIGURE_IMAGE_PREFIX.format(label="Table 2"), "Image of Table 2: ")

    def test_images_and_old_descriptions_are_never_evidence(self):
        self.assertIn("figure_image", fg.EXCLUDED_FROM_EVIDENCE)
        self.assertIn("figure_description", fg.EXCLUDED_FROM_EVIDENCE)
        self.assertNotIn("caption", fg.EXCLUDED_FROM_EVIDENCE)
        self.assertNotIn("text_chunk", fg.EXCLUDED_FROM_EVIDENCE)
