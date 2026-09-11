# tests/test_figures.py
"""Figures: crop from GROBID's box, describe with the right context.

The v1 path gave the VLM whatever text block sat below a Detectron2 crop —
30% of the time not the caption. Here the context is the caption plus the
sentences that cite the figure, and the output is marked as generated.
"""

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


class TestContext(unittest.TestCase):
    def test_caption_plus_citing_sentences(self):
        f = _fig(refs=["We find a 10% increase (Figure 1b).", "Figure 1c shows the decay."])
        ctx = fg.figure_context(f)
        self.assertTrue(ctx.startswith("Caption: Figure 1: Photoconductivity dynamics."))
        self.assertIn("Referenced in the text:", ctx)
        self.assertIn("- We find a 10% increase (Figure 1b).", ctx)

    def test_refs_are_capped(self):
        f = _fig(refs=[f"Ref {i}." for i in range(10)])
        self.assertEqual(fg.figure_context(f, max_refs=3).count("\n- "), 3)

    def test_no_refs(self):
        self.assertNotIn("Referenced", fg.figure_context(_fig()))


class TestCropName(unittest.TestCase):
    def test_matches_v1_naming(self):
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


class TestDescribe(unittest.TestCase):
    def test_routes_through_shared_llm_with_the_image_and_temperature_zero(self):
        seen = {}

        def fake_chat(messages, images=None, temperature=None, **kw):
            seen.update(messages=messages, images=images, temperature=temperature)
            return types.SimpleNamespace(content="Two peaks at ±1.")

        import research_assistant.shared.llm as llm_mod
        with patch.object(llm_mod, "chat", fake_chat):
            out = fg.describe_crop("/x/crop.png", _fig(), "Caption: ...")
        self.assertEqual(out, "Two peaks at ±1.")
        self.assertEqual(seen["images"], ["/x/crop.png"])
        self.assertEqual(seen["temperature"], 0.0)
        self.assertIn("Caption: ...", seen["messages"][-1]["content"])
        self.assertIn("figure", seen["messages"][-1]["content"].lower())

    def test_prefix_names_the_figure(self):
        self.assertEqual(fg.DESCRIPTION_PREFIX.format(label="Table 2"),
                         "Auto-generated description of Table 2 — verify values against the figure: ")
