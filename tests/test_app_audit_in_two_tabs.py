"""The citation-audit report is shown by two tabs: Citation auditor and Research idea.

Both render the same shared report, and Streamlit renders every tab on every run,
so a session that has a result in each tab draws it twice. Widget keys therefore
have to be per tab, and a button in the report has to act for the tab it is in:
Re-run from the idea tab must start an idea job, not an audit-tab job that then
surfaces its result in the wrong tab.

These run the real page (tests/app_harness.py) with results planted in
st.session_state.
"""

import json
import unittest

from tests.app_harness import HAVE_STREAMLIT, run_page

ITEMS = [
    {"sentence": "Graphene conducts well [1].", "claim": "Graphene conducts well.", "cite_text": "[1]",
     "ref": {"index": 1, "title": "Electric field effect", "authors": ["K. Novoselov"], "year": 2004, "doi": "10.1/1"},
     "outcome": "judged", "judgement": "Supports", "confidence": "High", "evidence_sufficiency": "sufficient",
     "supporting_span": "Ballistic transport.", "reason": "Direct evidence.",
     "slots": {s: {"assertion": s, "verdict": "Supports"} for s in ("finding", "scope", "strength")}},
    {"sentence": "A paywalled result holds [2].", "claim": "It holds.", "cite_text": "[2]",
     "ref": {"index": 2, "xml_id": "b2", "title": "Paywalled paper", "authors": ["P. Wall"], "year": 2020, "doi": "10.1/2"},
     "outcome": "deferred_paywalled", "judgement": None, "paywalled_doi": "10.1/2"},
]
AUDIT = {
    "totals": {"total": 2, "downloaded": 1, "judged": 1, "Supports": 1, "Partially supports": 0, "Contradicts": 0,
               "Does not support": 0, "Unclear / insufficient evidence": 0, "not_downloaded": 0, "deferred_paywalled": 1},
    "results": ITEMS,
}

# Planted in the subprocess: a seed PDF that exists on disk (the Re-run button needs one) and a result in each tab.
PLANT = """
import json, os
seed = os.path.join(data, "seed.pdf")
open(seed, "w").write("%PDF-1.4")
audit = json.loads({audit!r})
def final(label):
    return {{"seed_path": seed, "seed_label": label, "query": label, "citation_audit": audit}}
{states}
"""

# Click the Re-run button of one tab while the audit job is held open, and report what was started.
CLICK = """
import threading
from unittest import mock
from research_assistant.shared import run_jobs
gate = threading.Event()
def slow_audit(*args, **kwargs):
    gate.wait(30)
    return {{"totals": {{}}, "results": []}}
buttons = [b for b in at.button if b.label.endswith("Re-run Seed Citation Audit")]
out["rerun_buttons"] = len(buttons)
with mock.patch("research_assistant.shared.seed_audit.audit_seed_citations", slow_audit):
    buttons[{which}].click().run()
    job = run_jobs.get_job(at.session_state["active_job"])
    out["job_origin"] = job.origin
    out["active_job_origin"] = at.session_state["active_job_origin"]
    out["audit_result_kept"] = "audit_result" in at.session_state
    out["idea_result_kept"] = "idea_result" in at.session_state
    gate.set()
"""


def plant(audit_tab=True, idea_tab=True):
    states = []
    if audit_tab:
        states.append('at.session_state["audit_result"] = final("audit seed"); at.session_state["audit_query"] = "audit seed"; '
                      'at.session_state["build_result"] = final("audit seed"); at.session_state["build_query"] = "audit seed"')
    if idea_tab:
        states.append('at.session_state["idea_result"] = final("idea seed"); at.session_state["idea_query"] = "idea seed"')
    return PLANT.format(audit=json.dumps(AUDIT), states="\n".join(states))


@unittest.skipUnless(HAVE_STREAMLIT, "needs streamlit")
class TestAuditReportInTwoTabs(unittest.TestCase):
    def test_a_result_in_each_tab_renders_without_a_duplicate_widget_key(self):
        page = run_page(before=plant())
        self.assertEqual(page["exceptions"], [])

    def test_each_tab_draws_its_own_copy_of_the_report(self):
        page = run_page(
            before=plant(),
            after='out["downloads"] = [e.proto.label for e in at.get("download_button")]\n'
                  'out["uploaders"] = len(at.get("file_uploader"))',
        )
        self.assertEqual(page["exceptions"], [])
        self.assertEqual(sum(label.endswith("(.md)") for label in page["downloads"]), 2)
        self.assertEqual(sum(label.endswith("(.json)") for label in page["downloads"]), 2)
        self.assertGreaterEqual(page["uploaders"], 2, "the deferred-reference uploader is in the fixture so it is covered too")

    def test_rerun_in_the_idea_tab_starts_an_idea_job(self):
        page = run_page(before=plant(audit_tab=False), after=CLICK.format(which=0))
        self.assertEqual(page["rerun_buttons"], 1)
        self.assertEqual(page["job_origin"], "idea")
        self.assertEqual(page["active_job_origin"], "idea")

    def test_rerun_in_the_audit_tab_starts_an_audit_job(self):
        page = run_page(before=plant(idea_tab=False), after=CLICK.format(which=0))
        self.assertEqual(page["job_origin"], "audit")

    def test_rerun_in_the_idea_tab_leaves_the_audit_tabs_result_alone(self):
        page = run_page(before=plant(), after=CLICK.format(which=1))  # buttons are in tab order: audit, then idea
        self.assertEqual(page["rerun_buttons"], 2)
        self.assertEqual(page["job_origin"], "idea")
        self.assertTrue(page["audit_result_kept"], "the idea tab's re-run must not clear the audit tab's result")

    def test_rerun_in_the_audit_tab_leaves_the_idea_tabs_result_alone(self):
        page = run_page(before=plant(), after=CLICK.format(which=0))
        self.assertEqual(page["job_origin"], "audit")
        self.assertTrue(page["idea_result_kept"], "the audit tab's re-run must not clear the idea tab's result")


if __name__ == "__main__":
    unittest.main()
