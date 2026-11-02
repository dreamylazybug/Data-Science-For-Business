"""Run: python -m unittest discover -s gtm_intelligence_hub/tests -v"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_engine import assess_signal, demo_signals
from signal_quality import company_match, deduplicate, filter_icp, freshness_days, quality_flags

class TestPipeline(unittest.TestCase):
    def test_demo_opportunities(self):
        matches=[assess_signal(x) for x in demo_signals()]
        self.assertTrue(any(matches))
        self.assertTrue(all(0<=m["relevance"]<=100 for group in matches for m in group))
    def test_company_boundaries(self):
        self.assertTrue(company_match("Acme Cloud Inc.","Acme Cloud announces expansion"))
        self.assertFalse(company_match("Acme Cloud","Another company expands"))
        self.assertFalse(company_match("AI","AI launches a service"))
    def test_icp_exact(self):
        self.assertTrue(filter_icp({"company":"Acme Inc"},["Acme"]))
        self.assertFalse(filter_icp({"company":"Acme Cloud"},["Acme"]))
    def test_deduplicate(self):
        x={"company":"Acme Cloud","title":"Acme Cloud expands","url":"https://example.org/a"}
        self.assertEqual(len(deduplicate([x,x])),1)
    def test_unknown_freshness(self):
        self.assertIsNone(freshness_days("invalid"))
    def test_quality_flags(self):
        s={"company":"Acme Cloud","title":"Other company raises funds","url":"https://example.org","published":""}
        self.assertIn("Company name not confirmed in headline",quality_flags(s))

if __name__=="__main__":
    unittest.main()
