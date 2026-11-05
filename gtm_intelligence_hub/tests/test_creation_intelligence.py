"""Creation Intelligence smoke/contract tests."""
import sys
import unittest
from datetime import date
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfdc_mock import generate
from buyer_seller_mock import people,buyer_profiles,engagement_events,performance
from creation_intelligence import horizon_models,inspect

class TestCreationIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.as_of=date(2026,11,4)
        cls.rows=generate(cls.as_of)
        cls.sellers,_=people()
        cls.contacts=buyer_profiles(cls.rows)
        cls.engagements=engagement_events(cls.rows,cls.contacts,cls.as_of)
    def test_horizon_outputs(self):
        h=horizon_models(self.rows,self.as_of)
        self.assertGreaterEqual(h["current"]["booked_qtd"],0)
        self.assertEqual(len(h["future"]["expected_by_quarter"]),3)
        self.assertTrue(all(x>=0 for x in h["future"]["gaps"]))
    def test_inspection_created_qtd_only(self):
        items=inspect(self.rows,self.contacts,self.engagements,self.sellers,
                      performance(self.rows,self.sellers),self.as_of)
        self.assertTrue(items)
        created={r["Opportunity code"]:r["opportunity created date"] for r in self.rows}
        self.assertTrue(all("2026-10-01"<=created[x["opportunity_code"]]<="2026-11-04" for x in items))
        self.assertTrue(all(x["actions"] and x["gaps"] for x in items))

if __name__=="__main__":
    unittest.main()
