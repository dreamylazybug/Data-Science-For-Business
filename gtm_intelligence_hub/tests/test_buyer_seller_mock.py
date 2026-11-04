"""Buyer, seller, relationship and engagement synthetic data contract tests."""
import sys
import unittest
from datetime import date
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfdc_mock import generate
from buyer_seller_mock import people,buyer_profiles,buyer_relationships,engagement_events,performance

class TestBuyerSeller(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=generate(date(2026,11,4))
        cls.sellers,cls.specialists=people()
        cls.contacts=buyer_profiles(cls.rows)
    def test_seller_and_specialist_keys(self):
        sellers={s["seller_id"] for s in self.sellers}
        specialists={s["specialist_id"] for s in self.specialists}
        self.assertEqual(len(sellers),len(self.sellers))
        for row in self.rows:
            self.assertIn(row["seller_id"],sellers)
            if row["specialist_id"]:
                self.assertIn(row["specialist_id"],specialists)
    def test_certifications(self):
        for p in self.sellers+self.specialists:
            self.assertEqual(p["certification_count"],len(p["certifications"].split(" | ")))
    def test_buyer_relationship_keys(self):
        contacts={c["contact_id"] for c in self.contacts}
        for edge in buyer_relationships(self.contacts):
            self.assertIn(edge["source_contact_id"],contacts)
            self.assertIn(edge["target_contact_id"],contacts)
    def test_engagement_references(self):
        events=engagement_events(self.rows,self.contacts,date(2026,11,4))
        contacts={c["contact_id"] for c in self.contacts}
        opps={r["Opportunity code"] for r in self.rows}
        for e in events:
            self.assertIn(e["contact_id"],contacts)
            self.assertIn(e["opportunity_code"],opps)
            self.assertLessEqual(e["engagement_date"],"2026-11-04")
    def test_product_performance(self):
        rows=performance(self.rows,self.sellers)
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["won"]+r["lost"],r["closed_opportunities"])
            self.assertGreaterEqual(r["win_rate"],0)
            self.assertLessEqual(r["win_rate"],1)

if __name__=="__main__":
    unittest.main()
