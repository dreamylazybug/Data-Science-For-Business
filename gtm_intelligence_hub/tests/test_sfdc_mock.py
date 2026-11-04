"""Salesforce mock data contracts and grain tests."""
import sys
import unittest
from collections import Counter
from datetime import date
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfdc_mock import FIELDS, generate, creation_events, conversion_summary

class TestSfdcMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=generate(date(2026,11,4))
    def test_schema(self):
        self.assertEqual(set(self.rows[0]),set(FIELDS))
        self.assertEqual(len(FIELDS),42)
    def test_three_years_and_weekly_volume(self):
        ids={r["Opportunity code"] for r in self.rows}
        self.assertGreater(len(ids),450)
        self.assertLess(len(ids),650)
        years={r["opportunity created date"][:4] for r in self.rows}
        self.assertEqual(years,{"2023","2024","2025","2026"})
    def test_line_item_grain_and_allocations(self):
        lines=[r["line item"] for r in self.rows]
        self.assertEqual(len(lines),len(set(lines)))
        for row in self.rows:
            self.assertAlmostEqual(row["TCV"],row["Upsell TCV"]+row["Renew TCV"]+row["NetNew TCV"],places=2)
    def test_creation_rollup_avoids_duplicate_opportunities(self):
        events=creation_events(self.rows)
        self.assertEqual(len(events),len({r["Opportunity code"] for r in self.rows}))
        self.assertAlmostEqual(sum(e["created_amount"] for e in events),sum(r["TCV"] for r in self.rows),places=1)
    def test_loss_reason_only_for_lost(self):
        for row in self.rows:
            if row["stage"]!="Closed Lost":
                self.assertEqual(row["primary reason lost"],"")
    def test_conversion_is_product_specific(self):
        rates=conversion_summary(self.rows)
        self.assertEqual(len(rates),4)
        self.assertTrue(all(0<=v["closed_win_rate"]<=1 for v in rates.values()))
    def test_repeatability(self):
        self.assertEqual(self.rows,generate(date(2026,11,4)))

if __name__=="__main__":
    unittest.main()
