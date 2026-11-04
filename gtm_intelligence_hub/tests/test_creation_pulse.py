"""Creation Pulse contract tests: no external services."""
import sys
import unittest
from datetime import date
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from creation_pulse import pulse, quarter_start, next_quarter, demo_events, separate_horizons

class TestCreationPulse(unittest.TestCase):
    def test_quarter_boundaries(self):
        self.assertEqual(quarter_start(date(2026,11,4)),date(2026,10,1))
        self.assertEqual(next_quarter(date(2026,11,4)),date(2027,1,1))
    def test_qtd_excludes_future_and_previous_quarters(self):
        events=[{"created_date":"2026-10-01","created_amount":100},
                {"created_date":"2026-11-05","created_amount":900},
                {"created_date":"2026-09-30","created_amount":200}]
        result=pulse(events,date(2026,11,4),1000)
        self.assertEqual(result["qtd"],100)
        self.assertEqual(result["opportunity_count"],1)
    def test_prior_year_aligned_by_selling_day(self):
        events=[{"created_date":"2026-10-01","created_amount":200},
                {"created_date":"2025-10-01","created_amount":100}]
        result=pulse(events,date(2026,10,1),1000)
        self.assertEqual(result["qtd"],200)
        self.assertEqual(result["prior_qtd"],100)
        self.assertEqual(result["yoy_pct"],100)
    def test_no_prior_data_has_unknown_yoy(self):
        result=pulse([{"created_date":"2026-10-01","created_amount":50}],date(2026,10,1),100)
        self.assertIsNone(result["yoy_pct"])
    def test_demo_is_repeatable(self):
        self.assertEqual(demo_events(),demo_events())
    def test_separate_horizon_contracts(self):
        result=separate_horizons()
        self.assertEqual(set(result),{"current_quarter_bookings","future_quarter_health"})
        self.assertTrue(all(x["status"]=="not_modelled" for x in result.values()))
    def test_negative_creation_rejected(self):
        with self.assertRaises(ValueError):
            pulse([{"created_date":"2026-10-01","created_amount":-5}],date(2026,10,1),100)

if __name__=="__main__":
    unittest.main()
