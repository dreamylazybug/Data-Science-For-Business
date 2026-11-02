"""Tests for ICP fit and unknown attribute coverage."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from icp_scoring import fit

class TestICP(unittest.TestCase):
    def test_full_fit(self):
        x=fit({"industry":"Software","segment":"Commercial","country":"US","technologies":"Azure|Python","employees":"500"},
              {"industry":"software","segment":"commercial","country":"us","technologies":"azure","min_employees":100,"max_employees":1000})
        self.assertEqual(x["score"],100)
        self.assertEqual(x["coverage"],100)
    def test_missing_is_unknown(self):
        x=fit({"industry":"Software"},{"industry":"software","country":"US"})
        self.assertEqual(x["score"],100)
        self.assertEqual(x["coverage"],30)
        self.assertIn(("country","unknown"),x["details"])
    def test_mismatch(self):
        x=fit({"industry":"Healthcare"},{"industry":"Software"})
        self.assertEqual(x["score"],0)
    def test_no_data(self):
        x=fit({},{"industry":"Software"})
        self.assertIsNone(x["score"])
        self.assertEqual(x["coverage"],0)
    def test_employee_bounds(self):
        x=fit({"employees":"20"},{"min_employees":100,"max_employees":1000})
        self.assertEqual(x["score"],0)

if __name__=="__main__":
    unittest.main()
