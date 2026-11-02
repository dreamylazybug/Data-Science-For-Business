"""Candidate extraction tests: no live network dependency."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prospect_discovery import extract_candidate, TOPICS

class TestDiscovery(unittest.TestCase):
    def test_candidate_from_expansion(self):
        self.assertEqual(extract_candidate("Northstar Systems announces new data center - Publisher"),"Northstar Systems")
    def test_candidate_from_funding(self):
        self.assertEqual(extract_candidate("Vertex Labs raises funding"),"Vertex Labs")
    def test_reject_editorial_headline(self):
        self.assertIsNone(extract_candidate("How AI is changing the economy"))
    def test_reject_no_company(self):
        self.assertIsNone(extract_candidate("A new era for cloud security"))
    def test_topics_available(self):
        self.assertGreaterEqual(len(TOPICS),3)

if __name__=="__main__":
    unittest.main()
