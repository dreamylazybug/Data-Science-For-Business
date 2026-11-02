"""Unit tests for company entity resolution."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from entity_resolution import prepare_registry,resolve

class TestEntityResolution(unittest.TestCase):
    def setUp(self):
        self.registry=prepare_registry([
            {"company":"Acme Cloud Inc","domain":"acme.example","aliases":"Acme Cloud|Acme"},
            {"company":"Northstar Labs","domain":"northstar.example","aliases":"Northstar"},
        ])
    def test_exact_alias(self):
        result=resolve("Acme Cloud",self.registry)
        self.assertEqual(result["status"],"Registry match — unverified")
        self.assertEqual(result["entity"]["company"],"Acme Cloud Inc")
    def test_no_fuzzy_assignment(self):
        self.assertEqual(resolve("Acme Cloud Services",self.registry)["status"],"Unresolved")
    def test_ambiguous_name(self):
        registry=prepare_registry([{"company":"Acme"},{"company":"Acme"}])
        self.assertEqual(resolve("Acme",registry)["status"],"Ambiguous")
    def test_domain_disambiguation(self):
        registry=prepare_registry([{"company":"Acme","domain":"one.example"},{"company":"Acme","domain":"two.example"}])
        result=resolve("Acme",registry,domain_hint="two.example")
        self.assertEqual(result["entity"]["domain"],"two.example")
    def test_unknown(self):
        self.assertEqual(resolve("Unknown Inc",self.registry)["status"],"Unresolved")

if __name__=="__main__":
    unittest.main()
