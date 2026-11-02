"""Offline contract tests for RSS and local Ollama integrations."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_engine import fetch_news
from prospect_discovery import fetch_topic
from local_llm import analyze_local

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Northstar Systems announces cloud expansion - Publisher</title>
<link>https://news.google.com/articles/example</link>
<pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate>
<source>Publisher</source><description>Cloud investment</description></item>
</channel></rss>"""

class Response(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()
    @property
    def headers(self):
        return {"Content-Length": str(len(self.getvalue()))}

class TestIntegrations(unittest.TestCase):
    @patch("pipeline_engine.urlopen", side_effect=lambda *a, **k: Response(RSS))
    def test_company_feed(self, mock_url):
        rows = fetch_news("Northstar Systems")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "Publisher")
        self.assertTrue(rows[0]["url"].startswith("https://"))
    @patch("prospect_discovery.urlopen", side_effect=lambda *a, **k: Response(RSS))
    def test_topic_feed(self, mock_url):
        rows = fetch_topic("Cloud infrastructure expansion")
        self.assertEqual(rows[0]["company"], "Northstar Systems")
        self.assertTrue(rows[0]["discovery"])
    @patch("local_llm.urlopen")
    def test_valid_local_llm(self, mock_url):
        answer = {"event_type":"Expansion","product":"Cloud Security",
                  "evidence_quote":"cloud expansion",
                  "opportunity_hypothesis":"Review cloud security requirements.",
                  "validation_step":"Verify announcement."}
        mock_url.return_value = Response(json.dumps({"response":json.dumps(answer)}).encode())
        result = analyze_local({"title":"Northstar announces cloud expansion","summary":""}, ["Cloud Security"])
        self.assertEqual(result["product"], "Cloud Security")
    @patch("local_llm.urlopen")
    def test_hallucinated_quote_rejected(self, mock_url):
        answer = {"event_type":"Expansion","product":"Cloud Security",
                  "evidence_quote":"Guaranteed $5 million purchase",
                  "opportunity_hypothesis":"Confirmed order","validation_step":"Contact buyer"}
        mock_url.return_value = Response(json.dumps({"response":json.dumps(answer)}).encode())
        with self.assertRaises(ValueError):
            analyze_local({"title":"Northstar announces cloud expansion","summary":""}, ["Cloud Security"])
    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            analyze_local({"title":"Example","summary":""}, ["Cloud Security"],endpoint="https://remote.example")

if __name__ == "__main__":
    unittest.main()
