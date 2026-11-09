import tempfile
import unittest
from pathlib import Path

from project import SemanticCache, make_scope


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.cache = SemanticCache(Path(self.temp.name) / "cache.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_normalization_and_expiry(self):
        self.cache.put("a", " Hello WORLD ", "answer", ttl=10, now=100)
        self.assertEqual(self.cache.lookup("a", "hello world", now=105)["kind"], "exact")
        self.assertEqual(self.cache.lookup("a", "hello world", now=110)["kind"], "miss")

    def test_context_and_mode_are_isolated(self):
        one = make_scope("v1", "demo", {})
        two = make_scope("v1", "live", {})
        three = make_scope("v2", "demo", {})
        four = make_scope("v1", "demo", {"LLM_MODEL": "other"})
        five = make_scope("v1", "demo", {"LLM_BASE_URL": "https://other.example"})
        self.assertEqual(len({one, two, three, four, five}), 5)
        self.cache.put(one, "轴承温度问题", "answer", now=100)
        self.assertEqual(self.cache.lookup(two, "轴承温度问题", now=101)["kind"], "miss")

    def test_semantic_lookup_and_threshold(self):
        self.cache.put("scope", "如何分析轴承振动趋势", "answer", now=100)
        found = self.cache.lookup("scope", "如何分析轴承振动趋势？", threshold=0.5, now=101)
        self.assertEqual(found["kind"], "semantic")
        self.assertEqual(found["answer"], "answer")

    def test_numbers_negation_and_device_ids_block_approximate_reuse(self):
        cases = [("温度80度如何处理", "温度90度如何处理"),
                 ("应该停机吗", "不应该停机吗"),
                 ("设备A如何维护", "设备B如何维护")]
        for original, changed in cases:
            with self.subTest(query=changed):
                self.cache.put(original, original, "answer", now=100)
                self.assertEqual(self.cache.lookup(original, changed, threshold=0, now=101)["kind"], "miss")


if __name__ == "__main__":
    unittest.main()
