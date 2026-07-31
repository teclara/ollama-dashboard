import unittest
from unittest import mock

import control
from tests.helpers import fixture


class TestParseCatalogHtml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = control.parse_catalog_html(fixture("ollama_library.html"))

    def test_finds_many_models(self):
        self.assertGreater(len(self.rows), 100)

    def test_slugs_are_unique(self):
        slugs = [r["slug"] for r in self.rows]
        self.assertEqual(len(slugs), len(set(slugs)))

    def test_extracts_name_and_description(self):
        by_slug = {r["slug"]: r for r in self.rows}
        self.assertIn("llama3.1", by_slug)
        row = by_slug["llama3.1"]
        self.assertEqual(row["name"], "llama3.1")
        self.assertIn("Llama 3.1", row["description"])

    def test_extracts_parameter_sizes(self):
        by_slug = {r["slug"]: r for r in self.rows}
        self.assertEqual(by_slug["llama3.1"]["sizes"], ["8b", "70b", "405b"])

    def test_extracts_capability_badges(self):
        by_slug = {r["slug"]: r for r in self.rows}
        self.assertIn("tools", by_slug["llama3.1"]["capabilities"])

    def test_capabilities_and_sizes_do_not_mix(self):
        # Size badges and capability badges share the same markup slot; only a
        # shape test separates them.
        for r in self.rows:
            for cap in r["capabilities"]:
                self.assertFalse(cap.rstrip("bmBM").replace(".", "").isdigit()
                                 and cap[-1].lower() in "bm",
                                 f"{r['slug']}: size {cap!r} leaked into capabilities")
            for size in r["sizes"]:
                self.assertNotIn(size, ("tools", "vision", "thinking", "embedding"))

    def test_every_row_has_a_description(self):
        missing = [r["slug"] for r in self.rows if not r["description"]]
        self.assertEqual(missing, [])

    def test_embedding_models_have_no_size_badges(self):
        by_slug = {r["slug"]: r for r in self.rows}
        if "nomic-embed-text" in by_slug:
            self.assertIn("embedding", by_slug["nomic-embed-text"]["capabilities"])

    def test_empty_html(self):
        self.assertEqual(control.parse_catalog_html(""), [])
        self.assertEqual(control.parse_catalog_html(None), [])

    def test_html_without_model_cards(self):
        self.assertEqual(control.parse_catalog_html('<a href="/library/x">no card</a>'), [])


class TestCatalogCache(unittest.TestCase):
    def setUp(self):
        control._CAT_CACHE.update({"data": [], "fetched": 0, "error": None})

    def tearDown(self):
        control._CAT_CACHE.update({"data": [], "fetched": 0, "error": None})

    def test_serves_from_cache_within_ttl(self):
        html = fixture("ollama_library.html")
        with mock.patch("control._fetch_catalog_html", return_value=html) as fetch:
            control.catalog()
            control.catalog()
        self.assertEqual(fetch.call_count, 1)

    def test_force_bypasses_the_cache(self):
        html = fixture("ollama_library.html")
        with mock.patch("control._fetch_catalog_html", return_value=html) as fetch:
            control.catalog()
            control.catalog(force=True)
        self.assertEqual(fetch.call_count, 2)

    def test_fetch_failure_keeps_the_stale_copy_and_reports(self):
        with mock.patch("control._fetch_catalog_html",
                        return_value=fixture("ollama_library.html")):
            control.catalog()
        with mock.patch("control._fetch_catalog_html",
                        side_effect=Exception("network down")):
            out = control.catalog(force=True)
        self.assertIn("network down", out["error"])
        self.assertTrue(out["data"], "a failed refresh must not blank the catalog")

    def test_first_fetch_failure_returns_empty_with_an_error(self):
        with mock.patch("control._fetch_catalog_html",
                        side_effect=Exception("network down")):
            out = control.catalog()
        self.assertEqual(out["data"], [])
        self.assertIn("network down", out["error"])


if __name__ == "__main__":
    unittest.main()
