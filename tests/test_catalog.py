import unittest

import control
from tests.helpers import fixture


class TestParseCatalogHtml(unittest.TestCase):
    def setUp(self):
        self.items = control.parse_catalog_html(fixture("catalog_page.html"))

    def test_finds_every_card(self):
        self.assertEqual(len(self.items), 8)

    def test_extracts_slug_and_name(self):
        by_slug = {i["slug"]: i for i in self.items}
        self.assertEqual(by_slug["qwen3.6"]["name"], "Qwen3.6")
        self.assertEqual(by_slug["gemma-4"]["name"], "Gemma 4")
        self.assertEqual(by_slug["lfm2-24b-a2b"]["name"], "LFM2-24B-A2B")

    def test_sizes_are_deduped_preserving_order(self):
        """Size badges render twice for responsive layouts; each must appear once."""
        by_slug = {i["slug"]: i for i in self.items}
        self.assertEqual(by_slug["qwen3.6"]["sizes"], ["27B", "35B"])
        self.assertEqual(by_slug["gemma-4"]["sizes"],
                         ["5.1B", "7.9B", "12B", "26B", "31B"])
        self.assertEqual(by_slug["qwen3.5"]["sizes"],
                         ["2B", "4B", "9B", "27B", "35B"])

    def test_slug_is_usable_as_a_download_name(self):
        for i in self.items:
            self.assertNotIn("/", i["slug"])
            self.assertNotIn('"', i["slug"])

    def test_unrecognized_markup_returns_empty_not_raises(self):
        self.assertEqual(control.parse_catalog_html("<html><body>nope</body></html>"), [])
        self.assertEqual(control.parse_catalog_html(""), [])

    def test_card_without_a_name_is_skipped_not_fatal(self):
        html = 'href="/models/ghost"><div class="other">x</div>'
        self.assertEqual(control.parse_catalog_html(html), [])


if __name__ == "__main__":
    unittest.main()
