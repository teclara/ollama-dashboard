import unittest

import control
from tests.helpers import fixture_json


class TestMapChatResponse(unittest.TestCase):
    def setUp(self):
        self.resp = fixture_json("chat_completion.json")

    def test_maps_the_real_response(self):
        r = control.map_chat_response(self.resp, "baseline", "google/gemma-4-31b", 1.2)
        self.assertTrue(r["ok"])
        self.assertEqual(r["scenario"], "baseline")
        self.assertEqual(r["model"], "google/gemma-4-31b")
        self.assertEqual(r["tool_calls"], [])

    def test_reasoning_content_becomes_thinking(self):
        r = control.map_chat_response(self.resp, "baseline", "m", 1.0)
        self.assertIn("Target output length", r["thinking"])
        self.assertEqual(r["content"], "")

    def test_stats_come_from_the_server_not_recomputed(self):
        s = control.map_chat_response(self.resp, "baseline", "m", 1.2)["stats"]
        self.assertEqual(s["prompt_tokens"], 26)
        self.assertEqual(s["completion_tokens"], 30)
        self.assertEqual(s["reasoning_tokens"], 27)
        self.assertAlmostEqual(s["tokens_per_second"], 66.0, places=0)
        self.assertAlmostEqual(s["ttft_s"], 0.136, places=3)
        self.assertAlmostEqual(s["generation_s"], 0.591, places=3)
        self.assertEqual(s["stop_reason"], "maxPredictedTokensReached")
        self.assertEqual(s["wall_seconds"], 1.2)

    def test_tool_calls_pass_through(self):
        resp = dict(self.resp)
        resp["choices"] = [{"message": {"content": "", "tool_calls": [
            {"function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}]}}]
        r = control.map_chat_response(resp, "tool_call", "m", 1.0)
        self.assertEqual(len(r["tool_calls"]), 1)
        self.assertEqual(r["tool_calls"][0]["function"]["name"], "get_weather")

    def test_missing_stats_block_does_not_crash(self):
        r = control.map_chat_response({"choices": [{"message": {"content": "hi"}}]},
                                      "baseline", "m", 0.5)
        self.assertTrue(r["ok"])
        self.assertEqual(r["content"], "hi")
        self.assertIsNone(r["stats"]["tokens_per_second"])

    def test_empty_choices_does_not_crash(self):
        r = control.map_chat_response({"choices": []}, "baseline", "m", 0.5)
        self.assertEqual(r["content"], "")


class TestScenarios(unittest.TestCase):
    def test_all_six_scenarios_present(self):
        self.assertEqual(set(control.SCENARIOS),
                         {"baseline", "reasoning", "coding", "needle29k",
                          "tool_call", "abliter"})

    def test_unknown_scenario_rejected(self):
        r = control.run_scenario("m", "does-not-exist")
        self.assertFalse(r["ok"])
        self.assertIn("unknown scenario", r["error"])

    def test_custom_requires_a_prompt(self):
        r = control.run_scenario("m", "custom", None)
        self.assertFalse(r["ok"])
        self.assertIn("custom prompt required", r["error"])


if __name__ == "__main__":
    unittest.main()
