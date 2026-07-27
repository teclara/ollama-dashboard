import os, tempfile, unittest

import control
from tests.helpers import fixture_json

MODELS = "/home/dubthecoder/.lmstudio/models"
HUB = "/home/dubthecoder/.lmstudio/hub/models"


class TestIsInside(unittest.TestCase):
    def test_direct_child(self):
        self.assertTrue(control.is_inside("/a/b", "/a/b/c"))

    def test_the_root_itself_is_not_inside(self):
        self.assertFalse(control.is_inside("/a/b", "/a/b"))

    def test_sibling_with_shared_prefix_rejected(self):
        """/a/bad must not pass a /a/b root check via string prefix matching."""
        self.assertFalse(control.is_inside("/a/b", "/a/bad"))

    def test_traversal_rejected(self):
        self.assertFalse(control.is_inside("/a/b", "/a/b/../../etc"))

    def test_unrelated_absolute_path_rejected(self):
        self.assertFalse(control.is_inside("/a/b", "/etc/passwd"))

    def test_symlink_escaping_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "root"); outside = os.path.join(d, "outside")
            os.makedirs(root); os.makedirs(outside)
            link = os.path.join(root, "escape")
            os.symlink(outside, link)
            self.assertFalse(control.is_inside(root, link))


class TestResolveDeleteTargets(unittest.TestCase):
    def setUp(self):
        self.index = fixture_json("model_index_cache.json")

    def _resolve(self, key):
        return control.resolve_delete_targets(self.index, key, MODELS, HUB)

    def test_user_model_resolves_to_its_own_directory(self):
        # Note: the indexed id, not the model key "cyberpal2.0-20b-i1"
        r = self._resolve("mradermacher/CyberPal2.0-20B-i1-GGUF/CyberPal2.0-20B.i1-MXFP4_MOE.gguf")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual(r["targets"],
                         [f"{MODELS}/mradermacher/CyberPal2.0-20B-i1-GGUF"])

    def test_model_key_is_not_accepted_where_an_indexed_id_is_required(self):
        """Guards the join-key bug: modelKey matches the index for only some models."""
        r = self._resolve("cyberpal2.0-20b-i1")
        self.assertFalse(r["ok"])
        self.assertIn("not found", r["error"].lower())

    def test_hub_model_resolves_to_weights_and_stub(self):
        """google/gemma-4-31b's weights live under lmstudio-community, not google/."""
        r = self._resolve("google/gemma-4-31b")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual(sorted(r["targets"]), sorted([
            f"{MODELS}/lmstudio-community/gemma-4-31B-it-GGUF",
            f"{HUB}/google/gemma-4-31b",
        ]))

    def test_hub_model_never_resolves_to_its_key_as_a_path(self):
        """The naive bug: treating 'google/gemma-4-31b' as a relative path."""
        r = self._resolve("google/gemma-4-31b")
        self.assertNotIn(f"{MODELS}/google/gemma-4-31b", r["targets"])

    def test_second_hub_model(self):
        r = self._resolve("qwen/qwen3.6-27b")
        self.assertTrue(r["ok"], r.get("error"))
        self.assertIn(f"{MODELS}/lmstudio-community/Qwen3.6-27B-GGUF", r["targets"])

    def test_bundled_model_refused(self):
        r = self._resolve("nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf")
        self.assertFalse(r["ok"])
        self.assertIn("bundled", r["error"].lower())

    def test_unknown_model_refused(self):
        r = self._resolve("no/such-model")
        self.assertFalse(r["ok"])
        self.assertIn("not found", r["error"].lower())

    def test_hub_model_without_a_concrete_entry_refused_not_guessed(self):
        index = {"models": [{"indexedModelIdentifier": "orphan/model",
                             "containingDirAbsolutePath": f"{HUB}/orphan/model",
                             "sourceDirectoryType": "hub"}]}
        r = control.resolve_delete_targets(index, "orphan/model", MODELS, HUB)
        self.assertFalse(r["ok"])
        self.assertIn("could not resolve", r["error"].lower())

    def test_target_outside_permitted_roots_refused(self):
        index = {"models": [{"indexedModelIdentifier": "evil",
                             "containingDirAbsolutePath": "/etc",
                             "sourceDirectoryType": "user"}]}
        r = control.resolve_delete_targets(index, "evil", MODELS, HUB)
        self.assertFalse(r["ok"])
        self.assertIn("outside", r["error"].lower())

    def test_target_equal_to_root_refused(self):
        index = {"models": [{"indexedModelIdentifier": "root",
                             "containingDirAbsolutePath": MODELS,
                             "sourceDirectoryType": "user"}]}
        r = control.resolve_delete_targets(index, "root", MODELS, HUB)
        self.assertFalse(r["ok"])

    def test_empty_index_refused(self):
        r = control.resolve_delete_targets({"models": []}, "anything", MODELS, HUB)
        self.assertFalse(r["ok"])

    def test_malformed_index_refused_not_raised(self):
        r = control.resolve_delete_targets({}, "anything", MODELS, HUB)
        self.assertFalse(r["ok"])


class TestActualRemoval(unittest.TestCase):
    """Exercises the real rmtree path in a sandbox, so the destructive step
    is covered rather than only its path resolution."""

    def _sandbox(self, stack):
        d = tempfile.mkdtemp()
        stack.append(d)
        models = os.path.join(d, "models")
        hub = os.path.join(d, "hub", "models")
        weights = os.path.join(models, "pub", "Repo-GGUF")
        stub = os.path.join(hub, "owner", "model")
        for p in (weights, stub):
            os.makedirs(p)
            open(os.path.join(p, "weights.gguf"), "w").write("x")
        return d, models, hub, weights, stub

    def setUp(self):
        self._dirs = []
        self.addCleanup(lambda: [__import__("shutil").rmtree(p, ignore_errors=True)
                                 for p in self._dirs])

    def test_hub_delete_removes_both_dirs(self):
        d, models, hub, weights, stub = self._sandbox(self._dirs)
        index = {"models": [
            {"indexedModelIdentifier": "owner/model",
             "containingDirAbsolutePath": stub, "sourceDirectoryType": "hub"},
            {"indexedModelIdentifier": "owner/model@pub/Repo-GGUF/w.gguf",
             "containingDirAbsolutePath": weights, "sourceDirectoryType": "user"},
            {"indexedModelIdentifier": "pub/Repo-GGUF/w.gguf",
             "containingDirAbsolutePath": weights, "sourceDirectoryType": "user"},
        ]}
        r = control.resolve_delete_targets(index, "owner/model", models, hub)
        self.assertTrue(r["ok"], r.get("error"))
        import shutil
        for t in r["targets"]:
            shutil.rmtree(t)
        self.assertFalse(os.path.exists(weights))
        self.assertFalse(os.path.exists(stub))
        # The roots themselves must survive
        self.assertTrue(os.path.isdir(models))
        self.assertTrue(os.path.isdir(hub))

    def test_guard_blocks_removal_outside_the_sandbox(self):
        d, models, hub, weights, stub = self._sandbox(self._dirs)
        outside = os.path.join(d, "precious")
        os.makedirs(outside)
        index = {"models": [{"indexedModelIdentifier": "evil",
                             "containingDirAbsolutePath": outside,
                             "sourceDirectoryType": "user"}]}
        r = control.resolve_delete_targets(index, "evil", models, hub)
        self.assertFalse(r["ok"])
        self.assertTrue(os.path.isdir(outside), "guard must not have removed it")


class TestDeleteModelGuards(unittest.TestCase):
    def test_confirmation_mismatch_refused(self):
        r = control.delete_model("google/gemma-4-31b", confirm="wrong")
        self.assertFalse(r["ok"])
        self.assertIn("confirmation", r["error"].lower())

    def test_missing_confirmation_refused(self):
        r = control.delete_model("google/gemma-4-31b", confirm=None)
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
