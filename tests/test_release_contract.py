"""Release contract: config depths, weight binding, and packaging metadata."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onixbind.openfold.config import model_config  # noqa: E402
from onixbind.openfold.model.model import OnixBind  # noqa: E402

_WEIGHTS_DIR = ROOT / "src" / "weights"
_PACKAGES = sorted(_WEIGHTS_DIR.glob("*.pt"))
_MANIFESTS = sorted(_WEIGHTS_DIR.glob("*.manifest.json"))
WEIGHTS = _PACKAGES[0] if len(_PACKAGES) == 1 else None
# paired by directory, not by file name: the docs say the weight file may be
# renamed, so pinning the sidecar to the .pt's stem would fail a supported setup
MANIFEST = _MANIFESTS[0] if len(_MANIFESTS) == 1 else None


class ConfigContractTest(unittest.TestCase):
    def setUp(self):
        self.config = model_config(low_prec=True, use_deepspeed_evoformer_attention=False)

    def test_flash_depths(self):
        # the released weights use the flash configuration, not the upstream default
        self.assertEqual(self.config.backbone.pairformer_stack.no_blocks, 12)
        self.assertEqual(self.config.diffusion.diffusion_transformer.no_blocks, 6)
        self.assertEqual(self.config.diffusion.atom_attention_encoder.no_blocks, 2)
        self.assertEqual(self.config.diffusion.atom_attention_decoder.no_blocks, 2)

    def test_advanced_diffusion_conditioning(self):
        self.assertTrue(self.config.diffusion.diffusion_conditioning.advanced_conditioning)

    def test_reference_runtime_contract_values(self):
        # the reference runtime's own config contract requires these exactly;
        # they change what the attention path computes, not just its speed
        self.assertIsNone(self.config.globals["chunk_size"])
        self.assertAlmostEqual(self.config.affinity_head["eps"], 1e-4)
        self.assertAlmostEqual(self.config.affinity_head["inf"], 1e9)
        self.assertTrue(self.config.affinity_head["_mask_trans"])
        self.assertEqual(self.config.affinity_head["max_num_atoms"], 24)

    def test_deepspeed_kernel_is_disabled_where_the_reference_disables_it(self):
        import inspect
        from onixbind.openfold.model import pairformer
        for cls in (pairformer.AttentionPairBias, pairformer.MSAPairWeightedAveraging):
            source = inspect.getsource(cls.forward)
            self.assertIn(
                "use_deepspeed_evo_attention = False", source,
                f"{cls.__name__}.forward must force the DeepSpeed kernel off",
            )

    def test_features_use_the_reference_crop_and_sample_sizes(self):
        from onixbind import features
        # these are the reference deployment's values; cropping a record would
        # score a different complex than the input describes
        self.assertEqual(features.TOKEN_CROP_SIZE, 768)
        self.assertEqual(features.MSA_CROP_SIZE, 4096)
        self.assertEqual(features.MSA_SAMPLE_SIZE, 2048)
        self.assertEqual(features.ONE_HOT_CLASSES, {
            "template_aatype": 31, "msa": 32, "aatype": 31,
            "ref_element": 128, "ref_atom_name_chars": 64,
        })

    def test_vendored_af3_pipeline_is_present(self):
        from onixbind.features import runtime_root
        root = runtime_root()
        self.assertTrue((root / "alphafold3" / "data" / "data_module_affinity.py").is_file())
        self.assertTrue((root / "empty_alignment_mapping.json").is_file())

    def test_msa_sampling_matches_the_reference(self):
        embedder = self.config.backbone.msa.msa_embedder
        # the reference samples 2048 rows and pins the query row at index 0;
        # the upstream defaults were 1024 and free permutation, which drops the
        # query outright once an alignment is deeper than the cap
        self.assertEqual(embedder["msa_depth"], 2048)
        self.assertTrue(embedder["preserve_query_row"])

    def test_msa_embedder_puts_the_query_row_first(self):
        import torch
        from onixbind.openfold.model.embedders import MSAEmbedder
        # exercise the real sampler with a cap below the alignment depth, so it
        # has to choose, and check the query survives every draw
        module = MSAEmbedder(c_msa_feat=34, c_m=64, c_s_inputs=447, msa_depth=4)
        self.assertTrue(module.preserve_query_row)
        torch.manual_seed(0)
        for _ in range(20):
            rows = module.select_rows(num_alignments=9, num_select=4)
            self.assertEqual(int(rows[0]), 0)
            self.assertEqual(len(set(rows.tolist())), 4)
            self.assertTrue(all(0 <= int(r) < 9 for r in rows))




    def test_cli_defaults_match_the_reference_contract(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "onixbind_entry", ROOT / "src" / "run_onixbind.py"
        )
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        defaults = entry.build_parser().parse_args(["dummy"])
        # the weight path is resolved from src/weights/ at run time, not pinned
        # to a file name here
        self.assertIsNone(defaults.weights)
        # every one of these is a value the reference runtime pins
        self.assertEqual(defaults.msa_depth, 2048)
        self.assertEqual(defaults.recycling_iters, 10)
        self.assertEqual(defaults.sampling_steps, 200)
        self.assertEqual(defaults.num_diffusion_samples, 1)
        # AF3 records carry their own modelSeeds; only an explicit flag overrides
        self.assertIsNone(defaults.seed)
        self.assertFalse(defaults.save_features)

    def test_ensemble_is_two_equally_weighted_members(self):
        self.assertEqual(
            tuple(self.config.ensemble["members"]),
            ("head_0", "head_1"),
        )
        self.assertEqual(tuple(self.config.ensemble["weights"]), (0.5, 0.5))
        self.assertAlmostEqual(sum(self.config.ensemble["weights"]), 1.0)


def load_entry():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "onixbind_entry", ROOT / "src" / "run_onixbind.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WeightResolutionTest(unittest.TestCase):
    """The file name is not part of the contract; the package's contents are."""

    def setUp(self):
        self.entry = load_entry()

    def test_any_file_name_resolves(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            odd = Path(tmp) / "whatever-the-user-renamed-it-to.pt"
            odd.touch()
            self.assertEqual(self.entry.resolve_weights(str(tmp)), odd)
            self.assertEqual(self.entry.resolve_weights(str(odd)), odd)

    def test_empty_directory_points_at_the_docs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                self.entry.resolve_weights(tmp)
            self.assertIn("docs/installation.md", str(caught.exception))

    def test_two_packages_refuse_to_guess(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.pt").touch()
            (Path(tmp) / "b.pt").touch()
            with self.assertRaises(SystemExit) as caught:
                self.entry.resolve_weights(tmp)
            self.assertIn("--weights", str(caught.exception))

    def test_missing_file_is_an_error(self):
        with self.assertRaises(SystemExit):
            self.entry.resolve_weights("/nonexistent/weights.pt")


class NativeRuntimeTest(unittest.TestCase):
    def test_preload_reports_the_library_it_used(self):
        from onixbind.native import preload_libstdcxx
        used = preload_libstdcxx()
        # None is legitimate on a host whose default runtime is already new
        # enough; what must not happen is a silent failure to try
        self.assertTrue(used is None or used.is_file())

    def test_explicit_override_is_not_ignored(self):
        import os
        from onixbind.native import ENV_VAR, preload_libstdcxx
        previous = os.environ.get(ENV_VAR)
        os.environ[ENV_VAR] = "/nonexistent/libstdc++.so.6"
        try:
            with self.assertRaises(FileNotFoundError):
                preload_libstdcxx()
        finally:
            if previous is None:
                del os.environ[ENV_VAR]
            else:
                os.environ[ENV_VAR] = previous


@unittest.skipUnless(WEIGHTS is not None, "released weights not present")
class WeightBindingTest(unittest.TestCase):
    def test_weights_load_strictly(self):
        config = model_config(low_prec=True, use_deepspeed_evoformer_attention=False)
        model = OnixBind(config)
        package = torch.load(WEIGHTS, map_location="cpu")
        missing, unexpected = model.load_state_dict(package["state_dict"], strict=True)
        self.assertEqual(list(missing), [])
        self.assertEqual(list(unexpected), [])

    @unittest.skipUnless(MANIFEST is not None, "no single manifest to check against")
    def test_manifest_matches_the_file(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["size_bytes"], WEIGHTS.stat().st_size)
        self.assertEqual(
            manifest["sha256"], hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
        )
        self.assertEqual(manifest["tensors"], 1967)
        self.assertEqual(manifest["parameters"], 135_726_550)

    def test_weight_members_match_config(self):
        config = model_config(low_prec=True, use_deepspeed_evoformer_attention=False)
        package = torch.load(WEIGHTS, map_location="cpu")
        self.assertEqual(tuple(package["members"]), tuple(config.ensemble["members"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
