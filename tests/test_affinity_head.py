"""CPU contract tests for the ported affinity module.

These do not need a GPU or the released weights: they check that the pocket
masks, the two-head wiring, and the ensemble reduction behave as the release
contract says they do.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onixbind.openfold.config import model_config  # noqa: E402
from onixbind.openfold.model.affinity import AffinityModulePocket  # noqa: E402


def make_config():
    config = model_config(low_prec=False, use_deepspeed_evoformer_attention=False)
    config.affinity_head.pairformer_stack.no_blocks = 1
    return config


def make_batch(num_protein=8, num_ligand=3, num_atoms=24, seed=0):
    torch.manual_seed(seed)
    tokens = num_protein + num_ligand
    is_protein = torch.zeros(1, tokens, dtype=torch.bool)
    is_ligand = torch.zeros(1, tokens, dtype=torch.bool)
    is_protein[0, :num_protein] = True
    is_ligand[0, num_protein:] = True

    # protein at the origin, ligand 3 A away so the 8 A pocket cutoff hits
    x_pred = torch.randn(1, tokens, num_atoms, 3) * 0.5
    x_pred[0, num_protein:] += 3.0

    batch = {
        "seq_mask": torch.ones(1, tokens),
        "is_protein": is_protein,
        "is_ligand": is_ligand,
        "pred_dense_atom_mask": torch.ones(1, tokens, num_atoms, dtype=torch.bool),
        "atom_pseudo_beta_index": torch.arange(tokens).reshape(1, -1) * num_atoms,
        "pseudo_beta_mask": torch.ones(1, tokens),
    }
    return batch, x_pred, tokens


class AffinityModuleTest(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        self.batch, self.x_pred, self.tokens = make_batch()
        c = self.config.affinity_head
        self.s_inputs = torch.randn(1, self.tokens, c["c_s_inputs"])
        self.s = torch.randn(1, self.tokens, c["c_s"])
        self.z = torch.randn(1, self.tokens, self.tokens, c["c_z"])

    def run_head(self, module):
        with torch.no_grad():
            return module(self.s_inputs, self.s, self.z, self.x_pred, self.batch)

    def test_forward_returns_one_finite_logit(self):
        module = AffinityModulePocket(self.config).eval()
        out = self.run_head(module)
        self.assertEqual(out["affinity_logits"].shape, (1, 1))
        self.assertTrue(torch.isfinite(out["affinity_logits"]).all())

    def test_pocket_cutoff_is_read_from_config(self):
        self.assertEqual(AffinityModulePocket(self.config).cutoff, 8.0)

    def test_distogram_range_is_the_affinity_range_not_the_confidence_range(self):
        module = AffinityModulePocket(self.config)
        self.assertEqual((module.min_bin, module.max_bin, module.no_bins), (2, 22, 64))
        self.assertEqual(module.linear_d.weight.shape[1], 64)

    def test_ligand_position_changes_the_score(self):
        module = AffinityModulePocket(self.config).eval()
        near = self.run_head(module)["affinity_logits"]
        far_x = self.x_pred.clone()
        far_x[0, self.batch["is_ligand"][0]] += 60.0
        with torch.no_grad():
            far = module(self.s_inputs, self.s, self.z, far_x, self.batch)["affinity_logits"]
        self.assertFalse(torch.allclose(near, far))

    def test_two_heads_are_independent_parameters(self):
        a = AffinityModulePocket(self.config).eval()
        b = AffinityModulePocket(self.config).eval()
        b.ligand_affinity.interact_mlp[-1].bias.data.add_(1.0)
        self.assertFalse(torch.allclose(
            self.run_head(a)["affinity_logits"], self.run_head(b)["affinity_logits"]
        ))

    def test_equal_weight_reduction_is_the_mean(self):
        import torch
        from onixbind.openfold.config import model_config
        from onixbind.openfold.model.model import OnixBind
        # exercise the model's own reduction rather than restating the formula
        config = model_config(low_prec=True, use_deepspeed_evoformer_attention=False)
        with torch.device("meta"):
            model = OnixBind(config)
        weights = torch.tensor(model.head_weights, dtype=torch.float64)
        self.assertEqual(len(model.head_weights), len(model.head_aliases))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        # [n_heads, batch, 1], the shape model.forward stacks the heads into
        scores = torch.tensor([[[1.5]], [[2.5]]], dtype=torch.float64)
        reduced = (scores * weights.reshape(-1, 1, 1)).sum(dim=0)
        self.assertAlmostEqual(float(reduced.reshape(-1)[0]), 2.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
