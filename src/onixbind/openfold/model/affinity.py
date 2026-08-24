# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright 2024 IntelliGen-AI and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pocket-conditioned ligand affinity module.

The trunk and diffusion path are run once per record; this module consumes the
resulting ``s_inputs, s, z, x_pred`` and predicts one pX value.  Unlike the
trunk's auxiliary heads it needs the predicted coordinates, because the pocket
is defined by an 8 A protein-ligand contact cutoff on ``x_pred``.

Every class here is carried over unchanged from the checkpoint's training-time
definition; the parameter names and shapes are what the released weights bind
to, so edits here silently change what the weights mean.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Optional, Tuple

import einops
import torch
import torch.nn as nn

from onixbind.openfold.model.pairformer import PairformerStack, Transition
from onixbind.openfold.model.primitives import Linear


class PairwiseConditioning(nn.Module):
    """Algorithm 21."""

    def __init__(
        self,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            nn.Linear(token_z + dim_token_rel_pos_feats, token_z,bias=False),
        )

        self.transitions = Transition(
                token_z, num_transitions
            )

    def forward(
        self,
        z_trunk,  # Float['b n n tz'],
        pair_mask,
        token_rel_pos_feats,  # Float['b n n 3'],
    ):  # -> Float['b n n tz']:
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)
        z = self.transitions(z, mask=pair_mask)
        # for transition in self.transitions:
        #     z = transition(z) + z

        return z


class BaseAffinityModule(nn.Module):
    """Base class for AffinityModule family.

    Extracts the shared core: config reading, linear projections,
    distogram computation, pairwise conditioning, and pairformer stack.
    Subclasses provide their own pairwise_conditioner, head, pair_mask logic,
    and head calling convention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config.affinity_head
        self.globals = config.globals
        self.c_s_inputs = self.config["c_s_inputs"]
        self.c_s = self.config["c_s"]
        self.c_z = self.config["c_z"]
        self.inf = self.config["inf"]
        self.eps = self.config["eps"]
        self.config_pairformer_stack = self.config["pairformer_stack"]
        self.max_num_atoms = self.config["max_num_atoms"]

        self.min_bin = self.config["min_bin"]
        self.max_bin = self.config["max_bin"]
        self.no_bins = self.config["no_bins"]

        self.linear_s_inputs_row = Linear(self.c_s_inputs, self.c_z, bias=False)
        self.linear_s_inputs_col = Linear(self.c_s_inputs, self.c_z, bias=False)
        self.linear_d = Linear(self.no_bins, self.c_z, bias=False)
        self.pairformer_stack = PairformerStack(**self.config_pairformer_stack)

    def _compute_structure_features(
        self,
        s_inputs,
        s,
        z,
        x_pred,
        batch,
        pair_mask,
        single_mask=None,
    ):
        """Shared computation: s_inputs projection, representative atom extraction,
        distance binning, pairwise conditioning, and pairformer stack.

        Returns:
            s: updated single representation
            z: updated pair representation
            diffusion_batch_size: int
        """
        batch_size = s_inputs.shape[0]
        coordinate_batch_size = x_pred.shape[0]
        if (
            batch_size <= 0
            or coordinate_batch_size <= 0
            or coordinate_batch_size % batch_size != 0
        ):
            raise ValueError(
                "x_pred batch dimension must be positive and divisible by "
                "s_inputs batch dimension"
            )
        diffusion_batch_size = coordinate_batch_size // batch_size
        if batch_size > 1 and diffusion_batch_size > 1:
            raise ValueError(
                "batch size and diffusion batch size cannot both exceed one"
            )
        if s.shape[0] != batch_size or z.shape[0] != batch_size:
            raise ValueError(
                "s and z batch dimensions must match s_inputs before "
                "diffusion-sample expansion"
            )

        if single_mask is None:
            single_mask = batch["seq_mask"]
        allowed_mask_batches = {coordinate_batch_size}
        if batch_size == 1:
            allowed_mask_batches.add(1)
        if (
            single_mask.ndim != 2
            or single_mask.shape[0] not in allowed_mask_batches
        ):
            raise ValueError(
                "single_mask must have shape [B, L] or [B*K, L] for "
                "diffusion-sample expansion"
            )
        if pair_mask.ndim != 3 or pair_mask.shape[0] not in allowed_mask_batches:
            raise ValueError(
                "pair_mask must have shape [B, L, L] or [B*K, L, L] for "
                "diffusion-sample expansion"
            )
        atom_pseudo_beta_index = batch["atom_pseudo_beta_index"]
        pseudo_beta_mask = batch["pseudo_beta_mask"]

        # s_inputs projection into pair representation
        s_inputs_row = self.linear_s_inputs_row(s_inputs)
        s_inputs_col = self.linear_s_inputs_col(s_inputs)
        z = z + s_inputs_row.unsqueeze(-2) + s_inputs_col.unsqueeze(-3)
        if diffusion_batch_size > 1:
            z = einops.repeat(
                z, "b ... -> (b n) ...", n=diffusion_batch_size
            )

        # representative atom extraction
        x_pred_rep = torch.gather(
            einops.rearrange(x_pred, "b l n d -> b (l n) d"), dim=1,
            index=einops.repeat(atom_pseudo_beta_index, "b l -> (b n) l 3", n=diffusion_batch_size)
        ) * pseudo_beta_mask[..., None] * single_mask[..., None]

        # distance binning
        lower_breaks = torch.linspace(self.min_bin, self.max_bin, self.no_bins, device=z.device)
        lower_breaks = lower_breaks ** 2
        upper_breaks = torch.cat(
            [lower_breaks[1:], torch.tensor(self.inf, device=z.device).reshape(1)], dim=0
        )
        dist2 = torch.sum(
            (x_pred_rep[..., None, :] - x_pred_rep[..., None, :, :]) ** 2,
            dim=-1, keepdims=True
        ) * pair_mask.unsqueeze(-1)
        dgram = ((dist2 > lower_breaks).to(z.dtype)
                 * (dist2 <= upper_breaks).to(z.dtype)
                 * pair_mask.unsqueeze(-1))

        # pairwise conditioning (overridable via _apply_pairwise_conditioning)
        z = self._apply_pairwise_conditioning(z, pair_mask, dgram)

        # pairformer stack
        s, z = self._run_pairformer_stack(s, z, single_mask, pair_mask, diffusion_batch_size)

        return s, z, diffusion_batch_size

    def _apply_pairwise_conditioning(self, z, pair_mask, dgram):
        """Apply pairwise conditioning. Override for custom behavior (e.g., structure_scale)."""
        return z + self.pairwise_conditioner(
            z_trunk=z, pair_mask=pair_mask, token_rel_pos_feats=self.linear_d(dgram)
        )

    def _run_pairformer_stack(self, s, z, single_mask, pair_mask, diffusion_batch_size):
        """Run pairformer stack, handling diffusion_batch_size > 1."""
        inplace_safe = False
        if diffusion_batch_size > 1:
            for name, mask in (
                ("single_mask", single_mask),
                ("pair_mask", pair_mask),
            ):
                if mask.shape[0] not in (1, diffusion_batch_size):
                    raise ValueError(
                        f"{name} batch dimension must be 1 or match "
                        "diffusion_batch_size"
                    )
            s_output = einops.repeat(
                torch.zeros_like(s),
                "b ... -> (b n) ...",
                n=diffusion_batch_size,
            )
            z_output = torch.zeros_like(z)
            for j in range(diffusion_batch_size):
                single_mask_j = (
                    single_mask
                    if single_mask.shape[0] == 1
                    else single_mask[j:j+1]
                )
                pair_mask_j = (
                    pair_mask
                    if pair_mask.shape[0] == 1
                    else pair_mask[j:j+1]
                )
                s_output[j:j+1], z_output[j:j+1] = self.pairformer_stack(
                    s, z[j:j+1],
                    single_mask=single_mask_j.to(dtype=s.dtype),
                    pair_mask=pair_mask_j.to(dtype=z.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )
            return s_output, z_output
        else:
            return self.pairformer_stack(
                s, z,
                single_mask=single_mask.to(dtype=s.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )


@dataclass(frozen=True)
class PredictedAffinityMasks:
    pocket_mask: torch.Tensor
    single_mask: torch.Tensor
    pair_mask: torch.Tensor
    fallback_codes: torch.Tensor
    valid_protein_mask: torch.Tensor


def build_predicted_affinity_masks(
    x_pred: torch.Tensor,
    *,
    is_ligand: torch.Tensor,
    is_protein: torch.Tensor,
    seq_mask: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    cutoff: float = 8.0,
    retry_cutoff: float = 12.0,
    nearest_count: int = 64,
) -> PredictedAffinityMasks:
    """Build strict V1 affinity masks from predicted coordinates.

    Pocket selection uses the configured cutoff first, then a relaxed cutoff,
    then the nearest finite protein tokens. If coordinates cannot support any
    distance calculation, all valid protein tokens are used.
    """
    if not isinstance(x_pred, torch.Tensor):
        raise TypeError("x_pred must be a torch.Tensor")
    if x_pred.ndim != 4 or x_pred.shape[-1] != 3:
        raise ValueError("x_pred must have shape [batch, tokens, atoms, 3]")
    if not torch.is_floating_point(x_pred):
        raise TypeError("x_pred must use a floating point dtype")

    batch_size, token_count, atom_count, _ = x_pred.shape
    token_shape = (batch_size, token_count)
    for name, value in (
        ("is_ligand", is_ligand),
        ("is_protein", is_protein),
        ("seq_mask", seq_mask),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tuple(value.shape) != token_shape:
            raise ValueError(f"{name} must have shape {token_shape}")
        if value.device != x_pred.device:
            raise ValueError("all inputs must be on the same device")

    atom_shape = (batch_size, token_count, atom_count)
    if atom_mask is None:
        atom_mask = torch.ones(
            atom_shape, dtype=torch.bool, device=x_pred.device
        )
    else:
        if not isinstance(atom_mask, torch.Tensor):
            raise TypeError("atom_mask must be a torch.Tensor")
        if tuple(atom_mask.shape) != atom_shape:
            raise ValueError(f"atom_mask must have shape {atom_shape}")
        if atom_mask.device != x_pred.device:
            raise ValueError("all inputs must be on the same device")

    if isinstance(cutoff, bool) or not isinstance(cutoff, Real):
        raise TypeError("cutoff must be a real number")
    cutoff = float(cutoff)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be positive and finite")
    if isinstance(retry_cutoff, bool) or not isinstance(retry_cutoff, Real):
        raise TypeError("retry_cutoff must be a real number")
    retry_cutoff = float(retry_cutoff)
    if not math.isfinite(retry_cutoff) or retry_cutoff <= 0.0:
        raise ValueError("retry_cutoff must be positive and finite")
    if retry_cutoff < cutoff:
        raise ValueError("retry_cutoff must be greater than or equal to cutoff")
    if (
        isinstance(nearest_count, bool)
        or not isinstance(nearest_count, Integral)
        or nearest_count <= 0
    ):
        raise ValueError("nearest_count must be a positive integer")
    nearest_count = min(int(nearest_count), 64)

    raw_ligand_mask = is_ligand.bool()
    raw_protein_mask = is_protein.bool()
    if (raw_ligand_mask & raw_protein_mask).any():
        raise ValueError("is_ligand and is_protein must be disjoint")

    valid_tokens = seq_mask.bool()
    ligand_mask = raw_ligand_mask & valid_tokens
    valid_protein = raw_protein_mask & valid_tokens
    valid_atoms = atom_mask.bool() & torch.isfinite(x_pred).all(dim=-1)
    pocket_mask = torch.zeros_like(valid_protein)
    fallback_codes = torch.full(
        (batch_size,), 3, dtype=torch.long, device=x_pred.device
    )

    for batch_index in range(batch_size):
        protein_indices = torch.nonzero(
            valid_protein[batch_index], as_tuple=False
        ).flatten()
        ligand_indices = torch.nonzero(
            ligand_mask[batch_index], as_tuple=False
        ).flatten()
        if protein_indices.numel() == 0 or ligand_indices.numel() == 0:
            pocket_mask[batch_index] = valid_protein[batch_index]
            continue

        protein_xyz = x_pred[batch_index, protein_indices]
        protein_atom_mask = valid_atoms[batch_index, protein_indices]
        ligand_xyz = x_pred[batch_index, ligand_indices]
        ligand_atom_mask = valid_atoms[batch_index, ligand_indices]
        ligand_atoms = ligand_xyz[ligand_atom_mask]
        if ligand_atoms.numel() == 0 or not protein_atom_mask.any():
            pocket_mask[batch_index] = valid_protein[batch_index]
            continue

        delta = protein_xyz[:, :, None, :] - ligand_atoms[None, None, :, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
        distances = distances.masked_fill(
            ~protein_atom_mask[:, :, None], float("inf")
        )
        minimum_by_token = distances.amin(dim=(1, 2))

        selected = minimum_by_token < cutoff
        fallback_code = 0
        if not selected.any():
            selected = minimum_by_token < retry_cutoff
            fallback_code = 1
        if not selected.any():
            finite_indices = torch.nonzero(
                torch.isfinite(minimum_by_token), as_tuple=False
            ).flatten()
            if finite_indices.numel() == 0:
                pocket_mask[batch_index] = valid_protein[batch_index]
                continue
            keep_count = min(nearest_count, int(finite_indices.numel()))
            distance_order = torch.argsort(
                minimum_by_token[finite_indices], stable=True
            )
            selected = torch.zeros_like(minimum_by_token, dtype=torch.bool)
            selected[finite_indices[distance_order[:keep_count]]] = True
            fallback_code = 2

        pocket_mask[batch_index, protein_indices] = selected
        fallback_codes[batch_index] = fallback_code

    single_mask = valid_tokens & (pocket_mask | ligand_mask)
    pair_mask = make_ligand_pocket_pair_mask(
        pocket_mask=pocket_mask,
        is_ligand=ligand_mask,
        seq_mask=valid_tokens,
    )
    return PredictedAffinityMasks(
        pocket_mask=pocket_mask,
        single_mask=single_mask,
        pair_mask=pair_mask,
        fallback_codes=fallback_codes,
        valid_protein_mask=valid_protein,
    )


def _expand_affinity_batch_tensor(
    value: torch.Tensor,
    *,
    name: str,
    base_batch_size: int,
    coordinate_batch_size: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise TypeError(f"{name} must be a batched torch.Tensor")
    if value.shape[0] != base_batch_size:
        raise ValueError(
            f"{name} batch dimension must match seq_mask batch dimension"
        )
    if (
        base_batch_size <= 0
        or coordinate_batch_size <= 0
        or coordinate_batch_size % base_batch_size != 0
    ):
        raise ValueError(
            "x_pred batch dimension must be positive and divisible by "
            "seq_mask batch dimension"
        )
    diffusion_batch_size = coordinate_batch_size // base_batch_size
    if base_batch_size > 1 and diffusion_batch_size > 1:
        raise ValueError(
            "batch size and diffusion batch size cannot both exceed one"
        )
    if diffusion_batch_size == 1:
        return value
    return einops.repeat(
        value, "b ... -> (b n) ...", n=diffusion_batch_size
    )


def _build_module_predicted_affinity_masks(
    x_pred: torch.Tensor,
    batch: dict,
    *,
    cutoff: float,
) -> tuple[PredictedAffinityMasks, dict[str, torch.Tensor]]:
    if not isinstance(x_pred, torch.Tensor) or x_pred.ndim != 4:
        raise ValueError("x_pred must have shape [batch, tokens, atoms, 3]")
    seq_mask = batch["seq_mask"]
    if not isinstance(seq_mask, torch.Tensor) or seq_mask.ndim != 2:
        raise ValueError("seq_mask must have shape [batch, tokens]")
    if "pred_dense_atom_mask" not in batch:
        raise KeyError(
            "pred_dense_atom_mask is required for predicted pocket masks"
        )

    base_batch_size = seq_mask.shape[0]
    coordinate_batch_size = x_pred.shape[0]
    expanded = {
        name: _expand_affinity_batch_tensor(
            batch[name],
            name=name,
            base_batch_size=base_batch_size,
            coordinate_batch_size=coordinate_batch_size,
        )
        for name in (
            "seq_mask",
            "is_ligand",
            "is_protein",
            "pred_dense_atom_mask",
        )
    }
    masks = build_predicted_affinity_masks(
        x_pred,
        is_ligand=expanded["is_ligand"],
        is_protein=expanded["is_protein"],
        seq_mask=expanded["seq_mask"],
        atom_mask=expanded["pred_dense_atom_mask"],
        cutoff=cutoff,
        retry_cutoff=max(12.0, float(cutoff)),
        nearest_count=64,
    )
    return masks, expanded




def _compute_raw_pocket_hits_and_distances(
    x_pred: torch.Tensor,
    is_ligand: torch.Tensor,
    is_protein: torch.Tensor,
    atom_mask: torch.Tensor = None,
    seq_mask: torch.Tensor = None,
    cutoff: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, A, _ = x_pred.shape
    if atom_mask is None:
        atom_mask = torch.ones(B, L, A, dtype=torch.bool, device=x_pred.device)

    raw_hits = torch.zeros(B, L, dtype=torch.bool, device=x_pred.device)
    min_distances = torch.full((B, L), float("inf"), dtype=x_pred.dtype, device=x_pred.device)

    for b in range(B):
        valid_token = torch.ones(L, dtype=torch.bool, device=x_pred.device)
        if seq_mask is not None:
            valid_token = seq_mask[b].bool()

        pro_sel = is_protein[b].bool() & valid_token
        lig_sel = is_ligand[b].bool() & valid_token
        if not pro_sel.any() or not lig_sel.any():
            continue

        pro_xyz = x_pred[b, pro_sel]
        lig_xyz = x_pred[b, lig_sel]
        pro_mask = atom_mask[b, pro_sel].bool()
        lig_mask = atom_mask[b, lig_sel].bool()
        lig_atoms = lig_xyz[lig_mask]
        if lig_atoms.numel() == 0:
            continue

        n_protein = pro_xyz.shape[0]
        pro_atoms = pro_xyz.reshape(n_protein * A, 3)
        pro_atom_valid = pro_mask.reshape(n_protein * A)
        min_dist_per_atom = torch.full((n_protein * A,), float("inf"), dtype=x_pred.dtype, device=x_pred.device)

        if pro_atom_valid.any():
            dist_mat = torch.cdist(pro_atoms[pro_atom_valid], lig_atoms, p=2)
            min_dist_per_atom[pro_atom_valid] = dist_mat.min(dim=1).values

        min_dist_per_token = min_dist_per_atom.view(n_protein, A).min(dim=1).values
        protein_indices = pro_sel.nonzero(as_tuple=False).squeeze(-1)
        min_distances[b, protein_indices] = min_dist_per_token
        raw_hits[b, protein_indices] = min_dist_per_token < cutoff

    return raw_hits, min_distances




def make_ligand_pocket_pair_mask(
    pocket_mask: torch.Tensor,
    is_ligand: torch.Tensor,
    seq_mask: torch.Tensor = None,
) -> torch.Tensor:
    pocket_mask = pocket_mask.bool()
    is_ligand = is_ligand.bool()
    pocket_to_ligand = pocket_mask[..., None] & is_ligand[..., None, :]
    ligand_to_pocket = is_ligand[..., None] & pocket_mask[..., None, :]
    ligand_to_ligand = is_ligand[..., None] & is_ligand[..., None, :]
    pair_mask = pocket_to_ligand | ligand_to_pocket | ligand_to_ligand
    if seq_mask is not None:
        seq_mask = seq_mask.bool()
        pair_mask = pair_mask & seq_mask[..., None] & seq_mask[..., None, :]
    return pair_mask


class LigandPocketBinAffinityPoolHead(torch.nn.Module):
    def __init__(self, config):
        """
        Args:
            c_s:
                Input channel dimension
        """
        super().__init__()
        # pairformer blocks
        self.config = config.affinity_head    
        self.c_s = self.config["c_s"]
        self.c_z = self.config["c_z"]
        self.c_s_inputs = self.config["c_s_inputs"]
        self.pocket_no_bins = self.config["pocket_no_bins"]
        self.config_pairformer_stack = self.config["pairformer_stack"]

        self.gate_ln = torch.nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        self.gate_linear = torch.nn.Linear(self.c_s, 1, bias=True)
        self.ln = torch.nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        hidden_size = 512
        self.mlp = nn.Sequential(
            nn.Linear(self.c_s, hidden_size),  # First layer
            nn.ReLU(),                          # Activation
            nn.Linear(hidden_size, hidden_size), # Second layers
            nn.ReLU(),                          # Activation
        )

        self.z_gate_ln = torch.nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_gate_linear = torch.nn.Linear(self.c_z, 1, bias=True)
        self.z_ln = torch.nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_mlp = nn.Sequential(
            nn.Linear(self.c_z, hidden_size),  # First layer
            nn.ReLU(),                          # Activation
            nn.Linear(hidden_size, hidden_size), # Second layer
            nn.ReLU(),                          # Activation
        )

        self.interact_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # Second layer
            nn.ReLU(),                          # Activation
            nn.Linear(hidden_size, self.pocket_no_bins) # Output layer
        )


    def forward(self, s, z, single_mask=None, is_ligand=None, head_pair_mask=None, affinity_path=None, file_ids=None, weighted=2.0):
        """
        s: [b,l,d]
        z: [b,l,l,d]
        single_mask: [b,l]
        is_ligand: [b,l]
        head_pair_mask: [b,l,l],pocket and inter mask

        """
        inplace_safe = False
        pair_mask = single_mask[..., None] * single_mask[..., None, :]  
        lig_mask = is_ligand[..., None] * is_ligand[..., None, :] 
        head_pair_mask = head_pair_mask | lig_mask

        bs = s.shape[0]
        z_dim = z.shape[-1]


        single_mask = single_mask.bool()
        affinities = []

        for i in range(bs):
            s_batch = s[i][single_mask[i]]

            gate = self.gate_linear(self.gate_ln(s_batch)).sigmoid()
            s_affinity = (gate * self.mlp(self.ln(s_batch)))

            z_mask = pair_mask[i].bool()
            z_batch = z[i][z_mask] 
            z_mask_head = head_pair_mask[i][z_mask]

            num_tokens = single_mask[i].sum()
            z_batch = z_batch.reshape(num_tokens, num_tokens, z_dim)
            z_mask_head = z_mask_head.reshape(num_tokens, num_tokens, 1)
            z_gate = self.z_gate_linear(self.z_gate_ln(z_batch)).sigmoid() * z_mask_head
            z_affinity = (z_gate * self.z_mlp(self.z_ln(z_batch))).sum(dim=1) / (z_mask_head.sum(dim=1) + 1e-7)

            batch_affinity = s_affinity + z_affinity
            global_s_batch = torch.nn.AdaptiveAvgPool1d(1)(batch_affinity.transpose(0,1)).squeeze(-1)
            interact_logit = self.interact_mlp(global_s_batch)

            # affinities.append(batch_affinity.sum(dim=0, keepdim=True))
            affinities.append(interact_logit)
            
        affinity = torch.stack([a.view(-1) for a in affinities], dim=0) 
        
        return affinity


class AffinityModulePocket(BaseAffinityModule):
    def __init__(self, config):
        super().__init__(config)
        self.cutoff = float(self.config.get("cutoff", 8.0))
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=self.c_z, dim_token_rel_pos_feats=self.c_z, num_transitions=2,
        )
        self.ligand_affinity = LigandPocketBinAffinityPoolHead(config)

    def forward(self, s_inputs, s, z, x_pred, batch):
        masks, expanded = _build_module_predicted_affinity_masks(
            x_pred, batch, cutoff=self.cutoff
        )
        pairformer_pair_mask = (
            masks.single_mask[..., :, None]
            & masks.single_mask[..., None, :]
        )

        s, z, _ = self._compute_structure_features(
            s_inputs,
            s,
            z,
            x_pred,
            batch,
            pairformer_pair_mask,
            single_mask=masks.single_mask,
        )

        affinity_out = self.ligand_affinity(
            s, z,
            single_mask=masks.single_mask,
            is_ligand=expanded["is_ligand"],
            head_pair_mask=masks.pair_mask,
        )
        return {
            "affinity_logits": affinity_out,
            "pocket_fallback_code": masks.fallback_codes,
        }
