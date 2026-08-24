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

import json
import numpy as np
import torch
from onixbind.openfold.model.heads import compute_tm
from scipy import spatial


_IPTM_WEIGHT = 0.8
_FRACTION_DISORDERED_WEIGHT = 0.5
_CLASH_PENALIZATION_WEIGHT = 100.0


def calculate_chain_based_ptm(
    p_pae,
    input_features
    ):
    '''
    Calculate chain-based ptm and iptm
    
    Args:
        p_pae: [bs, num_token, num_token, 64] the logit of pae
        diffusion_batch_size: int
        input_features: dict
    '''
    diffusion_batch_size = p_pae.shape[0]
    single_mask = input_features["seq_mask"]
    frame_mask = input_features["frame_mask"]
    asym_id = input_features["asym_id"]
    asym_id_to_asym_mask = {aid.item(): asym_id == aid for aid in torch.unique(asym_id)}
    if 0 in asym_id_to_asym_mask:
        asym_id_to_asym_mask.pop(0)   ## pad asym_id
    N_chain = len(asym_id_to_asym_mask)
    chain_ptm = torch.zeros(size=(diffusion_batch_size, N_chain,)).to(p_pae.device)
    chain_pair_iptm = torch.zeros(size=(diffusion_batch_size, N_chain, N_chain)).to(p_pae.device)
    chain_iptm = torch.zeros(size=(diffusion_batch_size, N_chain,)).to(p_pae.device)
    
    for index in range(diffusion_batch_size):
        ### chain-based ptm
        for aid, asym_mask in asym_id_to_asym_mask.items():
            chain_ptm[index, aid-1] = compute_tm(p_pae[index:index+1], residue_weights=frame_mask * single_mask * asym_mask)
        ### chain-pair based iptm
        for aid_i in range(N_chain):
            for aid_j in range(N_chain):
                ### same chain iptm equivalent to ptm
                if aid_i == aid_j:
                    chain_pair_iptm[index, aid_i, aid_j] = chain_ptm[index, aid_i]
                    continue
                if aid_i > aid_j:
                    chain_pair_iptm[index, aid_i, aid_j] = chain_pair_iptm[index, aid_j, aid_i]
                pair_asym_mask = asym_id_to_asym_mask[aid_i+1] + asym_id_to_asym_mask[aid_j+1]
                chain_pair_iptm[index, aid_i, aid_j] = compute_tm(p_pae[index:index+1], residue_weights=frame_mask * single_mask * pair_asym_mask, interface=True, asym_id=asym_id)
                
        ### chain-based iptm
        for aid, asym_mask in asym_id_to_asym_mask.items():
            ## asym_id i with all other asym_id j(i!=j)
            pairs = [
                (i, j)
                for i in range(N_chain)
                for j in range(N_chain)
                if (i == aid-1 or j == aid-1) and (i != j)
            ]
            vals = [chain_pair_iptm[index, i, j] for (i, j) in pairs]
            if len(vals) > 0:
                chain_iptm[index, aid-1] = torch.stack(vals, dim=-1).mean(dim=-1)
    return chain_iptm.cpu().numpy(), chain_pair_iptm.cpu().numpy(), chain_ptm.cpu().numpy()


def calculate_chain_based_plddt(
    plddt,
    input_features
):
    """
    Calculate chain-based pLDDT

    Args:
        plddt: [bs, num_token, max_atoms] the logit of pLDDT
        input_features: dict
    """
    diffusion_batch_size = plddt.shape[0]
    pred_dense_atom_mask = input_features['pred_dense_atom_mask'].cpu()   
    single_mask = input_features["seq_mask"]
    asym_id = input_features["asym_id"].cpu()
    asym_id_to_asym_mask = {aid.item(): asym_id == aid for aid in torch.unique(asym_id)}
    if 0 in asym_id_to_asym_mask:
        asym_id_to_asym_mask.pop(0)   ## pad asym_id
    N_chain = len(asym_id_to_asym_mask)
    chain_plddt = torch.zeros(size=(diffusion_batch_size, N_chain,)).to(plddt.device)
    chain_pair_plddt = torch.zeros(size=(diffusion_batch_size, N_chain, N_chain)).to(plddt.device)
    
    for index in range(diffusion_batch_size):
        ### chain-based pLDDT
        for aid, asym_mask in asym_id_to_asym_mask.items():
            asym_pred_dense_atom_mask = (pred_dense_atom_mask * asym_mask.unsqueeze(-1)).squeeze(0)
            chain_plddt[index, aid-1] = plddt[index, asym_pred_dense_atom_mask].mean()
            
        #### chain-pair based pLDDT
        for aid_i in range(N_chain):
            for aid_j in range(N_chain):
                if aid_i == aid_j:
                    chain_pair_plddt[index, aid_i, aid_j] = chain_plddt[index, aid_i]
                    continue
                if aid_i > aid_j:
                    chain_pair_plddt[index, aid_i, aid_j] = chain_pair_plddt[index, aid_j, aid_i]
                pair_asym_mask = asym_id_to_asym_mask[aid_i+1] + asym_id_to_asym_mask[aid_j+1]
                pair_asym_pred_dense_atom_mask = (pred_dense_atom_mask * pair_asym_mask.unsqueeze(-1)).squeeze(0)
                chain_pair_plddt[index, aid_i, aid_j] = plddt[index, pair_asym_pred_dense_atom_mask].mean()

    return chain_plddt.cpu().numpy(), chain_plddt.cpu().numpy()


    
def calculate_clash(
    coords: np.ndarray,
    input_features: dict,
    cutoff_radius: float = 1.1,
    min_clashes_for_overlap: int = 100,
    min_fraction_for_overlap: float = 0.5,
):
    """Determine whether the structure has at least one clashing chain.

    A clashing chain is defined as having greater than 100 polymer atoms within
    1.1A of another polymer atom, or having more than 50% of the chain with
    clashing atoms.

    Args:
        coords: The coordinates of the atoms in the structure.
        cutoff_radius: atom distances under this threshold are considered a clash.
        min_clashes_for_overlap: The minimum number of atom-atom clashes for a chain
        to be considered overlapping.
        min_fraction_for_overlap: The minimum fraction of atoms within a chain that
        are clashing for the chain to be considered overlapping.

    Returns:
        True if the structure has at least one clashing chain.
    """
    has_clashes = np.zeros(len(coords), dtype=np.float32)
    is_polymer = (~input_features['is_ligand'] & input_features['seq_mask']).cpu()
    if is_polymer.sum() == 0:
        return has_clashes
    pred_dense_atom_mask = input_features['pred_dense_atom_mask'].cpu()
    pred_dense_atom_mask = pred_dense_atom_mask * is_polymer.unsqueeze(-1)
    coords = coords[:, pred_dense_atom_mask[0]]
    ## repeat the chain index and residue index for each atom
    max_atoms = 24
    atom_leval_resid = input_features['residue_index'].cpu().unsqueeze(-1).repeat(1, 1, max_atoms)[pred_dense_atom_mask]
    atom_level_chainid = input_features['asym_id'].cpu().unsqueeze(-1).repeat(1, 1, max_atoms)[pred_dense_atom_mask]
    chain_ids = np.unique(atom_level_chainid)
    
    
    for index in range(len(coords)):
        coord_kdtree = spatial.cKDTree(coords[index])
        clashes_per_atom = coord_kdtree.query_ball_point(coords[index], p=2.0, r=cutoff_radius)
        per_atom_has_clash = np.zeros(len(coords[index]), dtype=np.int32)
                
        for atom_idx, clashing_indices in enumerate(clashes_per_atom):
            for clashing_idx in clashing_indices:
                if np.abs(atom_leval_resid[atom_idx] - atom_leval_resid[clashing_idx]) > 1 or (atom_level_chainid[atom_idx] != atom_level_chainid[clashing_idx]):
                    per_atom_has_clash[atom_idx] = 1.0
                    break
                
        ### calculate the atom_clash ratio(frac_clashes) of each chain
        for chain_id in chain_ids:
            mask = (atom_level_chainid == chain_id).numpy()
            num_atoms = mask.sum()
            if num_atoms == 0:
                continue
            num_clashes = per_atom_has_clash[mask].sum()
            frac_clashes = num_clashes / num_atoms
            if (num_clashes > min_clashes_for_overlap or frac_clashes > min_fraction_for_overlap):
                has_clashes[index] = 1.0
                break
    return has_clashes


    

