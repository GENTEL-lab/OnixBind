# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from scipy.spatial.distance import cdist



def identify_mol_type(
    ref_space_uid: np.ndarray,
    atom_sums: np.ndarray,
    chain_id: np.ndarray,
    chain_lengths: np.ndarray,
) -> np.ndarray:
    """
    Generate mol_type masks based on the given rules.

    Args:
        ref_space_uid (np.ndarray): A tensor of unique ids, shape (N,).
        atom_sums (np.ndarray): A np.ndarray of atom sums corresponding to each unique id, shape (N,).
        chain_id (np.ndarray): A np.ndarray of chain IDs corresponding to each unique id, shape (N,).
        chain_lengths (np.ndarray): A np.ndarray of chain lengths, shape (num_chains,).

    Returns:
        is_metal (np.ndarray): A mask indicating metals.
        first_indices (np.ndarray): A np.ndarray of first indices for each unique id, shape (N,).
        last_indices (np.ndarray): A np.ndarray of last indices for each unique id, shape (N,).
    """

    assert (
        ref_space_uid.shape == atom_sums.shape
    ), "ref_space_uid and atom_sums must have the same shape."
    # Initialize masks
    is_metal = np.zeros_like(ref_space_uid, dtype=bool)
    first_indices = np.zeros_like(ref_space_uid, dtype=int)
    last_indices = np.zeros_like(ref_space_uid, dtype=int)

    # Count occurrences of each ref_space_uid
    unique_ids, counts = np.unique(ref_space_uid, return_counts=True)
    for unique_id, count in zip(unique_ids, counts):
        mask = ref_space_uid == unique_id
        
        first_index = np.nonzero(mask)[0][0]
        last_index = np.nonzero(mask)[0][-1]
        first_indices[mask] = first_index
        last_indices[mask] = last_index
        atom_sum = atom_sums[mask]

        if count == 1 and chain_lengths[chain_id[mask].astype(int)] == 1:
            is_metal[mask] = atom_sum == 1

    return (
        is_metal,
        first_indices,
        last_indices,
    )


def get_interface_token(
    chain_id: np.ndarray,
    reference_chain_id: np.ndarray,
    token_distance: np.ndarray,
    token_distance_mask: np.ndarray,
    interface_minimal_distance: int = 15,
) -> np.ndarray:
    """
    Get tokens in contact with the other chain.
    Args:
        chain_id:           [all_token_length, ], chain ID of each token
        reference_chain_id: [1] or [2], the reference atom is selected within the reference chains
        token_distance:     [chain/interface_token_length, all_token_length], distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask:[chain/interface_token_length, all_token_length], indicates valid distance
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        interface_token_indices: indices of tokens of interface
    """
    # expand reference_chain_id to chain_id shape
    expand_reference_chain_id = np.zeros_like(chain_id, dtype=int) #! noted
    for _chain_id in reference_chain_id:
        expand_reference_chain_id += chain_id == _chain_id

    # get distance mask, difference chain mask
    mask_distance = token_distance < interface_minimal_distance
    
    mask_diff_chain = (chain_id[None, :] != chain_id[:, None])[
        expand_reference_chain_id.nonzero()[0]
    ]

    mask = mask_distance * mask_diff_chain * token_distance_mask
    mask_interface = np.sum(mask, axis = -1)
    
    interface_token_indices = np.nonzero(mask_interface)[0]
    return interface_token_indices


def get_spatial_crop_index(
    chain_id: np.ndarray,
    token_distance: np.ndarray,
    token_distance_mask: np.ndarray,
    reference_chain_id: np.ndarray,
    ref_space_uid_token: np.ndarray,
    crop_size: int,
    crop_complete_ligand_unstdRes: bool = False,
    interface_crop: bool = False,
    interface_minimal_distance: int = 15,
) -> np.ndarray:
    """
    Crop sequences continuesly across chains.
    Args:
        chain_id: [all_token_length,], all tokens' chain ID within an assembly
        token_distance: [chain/interface_token_length, all_token_length], distance matrix between the chain/interface tokens and the assembly tokens
        token_distance_mask: [chain/interface_token_length, all_token_length], indicates valid distance
        reference_chain_id:  [1] or [2],the reference atom is selected within the reference_chains ID
        crop_size: total crop size of the whole assembly
        interface_crop: whether use interface tokens as referenced token
        interface_minimal_distance: the minimal distance to any other chains
    Returns:
        selected_token_indices: np.ndarray, shape=(min(crop_size, tokens.shape[0]), )
    """

    # interface spatial cropping: select reference tokens with contact to the other
    if interface_crop and interface_minimal_distance is not None:
        reference_token_indices = get_interface_token(
            chain_id=chain_id,
            reference_chain_id=reference_chain_id,
            token_distance=token_distance,
            token_distance_mask=token_distance_mask,
            interface_minimal_distance=interface_minimal_distance,
        )
        if len(reference_token_indices) < 1 and len(reference_chain_id) == 1:
            # If a chain does not contain any interfacial atoms, use all resolved tokens.
            reference_token_indices = np.nonzero(np.any(token_distance_mask.astype(bool), axis=-1))[0]
    else:
        # select reference tokens within the given chain or interface
        reference_token_indices = np.nonzero(np.any(token_distance_mask.astype(bool), axis=-1))[0]

    # random select one token from reference_token_indices
    if len(reference_token_indices) == 0:
        raise ValueError(f"No resolved atoms in reference tokens! Spatial Interface: {interface_crop}")

    random_idx = random.randint(0, len(reference_token_indices) - 1)
    reference_token_idx = reference_token_indices[random_idx].item()

    assert (
        token_distance_mask[reference_token_idx].astype(bool).any()
    ), "Select a unresolved reference token"
    distance_to_reference = token_distance[reference_token_idx]
    # add noise to break tie
    noise_break_tie = np.arange(0, distance_to_reference.shape[0]).astype(np.float32) * 1e-3

    distance_to_reference_mask = token_distance_mask[reference_token_idx]

    distance_to_reference = np.where(
        distance_to_reference_mask.astype(bool), distance_to_reference, np.inf
    )
    # find k nearest tokens
    nearest_k = min(crop_size, chain_id.shape[0])
    selected_token_indices = np.sort(np.argsort(distance_to_reference + noise_break_tie)[:nearest_k]) # need check
    #! noted

    def drop_uncompleted_mol(selected_token_indices):
        selected_uid = ref_space_uid_token[selected_token_indices]
        mask = np.ones_like(ref_space_uid_token, dtype=bool)
        mask[selected_token_indices] = False
        unselected_uid = ref_space_uid_token[mask]

        # Find overlap elements
        overlap_uid = np.array(np.intersect1d(selected_uid, unselected_uid))

        # Remove overlap elements from elements_B
        remain_indices = selected_token_indices[
            ~np.isin(selected_uid, overlap_uid)
        ].astype(int)
        return remain_indices
    
    selected_token_indices = selected_token_indices.flatten()
    if crop_complete_ligand_unstdRes is True:
        selected_token_indices = drop_uncompleted_mol(selected_token_indices)
    assert (
        selected_token_indices.shape[0] <= crop_size
    ), f"Spatial cropping crop {selected_token_indices.shape[0]}, more than {crop_size} tokens!!"
    return selected_token_indices, reference_token_idx


def get_continues_crop_index(
    chain_id: np.ndarray,
    ref_space_uid_token: np.ndarray,
    atom_sums: np.ndarray,
    crop_size: int,
    crop_complete_ligand_unstdRes: Optional[bool] = False,
    drop_last: Optional[bool] = False,
    remove_metal: Optional[bool] = False,
) -> np.ndarray:
    """
    Crop sequences continuesly across chains. Reference: AF-multimer Algorithm 1.
    Args:
        chain_id:  [all_token_length,], all tokens' chain ID within an assembly
        atom_sums: [all_token_length,] sum of atoms within one ref_space_uid
        ref_space_uid: [all_atom_length,] unique chain-residue id
        crop_size: total crop size of the whole assembly
        crop_complete_ligand_unstdRes: Whether to crop the complete ligand or unstandard residues.
                              If False, the ligand is usually fragmented during sequential cropping.
        drop_last: whether to ensure all ligands or unstandard residues to be cropped completely,
                    if not, we will ignore the completion of the last one to meet the crop_size quota.
        remove_metal: whether remove all metal/ions
    Returns:
        selected_token_indices: np.ndarray, shape=(crop_size, )
    """
    # get chain counts info
    unique_chain_id = np.unique(chain_id)
    chain_lengths = np.bincount(chain_id.astype(int))
    chain_offset_list = np.array(
        [np.where(chain_id == chain_idx)[0][0] for chain_idx in unique_chain_id],
    )
    
    
    # identify the mol type
    (
        is_metal,
        uid_first_indices,
        uid_last_indices,
    ) = identify_mol_type(ref_space_uid_token, atom_sums, chain_id, chain_lengths)

    def _qualify_crop_size(cur_crop_size, crop_size_min, N_added):
        if cur_crop_size < crop_size_min:
            return False
        if cur_crop_size + N_added > crop_size:
            return False
        return True

    def _determine_start_end_point(start_idx, end_idx, crop_size_min, N_added):
        if start_idx == end_idx:
            return start_idx, end_idx

        # determine the start_idx
        left_start_point = right_start_point = start_idx
        # if this is not the first time this uid occurants, then it must be a middle point
        if uid_first_indices[start_idx] != start_idx:
            start_in_middle = True
            left_start_point = uid_first_indices[start_idx]
            right_start_point = uid_last_indices[start_idx] + 1
        else:
            start_in_middle = False

        # determine the end_idx
        left_end_point = right_end_point = end_idx
        # if this is not the last time this uid occurants, then it must be a middle point
        if end_idx > 0 and uid_last_indices[end_idx - 1] != end_idx - 1:
            end_in_middle = True
            left_end_point = uid_first_indices[end_idx - 1]
            right_end_point = uid_last_indices[end_idx - 1] + 1
        else:
            end_in_middle = False

        if start_in_middle is False and end_in_middle is False:
            return start_idx, end_idx
        elif start_in_middle is True and end_in_middle is True:
            # alwalys use left edge
            start_in_middle = False
            start_idx = left_start_point

        if start_in_middle is False and end_in_middle is True:
            # need to determine: use left end or right end
            left_crop_size = left_end_point - start_idx
            right_crop_size = right_end_point - start_idx
            is_left_ok = _qualify_crop_size(left_crop_size, crop_size_min, N_added)
            is_right_ok = _qualify_crop_size(right_crop_size, crop_size_min, N_added)
            if is_left_ok and is_right_ok:
                end_idx = (
                    left_end_point
                    if np.random.randint(0, 2) == 0
                    else right_end_point
                )
                return start_idx, end_idx
            elif is_left_ok:
                return start_idx, left_end_point
            elif is_right_ok:
                return start_idx, right_end_point
            elif drop_last is True:
                end_point = left_end_point
                while end_point - start_idx + N_added > crop_size:
                    if end_point > start_idx:
                        end_point = uid_first_indices[end_point - 1]
                    else:
                        break
                return start_idx, end_point
            else:
                cur_crop_size = min(end_idx - start_idx, crop_size - N_added)
                return start_idx, start_idx + cur_crop_size
        elif start_in_middle is True and end_in_middle is False:
            # need to determine: use left start or right start
            left_crop_size = end_idx - left_start_point
            right_crop_size = end_idx - right_start_point
            is_left_ok = _qualify_crop_size(left_crop_size, crop_size_min, N_added)
            is_right_ok = _qualify_crop_size(right_crop_size, crop_size_min, N_added)
            if is_left_ok and is_right_ok:
                start_idx = (
                    left_start_point
                    if np.random.randint(0, 2) == 0
                    else right_start_point
                )
                return start_idx, end_idx
            elif is_left_ok:
                return left_start_point, end_idx
            elif is_right_ok:
                return right_start_point, end_idx
            elif drop_last is True:
                return right_start_point, end_idx
            else:
                return start_idx, end_idx

    # shuffle the list of chains
    chain_shuffle_index = np.random.permutation(len(unique_chain_id))

    # crop over chains iteratively
    selected_token_indices = []
    N_added = 0  # number of tokens already selected
    N_remaining = len(chain_id)  # number of tokens in remaining chains
    if remove_metal is True:
        N_remaining -= sum(is_metal).item()
    for idx in chain_shuffle_index:
        if N_added >= crop_size:
            break

        # get chain type: whether it is metal/ions
        curr_is_metal = is_metal[chain_offset_list[idx]]
        # whether remove metal chain
        if remove_metal is True and curr_is_metal:
            # skip if it is metal/ions
            continue

        chain_length = chain_lengths[unique_chain_id[idx].astype(int)]
        N_remaining -= chain_length

        # determine the crop size
        crop_size_min = min(chain_length, max(0, crop_size - (N_added + N_remaining)))
        crop_size_max = min(crop_size - N_added, chain_length)
        if crop_size_min > crop_size_max:
            print(f"error crop_size: {crop_size_min} > {crop_size_max}")

        chain_crop_size = np.random.randint(crop_size_min, crop_size_max + 1)
        
        chain_crop_start = np.random.randint(0, chain_length - chain_crop_size + 1)

        chain_offset = chain_offset_list[idx]
        start_token_index = chain_offset + chain_crop_start
        end_token_index = chain_offset + chain_crop_start + chain_crop_size
        if crop_complete_ligand_unstdRes is True:
            start_token_index, end_token_index = _determine_start_end_point(
                start_token_index, end_token_index, crop_size_min, N_added
            )
            assert (
                end_token_index >= start_token_index
            ), f"invalid crop indices!! {start_token_index}, {end_token_index}"
            chain_crop_size = end_token_index - start_token_index

        selected_token_indices.append(
            np.arange(start_token_index, end_token_index)
        )
        N_added += chain_crop_size
        if crop_complete_ligand_unstdRes is True and drop_last is True:
            if start_token_index < end_token_index:
                assert uid_first_indices[start_token_index] == start_token_index
                assert uid_last_indices[end_token_index - 1] == end_token_index - 1

    selected_token_indices = np.concatenate(selected_token_indices)
    selected_token_indices = np.sort(selected_token_indices)
    
    if drop_last is True:
        assert (
            selected_token_indices.shape[0] <= crop_size
        ), f"Continuous cropping crop {selected_token_indices.shape[0]}, more than {crop_size} tokens!!"
    return selected_token_indices


class CropData(object):
    """
    Crop the data based on the given crop size and reference chain indices (asym_id).
    """

    def __init__(
        self,
        crop_size: int,
        ref_chain_indices: list[int],
        features: dict[str, Any],
        method_weights: list[float] = [0.2, 0.4, 0.4],
        monomer_method_weights: list[float] = [0.25, 0.75, 0.0],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = True,
        drop_last: bool = True,
        remove_metal: bool = True,
    ) -> None:
        """
        Args:
            crop_size (int): The size of the crop to be sampled.
            ref_chain_indices (list[int]): The "asym_id_int" of the reference chains.
            token_array (TokenArray): The token array.
            atom_array (AtomArray): The atom array.
            method_weights (list[float]): The weights corresponding to these three cropping methods:
                                          ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"].
            contiguous_crop_complete_lig: Whether to crop the complete ligand in ContiguousCropping method.

        """
        self.crop_size = crop_size
        self.ref_chain_indices = ref_chain_indices
        self.features = features
        self.method_weights = method_weights
        self.monomer_method_weights = monomer_method_weights
        self.cand_crop_methods = [
            "ContiguousCropping",
            "SpatialCropping",
            "SpatialInterfaceCropping",
        ]
        self.contiguous_crop_complete_lig = contiguous_crop_complete_lig
        self.spatial_crop_complete_lig = spatial_crop_complete_lig
        self.drop_last = drop_last
        self.remove_metal = remove_metal

    def random_crop_method(self, monomer=False) -> str:
        """
        Choose a random cropping method based on the given weights.

        Returns:
            str: The name of the randomly selected cropping method.
        """
        if monomer:
            return random.choices(self.cand_crop_methods, k=1, weights=self.monomer_method_weights)[
                0
            ]
        else:  
            return random.choices(self.cand_crop_methods, k=1, weights=self.method_weights)[
                0
            ]

    def get_token_dist_mat(self, token_indices_in_ref: np.ndarray) -> np.ndarray:
        """
        Get the distance matrix of the tokens in the reference chain.

        Args:
            token_indices_in_ref (list): The indices of the tokens in the reference chain.

        Returns:
            numpy.ndarray: The distance matrix of the tokens in the reference chain,
                           shape=(len(tokens_in_ref_chain), len(tokens)).
        """
        
        centre_atom_indices = self.features["residue_center_index"]
        centre_atom_coords = self.features["atom_positions"][np.arange(centre_atom_indices.shape[0]), centre_atom_indices]
        partial_token_dist_matrix = cdist(
            centre_atom_coords[token_indices_in_ref],
            centre_atom_coords,
            "euclidean",
        )

        assert partial_token_dist_matrix.shape == (
            len(token_indices_in_ref),
            len(centre_atom_indices),
        )
        return partial_token_dist_matrix


    def get_crop_indices(self, crop_method: str = None) -> np.ndarray:
        """
        Get selected indices based on the selected crop method.

        Args:
            crop_method (str): The cropping method to be used. Default is None.
        Returns:
            selected_indices : np.ndarray, shape=(N_selected, )
        """
        # tokens, chain_id, resolved_centre_token_mask_1d, token_indices_in_ref, is_ligand = (
        #     self.extract_info()
        # )
        
        residue_center_index = self.features['residue_center_index']
        chain_id = self.features['asym_id']
        resolved_centre_token_mask = self.features['resolved_atom_mask'][np.arange(residue_center_index.shape[0]), residue_center_index]
        token_indices_in_ref = np.where( np.isin(chain_id, self.ref_chain_indices) )[0]

        if crop_method is not None:
            assert (
                crop_method in self.cand_crop_methods
            ), f"Unknown crop method: {crop_method}"
        else:
            # Sample a crop method based on the given weights
            crop_method = self.random_crop_method(monomer=(len(np.unique(self.features['asym_id'])) == 1))
                
        
        # add token level ref_space_uid
        ref_space_uid_token = self.features["ref_space_uid"][:,0]
        atom_num_in_tokens = self.features["pred_dense_atom_mask"].sum(axis = -1)

        uid_num_dict = defaultdict(int)
        for idx, uid in enumerate(ref_space_uid_token):
            uid_num_dict[uid] += atom_num_in_tokens[idx]

        atom_sums = np.array([uid_num_dict[uid] for idx, uid in enumerate(ref_space_uid_token)])
        assert (atom_sums > 0).all().item(), "zero atoms"

        ref_space_uid_token = np.array(ref_space_uid_token)

        if crop_method == "ContiguousCropping":
            selected_token_indices = get_continues_crop_index(
                chain_id=chain_id,
                ref_space_uid_token=ref_space_uid_token,
                atom_sums=atom_sums,
                crop_size=self.crop_size,
                crop_complete_ligand_unstdRes=self.contiguous_crop_complete_lig,
                drop_last=self.drop_last,
                remove_metal=self.remove_metal,
            )
            reference_token_index = -1

        else:
            interface_crop = (
                True if crop_method == "SpatialInterfaceCropping" else False
            )
            token_distance = self.get_token_dist_mat(
                token_indices_in_ref=token_indices_in_ref
            )
            token_distance_mask = (
                resolved_centre_token_mask[token_indices_in_ref][:, None]
                * resolved_centre_token_mask[None, :]
            )
            selected_token_indices, reference_token_index = get_spatial_crop_index(
                chain_id=chain_id,
                token_distance=token_distance,
                token_distance_mask=token_distance_mask,
                reference_chain_id=self.ref_chain_indices,
                ref_space_uid_token=ref_space_uid_token,
                crop_size=self.crop_size,
                crop_complete_ligand_unstdRes=self.spatial_crop_complete_lig,
                interface_crop=interface_crop,
            )
        return (
            selected_token_indices,
            token_indices_in_ref[reference_token_index].item(),
            crop_method,
        )
