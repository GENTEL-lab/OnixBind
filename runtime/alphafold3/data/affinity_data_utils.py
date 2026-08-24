import ast

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

from scipy.stats import t

from alphafold3.model.pipeline.pipeline import compute_template_features
from alphafold3.structure.structure import Structure


########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@  SHARED CONSTANTS, DATACLASS AND HELPER FUNCTIONS  @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CONSTANTS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

_BUCKETS = (
    256,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
)

FEATURES = {
    "token_features": {
        "aatype": [0],                  # [num_tokens] #!
        "residue_index": [0],           # [num_tokens]
        "token_index": [0],             # [num_tokens]
        "asym_id": [0],                 # [num_tokens]
        "entity_id": [0],               # [num_tokens]
        "sym_id": [0],                  # [num_tokens]
        "is_protein": [0],              # [num_tokens]
        "is_rna": [0],                  # [num_tokens]
        "is_dna": [0],                  # [num_tokens]
        "is_ligand": [0],               # [num_tokens]
        "frame_mask": [0],              # [num_tokens]
        "atom_frame_indices": [0],      # [num_tokens, 3] #! Global Atom Index
        "pseudo_beta_mask": [0],        # [num_tokens]
        "atom_pseudo_beta_index": [0],  # [num_tokens] #! Global Atom Index
        "seq_mask": [0],                # [num_tokens]
        "per_chain_token_index": [0],   # [num_tokens]
        "pred_dense_atom_mask": [0],    # [num_tokens, 24]
        "residue_center_index": [0],    # [num_tokens] #! Start from Zero for each token
        "pocket_mask": [0],             # [num_tokens]
        # "pocket_mask_expanded": [0],    # [num_tokens]
    },
    "msa_features": {
        "msa": [1],                     # [num_alignments, num_tokens]
        "msa_mask": [1],                # [num_alignments, num_tokens]
        "deletion_mean": [0],           # [num_tokens]
        "num_alignments": [],           # []
        "profile": [0],                 # [num_tokens, num_bins]
        "deletion_matrix": [1],         # [num_alignments, num_tokens] #! To be discarded
        # "deletion_value": [1],          # [num_alignments, num_tokens] #! To be computed after cropping
        # "has_deletion": [1],            # [num_alignments, num_tokens] #! To be computed after cropping
    },
    "template_features": {
        "template_aatype": [1],         # [num_templates, num_res]
        "template_atom_mask": [1],      # [num_templates, num_res, 24]
        "template_atom_positions": [1],  # [num_templates, num_res, 24, 3]
        # "template_backbone_frame_mask": [1],  # [num_templates, num_res] #! To be computed after cropping
        # "template_pseudo_beta_mask": [1],  # [num_templates, num_res] #! To be computed after cropping
        # "template_distogram": [1, 2],   # [num_templates, num_res, num_res, num_bins] #! To be computed after cropping
        # "template_unit_vector": [1, 2], # [num_templates, num_res, num_res, 3] #! To be computed after cropping
    },
    "ref_features": {
        "ref_atom_name_chars": [0],     # [num_tokens, 24, 4]
        "ref_charge": [0],              # [num_tokens, 24]
        "ref_element": [0],             # [num_tokens, 24]
        "ref_mask": [0],                # [num_tokens, 24]
        "ref_pos": [0],                 # [num_tokens, 24, 3]
        "ref_space_uid": [0]            # [num_tokens, 24]
    },
    "bond_features": {
        "token_bonds": [0, 1],          # [num_tokens, num_bonds]
    },
    "cropped_ground_truth_features": {
        "pred_dense_atom_mask": [0],     # [num_tokens, 24]
        "resolved_atom_mask": [0],       # [num_tokens, 24]
        "atom_positions": [0],           # [num_tokens, 24, 3]
        "asym_id": [0],                  # [num_tokens]
        "entity_id": [0],                # [num_tokens]
        "per_chain_token_index": [0],    # [num_tokens]
        "residue_center_index": [0],     # [num_tokens]
        "pseudo_beta": [0],              # [num_tokens, 3]
        "pseudo_beta_mask": [0],         # [num_tokens]
        "frame": [0],                    # [num_tokens, 3, 3]
        "frame_mask": [0],               # [num_tokens]
        "resolution": [],               # []
    },
    # DO NOT CROP
    "ground_truth_features": {
        "pred_dense_atom_mask": [],     # [num_tokens, 24]
        "resolved_atom_mask": [],       # [num_tokens, 24]
        "atom_positions": [],           # [num_tokens, 24, 3]
        "asym_id": [],                  # [num_tokens]
        "entity_id": [],                # [num_tokens]
        "per_chain_token_index": [],    # [num_tokens]
        "token_index": [],              # [num_tokens]
        "residue_center_index": [],     # [num_tokens]
        "pseudo_beta": [],              # [num_tokens, 3]
        "pseudo_beta_mask": [],         # [num_tokens]
        "frame": [],                    # [num_tokens, 3, 3]
        "frame_mask": [],               # [num_tokens]
        "resolution": [],               # []
    }
}


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DATACLASS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@dataclass
class DataSetOutput:
    input_features: dict
    cropped_ground_truth_features: dict | None = None
    ground_truth_features: dict | None = None
    empty_output_struc: Structure | None = None
    cropping_indices: list | None = None
    idx: int | None = None
    prediction_seed: int | None = None
    sampled_ids: tuple[str, ...] | None = None
    file_ids: tuple[str, ...] | None = None
    assay_batch_key: str | None = None
    sampler_target_group_id: str | None = None
    sampler_standard_type: str | None = None
    sampler_label: float | None = None
    sampler_smiles: str | None = None
    binary_label: float | None = None
    binary_label_mask: bool | None = None
    regression_label_mask: bool | None = None
    binary_label_source: str | None = None
    affinity_task_type: str | None = None
    affinity_bound_type: str | None = None
    sampler_affinity_bound_type: str | None = None


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HELPER FUNCTIONS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def pad_at_dim(
    t,
    pad: Tuple[int, int],
    *,
    dim=-1,
    value=0.
):
    pad_width = [(0, 0)] * t.ndim
    pad_width[dim] = pad
    return np.pad(t, pad_width, mode='constant', constant_values=value)

def complete_features(features):
    # template features
    (
        template_unit_vector,
        template_distogram,
        template_backbone_frame_mask,
        template_pseudo_beta_mask
    ) = compute_template_features(
        features['template_aatype'],
        features['template_atom_positions'],
        features['template_atom_mask'],
        features['asym_id']
    )
    features['template_unit_vector'] = template_unit_vector
    features['template_distogram'] = template_distogram
    features['template_backbone_frame_mask'] = template_backbone_frame_mask
    features['template_pseudo_beta_mask'] = template_pseudo_beta_mask

    # Handle deletion
    deletion_matrix = features.pop('deletion_matrix')
    has_deletion = np.clip(deletion_matrix, 0.0, 1.0).astype(np.float32)
    deletion_value = (np.arctan(deletion_matrix / 3.0) * (2.0 / np.pi)).astype(np.float32)
    features['deletion_value'] = deletion_value
    features['has_deletion'] = has_deletion

    return features

def aggregate_crop_features_by_indices(features, cropping_indices, features_dicts, pad_to_length=None, crop_method=None):
      """
      Aggregate and Crop the features from a dictionary based on specified crop indices and axes.

      Args:
          features (dict): A dictionary of feature arrays.
          cropping_indices (list or ndarray): Indices specifying the crop positions.
          features_dicts (list of dict): A list of dictionaries specifying the axes to crop for each feature.
          pad_to_length (int, optional): The length to pad the cropped features to at the same axes to crop. Defaults to None.
          crop_method (str, optional): The crop method used.

      Returns:
            dict: A dictionary of cropped feature arrays.
      """
      if cropping_indices is None:
        output_features = {}
        ground_truth_features = {}
        cropped_ground_truth_features = {}
        for feat_type_key in features_dicts.keys():
            feat_dict = features_dicts[feat_type_key]
            for key, axes in feat_dict.items():
                if feat_type_key != 'ground_truth_features':
                    output_features[key] = features[key]
                else:
                    ground_truth_features[key] = features[key]
                    cropped_ground_truth_features[key] = features[key]

        output_features['entity_mol_id'] = output_features['entity_id']
        output_features['mol_id'] = output_features['asym_id']
        output_features['mol_token_index'] = output_features['per_chain_token_index']

        cropped_ground_truth_features['entity_mol_id'] = cropped_ground_truth_features['entity_id']
        cropped_ground_truth_features['mol_id'] = cropped_ground_truth_features['asym_id']
        cropped_ground_truth_features['mol_token_index'] = cropped_ground_truth_features['per_chain_token_index']

        ground_truth_features['entity_mol_id'] = ground_truth_features['entity_id']
        ground_truth_features['mol_id'] = ground_truth_features['asym_id']
        ground_truth_features['mol_token_index'] = ground_truth_features['per_chain_token_index']
      else:
        output_features = {}
        ground_truth_features = {}
        cropped_ground_truth_features = {}

        for feat_type_key in features_dicts.keys():
            feat_dict = features_dicts[feat_type_key]
            for key, axes in feat_dict.items():
                if feat_type_key != 'ground_truth_features' and feat_type_key != 'cropped_ground_truth_features':
                    if len(axes) == 0:
                        # No cropping required, directly copy the feature.
                        output_features[key] = features[key]
                    elif len(axes) == 1:
                        axis1 = axes[0]
                        # Crop along the specified axis using cropping_indices.
                        output_features[key] = features[key].take(cropping_indices, axis=axis1)
                        if pad_to_length is not None:
                            # Pad the cropped feature to the specified length. At the same axis with cropping_indices.
                            output_features[key] = pad_at_dim(output_features[key], (0, pad_to_length - len(cropping_indices)), dim=axis1)
                    elif len(axes) == 2:
                        axis1, axis2 = axes
                        # Assuming a 2D crop, apply cropping_indices along both axes.
                        cropped = features[key].take(cropping_indices, axis=axis1)
                        output_features[key] = cropped.take(cropping_indices, axis=axis2)

                        if pad_to_length is not None:
                            # Pad the cropped feature to the specified length. At the same axes with cropping_indices.
                            output_features[key] = pad_at_dim(output_features[key],(0, pad_to_length - len(cropping_indices)), dim=axis1)
                            output_features[key] = pad_at_dim(output_features[key], (0, pad_to_length - len(cropping_indices)), dim=axis2)
                    else:
                        raise ValueError(f"Unsupported number of axes for cropping: {len(axes)} in {key}")
                elif feat_type_key == 'cropped_ground_truth_features':
                    if key not in features:
                        continue

                    if len(axes) == 0:
                        cropped_ground_truth_features[key] = features[key]
                    elif len(axes) == 1:
                        axis1 = axes[0]
                        cropped_ground_truth_features[key] = features[key].take(cropping_indices, axis=axis1)
                    else:
                        raise ValueError(f"Unsupported number of axes for cropping: {len(axes)} in {key}")
                else:
                    if key not in features:
                        continue
                    if len(axes) != 0:
                        raise ValueError(f"Ground truth features should not have cropping axes: {axes} in {key}")
                    ground_truth_features[key] = features[key]


        # Special Handling of atom_frame_indices & atom_pseudo_beta_index after cropping
        uncropped_token_index = ground_truth_features['token_index']
        cropped_token_index = output_features['token_index']
        token_cropped_mask = np.isin(uncropped_token_index, cropped_token_index, assume_unique=True, invert=True)
        num_tokens_cropped = np.cumsum(token_cropped_mask, axis=0)[cropping_indices]
        if pad_to_length is not None:
            num_tokens_cropped = pad_at_dim(num_tokens_cropped, (0, pad_to_length - len(cropping_indices)), dim=0)
        output_features['atom_frame_indices'][output_features['frame_mask']] = output_features['atom_frame_indices'][output_features['frame_mask']]  - \
            num_tokens_cropped[:, None][output_features['frame_mask']] * 24

        output_features['atom_pseudo_beta_index'][output_features['pseudo_beta_mask']] =  output_features['atom_pseudo_beta_index'][output_features['pseudo_beta_mask']] - \
            num_tokens_cropped[output_features['pseudo_beta_mask']] * 24

        # The token_index should be reindexed
        output_features['token_index'] = np.arange(len(cropped_token_index))

        # entity_mol_id = entity_id, mol_id = asym_id, mol_token_index = per_chain_token_index
        output_features['entity_mol_id'] = output_features['entity_id']
        output_features['mol_id'] = output_features['asym_id']
        output_features['mol_token_index'] = output_features['per_chain_token_index']

        cropped_ground_truth_features['entity_mol_id'] = cropped_ground_truth_features['entity_id']
        cropped_ground_truth_features['mol_id'] = cropped_ground_truth_features['asym_id']
        cropped_ground_truth_features['mol_token_index'] = cropped_ground_truth_features['per_chain_token_index']

        ground_truth_features['entity_mol_id'] = ground_truth_features['entity_id']
        ground_truth_features['mol_id'] = ground_truth_features['asym_id']
        ground_truth_features['mol_token_index'] = ground_truth_features['per_chain_token_index']

        cand_crop_methods = [
            "ContiguousCropping",
            "SpatialCropping",
            "SpatialInterfaceCropping",
        ]
        if crop_method not in cand_crop_methods:
            raise ValueError(f"Unknown crop method: {crop_method}")

        if crop_method != "ContiguousCropping":
            # We only keep the mol_ids in the ground_truth_features that appear in the cropped_ground_truth_features
            uncropped_asym_ids = ground_truth_features['mol_id'] # [num_tokens]
            cropped_asym_ids = cropped_ground_truth_features['mol_id'] # [cropped_num_tokens]
            asym_id_mask = np.isin(uncropped_asym_ids, cropped_asym_ids) # [num_tokens], True indicates the asym_id is in the cropped_ground_truth_features

            for key in ground_truth_features.keys():
                # exclude resolution
                if key == "resolution":
                    continue
                elif key == "token_index":
                    ground_truth_features[key] = np.arange(len(asym_id_mask))[asym_id_mask]
                else:
                    ground_truth_features[key] = ground_truth_features[key][asym_id_mask]

      return output_features, ground_truth_features, cropped_ground_truth_features


def aggregate_crop_features_by_indices_no_tempalte(features, cropping_indices, features_dicts, pad_to_length=None, crop_method=None):
      """
      Aggregate and Crop the features from a dictionary based on specified crop indices and axes.

      Args:
          features (dict): A dictionary of feature arrays.
          cropping_indices (list or ndarray): Indices specifying the crop positions.
          features_dicts (list of dict): A list of dictionaries specifying the axes to crop for each feature.
          pad_to_length (int, optional): The length to pad the cropped features to at the same axes to crop. Defaults to None.
          crop_method (str, optional): The crop method used.

      Returns:
            dict: A dictionary of cropped feature arrays.
      """
      if cropping_indices is None:
        output_features = {}
        ground_truth_features = {}
        cropped_ground_truth_features = {}
        for feat_type_key in features_dicts.keys():
            feat_dict = features_dicts[feat_type_key]
            for key, axes in feat_dict.items():
                if feat_type_key != 'ground_truth_features':
                    output_features[key] = features[key]
                else:
                    ground_truth_features[key] = features[key]
                    cropped_ground_truth_features[key] = features[key]

        output_features['entity_mol_id'] = output_features['entity_id']
        output_features['mol_id'] = output_features['asym_id']
        output_features['mol_token_index'] = output_features['per_chain_token_index']

        cropped_ground_truth_features['entity_mol_id'] = cropped_ground_truth_features['entity_id']
        cropped_ground_truth_features['mol_id'] = cropped_ground_truth_features['asym_id']
        cropped_ground_truth_features['mol_token_index'] = cropped_ground_truth_features['per_chain_token_index']

        ground_truth_features['entity_mol_id'] = ground_truth_features['entity_id']
        ground_truth_features['mol_id'] = ground_truth_features['asym_id']
        ground_truth_features['mol_token_index'] = ground_truth_features['per_chain_token_index']
      else:
        output_features = {}
        ground_truth_features = {}
        cropped_ground_truth_features = {}

        for feat_type_key in features_dicts.keys():
            feat_dict = features_dicts[feat_type_key]
            for key, axes in feat_dict.items():
                if feat_type_key != 'ground_truth_features' and feat_type_key != 'cropped_ground_truth_features':
                    if len(axes) == 0:
                        # No cropping required, directly copy the feature.
                        output_features[key] = features[key]
                    elif len(axes) == 1:
                        axis1 = axes[0]
                        # Crop along the specified axis using cropping_indices.
                        output_features[key] = features[key].take(cropping_indices, axis=axis1)
                        if pad_to_length is not None:
                            # Pad the cropped feature to the specified length. At the same axis with cropping_indices.
                            output_features[key] = pad_at_dim(output_features[key], (0, pad_to_length - len(cropping_indices)), dim=axis1)
                    elif len(axes) == 2:
                        axis1, axis2 = axes
                        # Assuming a 2D crop, apply cropping_indices along both axes.
                        cropped = features[key].take(cropping_indices, axis=axis1)
                        output_features[key] = cropped.take(cropping_indices, axis=axis2)

                        if pad_to_length is not None:
                            # Pad the cropped feature to the specified length. At the same axes with cropping_indices.
                            output_features[key] = pad_at_dim(output_features[key],(0, pad_to_length - len(cropping_indices)), dim=axis1)
                            output_features[key] = pad_at_dim(output_features[key], (0, pad_to_length - len(cropping_indices)), dim=axis2)
                    else:
                        raise ValueError(f"Unsupported number of axes for cropping: {len(axes)} in {key}")
                # elif feat_type_key == 'cropped_ground_truth_features':
                #     if key not in features:
                #         continue

                #     if len(axes) == 0:
                #         cropped_ground_truth_features[key] = features[key]
                #     elif len(axes) == 1:
                #         axis1 = axes[0]
                #         cropped_ground_truth_features[key] = features[key].take(cropping_indices, axis=axis1)
                #     else:
                #         raise ValueError(f"Unsupported number of axes for cropping: {len(axes)} in {key}")
                # else:
                #     if key not in features:
                #         continue
                #     if len(axes) != 0:
                #         raise ValueError(f"Ground truth features should not have cropping axes: {axes} in {key}")
                #     ground_truth_features[key] = features[key]


        # Special Handling of atom_frame_indices & atom_pseudo_beta_index after cropping
        # uncropped_token_index = ground_truth_features['token_index']
        cropped_token_index = output_features['token_index']
        # token_cropped_mask = np.isin(uncropped_token_index, cropped_token_index, assume_unique=True, invert=True)
        # num_tokens_cropped = np.cumsum(token_cropped_mask, axis=0)[cropping_indices]
        # if pad_to_length is not None:
        #     num_tokens_cropped = pad_at_dim(num_tokens_cropped, (0, pad_to_length - len(cropping_indices)), dim=0)
        # output_features['atom_frame_indices'][output_features['frame_mask']] = output_features['atom_frame_indices'][output_features['frame_mask']]  - \
        #     num_tokens_cropped[:, None][output_features['frame_mask']] * 24

        # output_features['atom_pseudo_beta_index'][output_features['pseudo_beta_mask']] =  output_features['atom_pseudo_beta_index'][output_features['pseudo_beta_mask']] - \
        #     num_tokens_cropped[output_features['pseudo_beta_mask']] * 24

        # The token_index should be reindexed
        output_features['token_index'] = np.arange(len(cropped_token_index))

        # entity_mol_id = entity_id, mol_id = asym_id, mol_token_index = per_chain_token_index
        output_features['entity_mol_id'] = output_features['entity_id']
        output_features['mol_id'] = output_features['asym_id']
        output_features['mol_token_index'] = output_features['per_chain_token_index']

        # cropped_ground_truth_features['entity_mol_id'] = cropped_ground_truth_features['entity_id']
        # cropped_ground_truth_features['mol_id'] = cropped_ground_truth_features['asym_id']
        # cropped_ground_truth_features['mol_token_index'] = cropped_ground_truth_features['per_chain_token_index']

        # ground_truth_features['entity_mol_id'] = ground_truth_features['entity_id']
        # ground_truth_features['mol_id'] = ground_truth_features['asym_id']
        # ground_truth_features['mol_token_index'] = ground_truth_features['per_chain_token_index']

        cand_crop_methods = [
            "ContiguousCropping",
            "SpatialCropping",
            "SpatialInterfaceCropping",
        ]
        if crop_method not in cand_crop_methods:
            raise ValueError(f"Unknown crop method: {crop_method}")

        if crop_method != "ContiguousCropping":
            # We only keep the mol_ids in the ground_truth_features that appear in the cropped_ground_truth_features
            uncropped_asym_ids = ground_truth_features['mol_id'] # [num_tokens]
            cropped_asym_ids = cropped_ground_truth_features['mol_id'] # [cropped_num_tokens]
            asym_id_mask = np.isin(uncropped_asym_ids, cropped_asym_ids) # [num_tokens], True indicates the asym_id is in the cropped_ground_truth_features

            for key in ground_truth_features.keys():
                # exclude resolution
                if key == "resolution":
                    continue
                elif key == "token_index":
                    ground_truth_features[key] = np.arange(len(asym_id_mask))[asym_id_mask]
                else:
                    ground_truth_features[key] = ground_truth_features[key][asym_id_mask]

      return output_features, ground_truth_features, cropped_ground_truth_features


def t_weighted_sampling_no_replacement_optimized(data, column='REG_LABEL', n_samples=1000, bins=20, df=10):
    data = data.copy()
    bin_edges = np.histogram_bin_edges(data[column], bins=bins)
    data['bins'] = np.digitize(data[column], bins=bin_edges, right=False) - 1
    bin_counts = data['bins'].value_counts().sort_index()

    valid_bins = bin_counts.index[bin_counts.index < len(bin_edges) - 1]
    bin_counts = bin_counts[valid_bins].values
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid_bin_centers = bin_centers[valid_bins]

    normalized_centers = 6 * (valid_bin_centers - valid_bin_centers.min()) / (valid_bin_centers.max() - valid_bin_centers.min()) - 3
    t_weights = t.pdf(normalized_centers, df=df)
    t_weights /= t_weights.sum()

    hist_weights = bin_counts / bin_counts.sum()

    combined_weights = t_weights * hist_weights

    left_tail = np.arange(len(combined_weights))[:len(combined_weights) // 3]
    right_tail = np.arange(len(combined_weights))[-len(combined_weights) // 3:]

    combined_weights[left_tail] *= 2.0
    combined_weights[right_tail] *= 2.0
    combined_weights /= combined_weights.sum()

    target_samples = (combined_weights * n_samples).astype(int)
    target_samples = np.minimum(target_samples, bin_counts)
    remaining_samples = n_samples - target_samples.sum()


    if remaining_samples > 0:
        available_samples = bin_counts - target_samples
        additional_weights = combined_weights * (available_samples > 0)
        additional_weights /= additional_weights.sum()
        extra_samples = np.floor(additional_weights * remaining_samples).astype(int)
        target_samples += np.minimum(extra_samples, available_samples)

    sampled_data = []
    for bin_index, n_samples_in_bin in zip(valid_bins, target_samples):
        if n_samples_in_bin > 0:
            bin_data = data[data['bins'] == bin_index]
            sampled = bin_data.sample(n=int(n_samples_in_bin), replace=False)
            sampled_data.append(sampled)

    sampled_data = pd.concat(sampled_data)

    if len(sampled_data) < n_samples:
        extra_samples = sampled_data.sample(n=n_samples - len(sampled_data), replace=True)
        sampled_data = pd.concat([sampled_data, extra_samples])
    elif len(sampled_data) > n_samples:
        sampled_data = sampled_data.sample(n=n_samples, replace=False)

    return sampled_data


def normalize_pocket_index(pocket_index, protein_indices):
    """Parse and validate pocket_index from various input formats."""
    protein_indices = np.asarray(protein_indices, dtype=np.int64)
    if protein_indices.size == 0:
        return np.array([], dtype=np.int32)

    protein_set = set(protein_indices.tolist())

    if pocket_index is None:
        return np.array([], dtype=np.int32)

    if isinstance(pocket_index, np.ndarray):
        values = pocket_index.tolist()
    elif isinstance(pocket_index, (list, tuple, set)):
        values = list(pocket_index)
    elif isinstance(pocket_index, str):
        text = pocket_index.strip()
        if not text:
            return np.array([], dtype=np.int32)
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return np.array([], dtype=np.int32)
        if isinstance(parsed, (list, tuple, set)):
            values = list(parsed)
        else:
            return np.array([], dtype=np.int32)
    else:
        try:
            values = list(pocket_index)
        except Exception:
            return np.array([], dtype=np.int32)

    valid = sorted(int(v) for v in values if int(v) in protein_set)
    return np.array(valid, dtype=np.int32)


def pocket_crop_expand(input_features, pocket_index, token_crop_size, max_extend=10):
    """Keep every ligand token and fill the remaining budget around the pocket."""
    ligand_indices = np.flatnonzero(input_features['is_ligand'])
    protein_indices = np.flatnonzero(input_features['is_protein'])
    protein_index_set = set(protein_indices.tolist())
    pocket_indices = np.unique(np.asarray(pocket_index, dtype=np.int64))
    pocket_indices = pocket_indices[np.isin(pocket_indices, protein_indices)]

    if token_crop_size is None:
        protein_budget = len(protein_indices)
    else:
        if token_crop_size < 0:
            raise ValueError(f"token_crop_size must be non-negative, got {token_crop_size}")
        if len(ligand_indices) > token_crop_size:
            raise ValueError(
                f"ligand has {len(ligand_indices)} tokens, exceeding "
                f"token_crop_size={token_crop_size}"
            )
        protein_budget = token_crop_size - len(ligand_indices)

    if len(pocket_indices) > protein_budget:
        positions = np.linspace(
            0, len(pocket_indices) - 1, num=protein_budget, dtype=np.int64
        )
        pocket_indices = pocket_indices[positions]

    selected_protein = set(pocket_indices.tolist())
    for offset in range(1, max_extend + 1):
        for pocket_idx in pocket_indices:
            for candidate in (pocket_idx - offset, pocket_idx + offset):
                if candidate in protein_index_set:
                    selected_protein.add(candidate)
        if len(selected_protein) >= protein_budget:
            break

    if len(selected_protein) > protein_budget:
        selected_protein = set(
            sorted(
                selected_protein,
                key=lambda index: (
                    min(abs(index - pocket_idx) for pocket_idx in pocket_indices),
                    index,
                ),
            )[:protein_budget]
        )

    chosen = np.asarray(
        sorted(set(ligand_indices.tolist()).union(selected_protein)),
        dtype=np.int32,
    )
    if token_crop_size is not None and len(chosen) > token_crop_size:
        raise AssertionError("pocket crop exceeded its token budget")
    return chosen


def pocket_crop_expandV2(input_features, pocket_index, token_crop_size, max_extend=10):
    """
    input_features: dict, 包含 'is_ligand', 'is_protein' 等 mask
    pocket_index: list or array, pocket 残基的索引
    token_crop_size: int or None.
                     - 如果为 None, chosen 不进行截断，保留所有扩展节点。
                     - 如果为 int, chosen 按距离由近及远截断到该长度。
    max_extend: int, 最大向外扩展的残基数 (默认10)

    Returns:
    1. chosen (indices array): 最终选中的索引数组。
    2. pocket_mask (binary array): 全局掩码，标记 Ligand + Pocket + 完整 max_extend 范围。
    """
    ligand_mask = input_features['is_ligand']
    protein_mask = input_features['is_protein']
    total_len = len(ligand_mask)

    ligand_indices = np.where(ligand_mask)[0]
    protein_indices = np.where(protein_mask)[0]
    protein_set = set(protein_indices.tolist())

    # --- 1. 确定基础集合 (Ligand + Pocket) ---
    pocket_indices = np.unique(pocket_index)
    pocket_indices = pocket_indices[np.isin(pocket_indices, protein_indices)]

    base_indices = set(ligand_indices.tolist()) | set(pocket_indices.tolist())

    # 初始化两个集合
    # mask_set: 用于生成 pocket_mask (无限制扩展)
    # chosen_set: 用于生成返回值 chosen (受 token_crop_size 限制)
    mask_set = base_indices.copy()
    chosen_set = base_indices.copy()

    # 标记 chosen 是否已填满 (如果 crop_size 为 None，则永远不满)
    chosen_full = False

    # 处理初始 base 集合就超过 crop_size 的情况
    if token_crop_size is not None:
        if len(chosen_set) >= token_crop_size:
            # 即使截断，mask_set 也要继续完整计算，所以只截断 chosen_set 并标记已满
            temp_sorted = sorted(list(chosen_set))
            chosen_set = set(temp_sorted[:token_crop_size])
            chosen_full = True

    # --- 2. 逐层向外扩展 ---
    for offset in range(1, max_extend + 1):
        # 寻找这一层的新邻居
        candidates = set()
        for idx in pocket_indices:
            left, right = idx - offset, idx + offset
            if left in protein_set:
                candidates.add(left)
            if right in protein_set:
                candidates.add(right)

        # 只处理那些还未加入 mask_set 的新节点
        # (mask_set 是全集，用来判断是否是"新"节点最准确)
        new_nodes = candidates - mask_set

        if not new_nodes:
            continue

        # A. 更新 mask_set (始终无条件加入)
        mask_set.update(new_nodes)

        # B. 更新 chosen_set (根据 crop_size 决定)
        if token_crop_size is None:
            # 如果无限制，chosen 跟随 mask 一起扩展
            chosen_set.update(new_nodes)
        elif not chosen_full:
            # 如果有限制且没满
            needed = len(new_nodes)
            remaining = token_crop_size - len(chosen_set)

            if remaining >= needed:
                # 空间足够，全加
                chosen_set.update(new_nodes)
            else:
                # 空间不足，填满为止
                sorted_nodes = sorted(list(new_nodes))
                chosen_set.update(sorted_nodes[:remaining])
                chosen_full = True

    # --- 3. 生成输出 ---

    # 生成 chosen 数组 (排序)
    chosen = np.array(sorted(chosen_set), dtype=np.int32)

    # 生成 pocket_mask 数组 (0/1 掩码)
    pocket_mask = np.zeros(total_len, dtype=np.int32)
    pocket_mask[list(mask_set)] = 1

    return chosen, pocket_mask


def pocket_crop_random_expand(input_features, pocket_index, token_crop_size, max_extend=10):
    """
    Keep all ligand tokens, keep all pocket residues, and randomly expand each
    pocket residue with a radius sampled from [0, max_extend].
    Deduplicate and sort automatically. Final length does not exceed token_crop_size.
    """
    ligand_mask = input_features["is_ligand"]
    protein_mask = input_features["is_protein"]

    ligand_indices = np.where(ligand_mask)[0]
    protein_indices = np.where(protein_mask)[0]
    protein_index_set = set(protein_indices.tolist())

    # Keep valid pocket residues only.
    pocket_indices = np.sort(pocket_index)
    pocket_indices = pocket_indices[np.isin(pocket_indices, protein_indices)]

    chosen = set(ligand_indices.tolist()) | set(pocket_indices.tolist())

    # Randomly expand each pocket residue with an independent radius.
    for idx in pocket_indices:
        radius = np.random.randint(0, max_extend + 1)
        for offset in range(1, radius + 1):
            left, right = idx - offset, idx + offset
            if left in protein_index_set:
                chosen.add(left)
            if right in protein_index_set:
                chosen.add(right)

    chosen = np.array(sorted(chosen), dtype=np.int32)

    # Crop if the total length exceeds token_crop_size.
    if token_crop_size is not None and len(chosen) > token_crop_size:
        must_keep_size = len(ligand_indices) + len(pocket_indices)

        if must_keep_size > token_crop_size:
            available_for_pocket = token_crop_size - len(ligand_indices)

            if available_for_pocket <= 0:
                if len(ligand_indices) > token_crop_size:
                    chosen = ligand_indices[:token_crop_size]
                else:
                    chosen = ligand_indices
            elif len(pocket_indices) > available_for_pocket:
                max_start = len(pocket_indices) - available_for_pocket
                start_idx = np.random.randint(0, max_start + 1)
                pocket_indices_cropped = pocket_indices[start_idx:start_idx + available_for_pocket]
                chosen = np.array(
                    sorted(set(ligand_indices.tolist()) | set(pocket_indices_cropped.tolist())),
                    dtype=np.int32,
                )
            else:
                chosen = np.array(
                    sorted(set(ligand_indices.tolist()) | set(pocket_indices.tolist())),
                    dtype=np.int32,
                )
        else:
            must_keep = set(ligand_indices.tolist()) | set(pocket_indices.tolist())
            must_keep = np.array(sorted(must_keep), dtype=np.int32)

            final = list(must_keep)
            extra = list(set(chosen) - set(must_keep))
            extra_sorted = sorted(extra, key=lambda x: min(abs(x - p) for p in pocket_indices))

            allowed_extra = token_crop_size - len(final)
            final.extend(extra_sorted[:allowed_extra])

            chosen = np.array(sorted(final), dtype=np.int32)

    return chosen


def pocket_crop_distance_weighted(input_features, pocket_index, token_crop_size, sigma=10.0):
    """
    Keep all ligand + pocket tokens, then contiguously expand from the pocket
    boundary along the protein sequence to fill remaining slots.

    Left/right expansion ratio is randomised (controlled by sigma — higher
    sigma means more uniform split, lower sigma means more random bias to
    one side).  This gives crop-level data augmentation while keeping the
    expanded region spatially coherent in sequence space.

    Args:
        input_features: dict with 'is_ligand', 'is_protein', 'seq_mask'
        pocket_index: ndarray of pocket token indices
        token_crop_size: max tokens (None = keep all)
        sigma: controls randomness of left/right split.
               Can be a scalar (fixed) or a tuple/list (low, high) — when a
               range is given, a sigma is sampled uniformly per call.
               Higher = more even, lower = more biased.

    Returns:
        sorted ndarray of selected token indices
    """
    # Support dynamic sigma: if (low, high) range given, sample per call
    if isinstance(sigma, (tuple, list)):
        if len(sigma) >= 2:
            sigma = np.random.uniform(sigma[0], sigma[1])
        else:
            sigma = sigma[0]

    ligand_mask = input_features["is_ligand"]
    protein_mask = input_features["is_protein"]

    ligand_indices = np.where(ligand_mask)[0]
    protein_indices = np.where(protein_mask)[0]

    pocket_indices = np.sort(pocket_index)
    pocket_indices = pocket_indices[np.isin(pocket_indices, protein_indices)]

    must_keep = set(ligand_indices.tolist()) | set(pocket_indices.tolist())

    if token_crop_size is None or len(must_keep) >= token_crop_size:
        if token_crop_size is not None and len(must_keep) > token_crop_size:
            available_for_pocket = token_crop_size - len(ligand_indices)
            if available_for_pocket <= 0:
                chosen = ligand_indices[:token_crop_size] if len(ligand_indices) > token_crop_size else ligand_indices
            elif len(pocket_indices) > available_for_pocket:
                max_start = len(pocket_indices) - available_for_pocket
                start_idx = np.random.randint(0, max_start + 1)
                pocket_cropped = pocket_indices[start_idx:start_idx + available_for_pocket]
                chosen = sorted(set(ligand_indices.tolist()) | set(pocket_cropped.tolist()))
            else:
                chosen = sorted(must_keep)
            return np.array(chosen, dtype=np.int32)
        return np.array(sorted(must_keep), dtype=np.int32)

    # Build ordered protein tokens left/right of pocket for contiguous expansion
    pocket_set = set(pocket_indices.tolist())
    protein_list = protein_indices.tolist()

    pocket_min = int(pocket_indices.min())
    pocket_max = int(pocket_indices.max())

    # Protein tokens to the left of pocket (reverse order: nearest first)
    left_candidates = [idx for idx in protein_list if idx < pocket_min and idx not in pocket_set]
    left_candidates = list(reversed(left_candidates))

    # Protein tokens to the right of pocket (nearest first)
    right_candidates = [idx for idx in protein_list if idx > pocket_max and idx not in pocket_set]

    remaining_slots = token_crop_size - len(must_keep)

    if len(left_candidates) + len(right_candidates) == 0:
        return np.array(sorted(must_keep), dtype=np.int32)

    n_expand = min(remaining_slots, len(left_candidates) + len(right_candidates))

    # Randomly decide left/right split ratio
    max_left = min(n_expand, len(left_candidates))
    max_right = min(n_expand, len(right_candidates))

    # Sample a left fraction from a clipped Gaussian centered at 0.5
    left_frac = np.clip(np.random.normal(0.5, 1.0 / sigma), 0.0, 1.0)
    n_left = int(round(left_frac * n_expand))
    n_left = min(n_left, max_left)
    n_right = min(n_expand - n_left, max_right)
    # Re-allocate leftover to the other side
    n_left = min(n_expand - n_right, max_left)

    chosen = must_keep | set(left_candidates[:n_left]) | set(right_candidates[:n_right])
    return np.array(sorted(chosen), dtype=np.int32)
