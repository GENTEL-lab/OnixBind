import os
import sys
sys.path.insert(0, './')
import random
from torch.utils.data import DataLoader, Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from alphafold3.common import folding_input
import numpy as np
import json
from typing import Tuple
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.model.pipeline.pipeline import WholePdbPipeline
from alphafold3.model.pipeline.pipeline import compute_template_features
from alphafold3.structure.structure import Structure
from dataclasses import dataclass
import datetime
import torch
import re
import tqdm
########################################################
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#@                                                    @#
#@  BASIC DATA MODULE                                 @#
#@                                                    @#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
########################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# HELPER FUNCTIONS AND CONSTANTS
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

def add_padding(cropped_ground_truth_features, pad_to_length):
    """Add padding to features up to specified length.
    
    Args:
        cropped_ground_truth_features (dict): Dictionary of cropped ground truth features without padding (torch tensors)
        pad_to_length (int): Length to pad features to
        
    Returns:
        dict: cropped_ground_truth_features with padding added
    """
    curr_length = cropped_ground_truth_features['asym_id'].shape[0]
    pad_size = pad_to_length - curr_length
    
    if pad_size <= 0:
        return cropped_ground_truth_features
        
    # Add padding to cropped ground truth features
    for key in cropped_ground_truth_features.keys():
        if key != 'resolution':
            pad_shape = (0, 0) * (len(cropped_ground_truth_features[key].shape) - 1) + (0, pad_size)
            cropped_ground_truth_features[key] = torch.nn.functional.pad(
                cropped_ground_truth_features[key], 
                pad_shape, 
                mode='constant', 
                value=0
            )
                
    return cropped_ground_truth_features

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
                    if len(axes) == 0:
                        cropped_ground_truth_features[key] = features[key]
                    elif len(axes) == 1:
                        axis1 = axes[0]
                        cropped_ground_truth_features[key] = features[key].take(cropping_indices, axis=axis1)
                    else:
                        raise ValueError(f"Unsupported number of axes for cropping: {len(axes)} in {key}")
                else:
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAIN CLASS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class AlphaFoldDataSet(Dataset):
    def __init__(self,
                 mmcif_path,
                 alignment_path,
                 seq_to_msa_mapping_path,
                 use_templates=False,
                 template_path=None,
                 seq_to_template_mapping_path=None,
                 template_mmcif_path=None,
                 metadata_path=None,
                 valid_chains_path=None,
                 token_crop_size=384, 
                 msa_crop_size=16384, 
                 epoch_len=10000,
                 evaluate_epoch_len=None,
                 sampler=None,
                 sample_targets_path=None,
                 input_json_dir=None,
                 spatial_crop_ratio=0.4,
                 spatial_inter_crop_ratio=0.4,
                 fix_size=True,
                 deterministic_frames=False,
                 use_ideal_ref_coords=False,
                 check_release_date=False,
                 distillation_sampling_prob=0.0,
                 distillation_mmcif_path=None,
                 distillation_sampled_targets_path=None,
                 distillation_alignment_path=None,
                 distillation_seq_to_msa_mapping_path=None,
                 distillation_maskout_low_plddt=False,
                 distillation_plddt_threshold=80.0,
                 recropping_tolerance=10,
                 only_sampled_chains=False,
                 error_dir=None,
                 only_protein_chains=False,
                 prediction_seeds=[42,43,44,45,46],
                 output_dir=None,
                 mode="train"):
        """Initialize the AlphaFoldDataSet.

        Args:
            mmcif_path (str): The path to the directory containing the mmcif files.
            alignment_path (str): The path to the directory containing the alignment files.
            seq_to_msa_mapping_path (str): The path to the sequence to MSA mapping file.
            metadata_path (str): The path to the meta data file of cif. (resolution, release date, etc.)
            token_crop_size (int, optional): The size of the cropping window at token level. Defaults to 384.
            msa_crop_size (int, optional): The maximum depth of the MSA. Defaults to 16384.
            epoch_len (int, optional): Virtual epoch length. Defaults to 10000.
            evaluate_epoch_len (int, optional): Virtual epoch length for evaluation. Defaults to 1000.
            sampler (_type_, optional): The sampler to use, in training mode. Defaults to None.
            sample_targets_path (_type_, optional): The cache to use, pointing to the sampled mmcifs use in evaluation. Defaults to None.
            input_json_dir (_type_, optional): The directory containing the input json files, use in prediction mode. Defaults to None.
            spatial_crop_ratio (float, optional): The probability of spatial cropping. Defaults to 0.4.
            spatial_inter_crop_ratio (float, optional):  The probability of spatial intercropping. Defaults to 0.4.
            deterministic_frames (bool, optional): Whether to use deterministic frames. Defaults to False.
            use_ideal_ref_coords (bool, optional): Whether to use ideal reference coordinates. Defaults to False.
            check_release_date (bool, optional): Whether to check the release date. Defaults to False.
            recropping_tolerance (int, optional): The tolerance of recropping. Defaults to 10.
        """
        self.token_crop_size = token_crop_size
        self.msa_crop_size = msa_crop_size
        self.check_release_date = check_release_date
        self.mmcif_path = mmcif_path
        self.alignment_path = alignment_path
        self.seq_to_msa_mapping_path = seq_to_msa_mapping_path
        
        self.use_templates = use_templates
        self.template_path = template_path
        self.seq_to_template_mapping_path = seq_to_template_mapping_path
        self.template_mmcif_path = template_mmcif_path
        
        self.spatial_inter_crop_ratio = spatial_inter_crop_ratio
        self.spatial_crop_ratio = spatial_crop_ratio
        self.contiguous_crop_ratio = 1.0 - spatial_crop_ratio - spatial_inter_crop_ratio

        self.fix_size = fix_size
        self.deterministic_frames = deterministic_frames
        self.use_ideal_ref_coords = use_ideal_ref_coords
        self.recropping_tolerance = recropping_tolerance
        self.only_sampled_chains = only_sampled_chains
        self.only_protein_chains = only_protein_chains
        self.distillation_sampling_prob=distillation_sampling_prob
        self.distillation_mmcif_path=distillation_mmcif_path
        self.distillation_alignment_path=distillation_alignment_path
        self.distillation_seq_to_msa_mapping_path=distillation_seq_to_msa_mapping_path
        self.distillation_maskout_low_plddt = distillation_maskout_low_plddt
        self.distillation_plddt_threshold = distillation_plddt_threshold
        
        self.mode = mode
        self.prediction_seeds = prediction_seeds
        self.error_dir = error_dir

        if self.mode == 'train':
            if sampler is None:
                raise ValueError(f"Sampler is required in training mode.")
            self.sampler = sampler
            self.epoch_len = epoch_len
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                
            if valid_chains_path is not None:
                with open(valid_chains_path, 'r') as f:
                    self.valid_chains = json.load(f)
            else:
                self.valid_chains = None
                
            if self.distillation_sampling_prob == 0:
                self.sampled_targets = self.sampler.sample(num_samples=epoch_len)
                random.shuffle(self.sampled_targets)
            else:
                with open(distillation_sampled_targets_path, 'r') as f:
                    self.distillation_targets = json.load(f)
                num_pdb_samples = int((1 - self.distillation_sampling_prob) * epoch_len)
                num_distillation_samples = epoch_len - num_pdb_samples
                pdb_sampled_targets = self.sampler.sample(num_samples=num_pdb_samples)
                distillation_sampled_targets = random.sample(self.distillation_targets, num_distillation_samples)
                self.sampled_targets = pdb_sampled_targets + distillation_sampled_targets
                random.shuffle(self.sampled_targets)
                
        elif self.mode == 'eval':
            with open(sample_targets_path, 'r') as f:
                self.sampled_targets = json.load(f)
            if evaluate_epoch_len is not None:
                self.sampled_targets = self.sampled_targets[:evaluate_epoch_len]
            self.epoch_len = len(self.sampled_targets)
            
        elif self.mode == 'predict':
            if input_json_dir is not None:
                self.input_json_dir = input_json_dir
                # self.sampled_targets = [f.split('.')[0] for f in os.listdir(input_json_dir) if f.endswith('.json')]
                self.sampled_targets = [f.replace('.json', '') for f in os.listdir(input_json_dir) if f.endswith('.json')]
                self.epoch_len = len(self.sampled_targets) * len(self.prediction_seeds)
            else:
                self.input_json_dir = None
                with open(sample_targets_path, 'r') as f:
                    self.sampled_targets = json.load(f)
                self.epoch_len = len(self.sampled_targets) * len(self.prediction_seeds)
            
            # if output_dir is not None:
            #     remained = []
            #     for t in self.sampled_targets:
            #         folder_path = os.path.join(output_dir, t)
            #         if not os.path.exists(folder_path):
            #             remained.append(t)
            #     self.sampled_targets = remained
            #     self.epoch_len = len(self.sampled_targets) * len(self.prediction_seeds)
                # breakpoint()
            
        else:
            raise ValueError(f"Invalid mode: {mode}.")
        
        
        data_pipeline_config = pipeline.DataPipelineConfig(
            use_precomputed_alignments=True,
            precomputed_alignments_path=self.alignment_path,
            sequence_to_precomputed_alignment_id_mapping_path=self.seq_to_msa_mapping_path,
            use_templates=self.use_templates,
            precomputed_templates_path=self.template_path,
            sequence_to_precomputed_template_id_mapping_path=self.seq_to_template_mapping_path,
            pdb_database_path=self.template_mmcif_path,
        )
        self.data_pipepine = pipeline.DataPipeline(data_pipeline_config)
        
        if self.distillation_sampling_prob > 0:
            distillation_data_pipeline_config = pipeline.DataPipelineConfig(
                use_precomputed_alignments=True,
                precomputed_alignments_path=self.distillation_alignment_path,
                sequence_to_precomputed_alignment_id_mapping_path=self.distillation_seq_to_msa_mapping_path,
                use_templates=False,
            )
            self.distillation_data_pipepine = pipeline.DataPipeline(distillation_data_pipeline_config)
            
        self.whole_pdb_config = WholePdbPipeline.Config(
                token_crop_size=self.token_crop_size,
                msa_crop_size=self.msa_crop_size,
                # 2021 09 30
                buckets=_BUCKETS if self.mode != 'train' else None,
                max_template_date=datetime.date(2021, 9, 30),
                deterministic_frames=self.deterministic_frames,
                use_ideal_ref_coords=self.use_ideal_ref_coords,
                distillation_plddt_threshold=self.distillation_plddt_threshold,
            )
        self.ccd_dict = chemical_components.cached_ccd()

    
    def resample(self):
        """
        Resample the targets for the dataset. This should be called at the start of each epoch.
        """
        if self.distillation_sampling_prob == 0:
            self.sampled_targets = self.sampler.sample(num_samples=self.epoch_len)
            random.shuffle(self.sampled_targets)
        else:
            num_pdb_samples = int((1 - self.distillation_sampling_prob) * self.epoch_len)
            num_distillation_samples = self.epoch_len - num_pdb_samples
            pdb_sampled_targets = self.sampler.sample(num_samples=num_pdb_samples)
            distillation_sampled_targets = random.sample(self.distillation_targets, num_distillation_samples)
            self.sampled_targets = pdb_sampled_targets + distillation_sampled_targets
            random.shuffle(self.sampled_targets)
        
    def __getitem__(self, idx, recropping_times=0):
        """Get the item at the specified index. (The index are fake, we will sample the data randomly)

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the features of the item.
        """    
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # SAMPLE AN ITEM
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mode == 'train':
            if isinstance(self.sampled_targets[idx], tuple):
                pdb_id, chain_id_1, chain_id_2 = self.sampled_targets[idx]
                chain_id_1 = chain_id_1 or None
                chain_id_2 = chain_id_2 or None
            else:
                # distillation
                pdb_id =  self.sampled_targets[idx]
                chain_id_1 = None
                chain_id_2 = None
        elif self.mode == 'eval':
            pdb_id = self.sampled_targets[idx]
            chain_id_1 = None
            chain_id_2 = None
        elif self.mode == 'predict':
            pdb_id = self.sampled_targets[idx // len(self.prediction_seeds)]
            chain_id_1 = None
            chain_id_2 = None

        # Filter out None values in sampled_ids
        sampled_ids = [id for id in [pdb_id, chain_id_1, chain_id_2] if id is not None]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GET THE FEATURES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        try:
            if self.mode == 'predict':
                dataset_output = self.process(sampled_ids, self.prediction_seeds[idx % len(self.prediction_seeds)])
            else:
                dataset_output = self.process(sampled_ids)
            
            if self.mode == 'train':
                if dataset_output.input_features['is_ligand'].sum() == 0:
                    return self.__getitem__(idx, recropping_times + 1)
                if dataset_output.input_features['seq_mask'].sum() < 4 or ((dataset_output.cropped_ground_truth_features['resolved_atom_mask']).sum(axis = -1) != 0).sum() < 4:
                    if recropping_times < self.recropping_tolerance:
                        return self.__getitem__(idx, recropping_times + 1)
                    else:
                        raise ValueError(f"Too few resolved residues! pdb_id: {pdb_id}, chain_id_1: {chain_id_1}, chain_id_2: {chain_id_2}")
            else:
                if dataset_output.cropping_indices is not None:
                    raise ValueError("Cropping indices should be None in evaluation mode.")

        except Exception as e:
            print(f"Error in processing the data! Error type: {type(e)} Error Msg: {e}, pdb_id: {pdb_id}, chain_id_1: {chain_id_1}, chain_id_2: {chain_id_2}")
            if self.error_dir is not None:
                with open(os.path.join(self.error_dir, "error.txt"), 'a') as f:
                    f.write(f"Error in processing the data! Error type: {type(e)} Error Msg: {e}, pdb_id: {pdb_id}, chain_id_1: {chain_id_1}, chain_id_2: {chain_id_2}\n")
            random_idx = random.randint(0, self.epoch_len - 1)
            return self.__getitem__(random_idx)
        
        # if self.mode == 'predict':
        #     dataset_output = self.process(sampled_ids, self.prediction_seeds[idx % len(self.prediction_seeds)])
        # else:
        #     dataset_output = self.process(sampled_ids)
            
        
                
        dataset_output.idx = idx
        dataset_output.sampled_ids = self.idx_to_sampled_target(idx)
        
        return dataset_output

    def __len__(self):
        return self.epoch_len

    def idx_to_sampled_target(self, idx):
        if self.mode == 'predict':
            return self.sampled_targets[idx // len(self.prediction_seeds)]
        else:
            return self.sampled_targets[idx]
    
    def process(self, sampled_ids, prediction_seed=None):
        """Given the sampled pdb_id and chain_id(s), process the data and return the feature dictionary.

        Args:
            sampled_ids (List): A list containing the pdb_id and the chain_id(s) of the complex. The length of the list is 1, 2 or 3, depending on the sampling method used. 
                                   When a chain is sampled, the list will contain only two element. When an interface is sampled, the list will contain three elements.
                                   When the entire complex is sampled, the list will contain only one element.

        Returns:
            dict: A dictionary containing the features of the complex.
        """        
        

        if self.mode == "predict":

            if self.input_json_dir is not None:
                fold_input = folding_input.load_fold_input_from_json_path(os.path.join(self.input_json_dir, f"{sampled_ids[0]}.json"), only_protein_chains=self.only_protein_chains)
                atoms_iter = None
                resolution = None
                release_date = None
                filtered_struc = None
                sampled_chain_ids = []
                distillation = False
            else:
                sampled_pdb_id = sampled_ids[0]
                mmcif_file_path = os.path.join(self.mmcif_path, f"{sampled_pdb_id}.cif")
                if not os.path.exists(mmcif_file_path):
                    sub_dir = sampled_pdb_id[1:3]
                    mmcif_file_path = os.path.join(self.mmcif_path, sub_dir, f"{sampled_pdb_id}.cif")
                    
                atoms_iter = None
                resolution = None
                release_date = None
                sampled_chain_ids = ['A']
                self.only_sampled_chains = True
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # LOAD THE RAW INPUT
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                _,_,filtered_struc, fold_input = folding_input.load_fold_input_from_mmcif_path(
                    mmcif_file_path, self.ccd_dict, self.whole_pdb_config, sampled_chains=sampled_chain_ids, only_sampled_chains=self.only_sampled_chains, only_protein_chains=self.only_protein_chains
                )
                distillation = False
            
        elif self.mode == "eval":
            sampled_pdb_id = sampled_ids[0]
            mmcif_file_path = os.path.join(self.mmcif_path, f"{sampled_pdb_id}.cif")
            if not os.path.exists(mmcif_file_path):
                sub_dir = sampled_pdb_id[1:3]
                mmcif_file_path = os.path.join(self.mmcif_path, sub_dir, f"{sampled_pdb_id}.cif")
            sampled_chain_ids = []
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # LOAD THE RAW INPUT
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            atoms_iter, resolution,filtered_struc, fold_input = folding_input.load_fold_input_from_mmcif_path(
                    mmcif_file_path, self.ccd_dict, self.whole_pdb_config, sampled_chains=sampled_chain_ids, only_sampled_chains=self.only_sampled_chains, only_protein_chains=self.only_protein_chains
                )
            release_date = None
            distillation = False
            
        else:
            sampled_pdb_id = sampled_ids[0]
            sampled_chain_ids = sampled_ids[1:]
            if sampled_chain_ids:
                sub_dir = sampled_pdb_id[1:3]
                if self.valid_chains is not None:
                    try:
                        valid_chains = self.valid_chains[sampled_pdb_id[:4].lower()]
                    except:
                        valid_chains = self.valid_chains[sampled_pdb_id[:4].upper()]
                    if type(valid_chains) == str:
                        valid_chains = valid_chains.split(',')
                else:
                    valid_chains = None
                    
                mmcif_file_path = os.path.join(self.mmcif_path,sub_dir, f"{sampled_pdb_id}.cif")
                
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # LOAD THE RAW INPUT
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                atoms_iter, resolution, filtered_struc, fold_input = folding_input.load_fold_input_from_mmcif_path(
                    mmcif_file_path, self.ccd_dict, self.whole_pdb_config, sampled_chains=sampled_chain_ids, only_sampled_chains=self.only_sampled_chains, only_protein_chains=self.only_protein_chains,valid_chains = valid_chains
                )
                distillation = False
                if resolution is None:
                    resolution = self.metadata[re.match(r'^([a-zA-Z0-9]+)', sampled_pdb_id).group(1)]['resolution']
                release_date = self.metadata[re.match(r'^([a-zA-Z0-9]+)', sampled_pdb_id).group(1)]['release_date']
            else:
                # distillation
                sampled_chain_ids=['A']
                mmcif_file_path = os.path.join(self.distillation_mmcif_path, f"{sampled_pdb_id}.cif")
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # LOAD THE RAW INPUT
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                atoms_iter, resolution, filtered_struc, fold_input = folding_input.load_fold_input_from_mmcif_path(
                    mmcif_file_path, self.ccd_dict, self.whole_pdb_config, sampled_chains=sampled_chain_ids, only_sampled_chains=self.only_sampled_chains, only_protein_chains=self.only_protein_chains
                )
                distillation = True
                resolution=None
                release_date = None
                
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # LOAD THE ALIGNMENT
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        if distillation:
            fold_input = self.distillation_data_pipepine.process(fold_input, release_date)
        else:
            fold_input = self.data_pipepine.process(fold_input, release_date)
            
        if resolution is None:
            resolution = 0.0
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # FEATURISE THE INPUT
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input_features, cropping_indices, crop_method = featurisation.featurise_input(
            fold_input=fold_input,
            filtered_struc=filtered_struc,
            atoms_iter=atoms_iter,
            whole_pdb_pipeline_config=self.whole_pdb_config,
            resolution=resolution, 
            ccd=self.ccd_dict,
            verbose=False,
            sampled_chains=sampled_chain_ids,
            prediction_seed=prediction_seed,
            distillation_mode=distillation if self.distillation_maskout_low_plddt else False,
        )
        empty_output_struc = input_features['empty_output_struc']
        if len(input_features['seq_mask']) > 3072:
            print(f"too long ,skip, {len(input_features['seq_mask'])}")
            return None
        input_features = {k: v for k, v in input_features.items() if isinstance(v, np.ndarray) and v.dtype != np.dtype('O')}
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # CROP THE FEATURES
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mode == 'predict': 
            # input_features, ground_truth_features, cropped_ground_truth_features = aggregate_crop_features_by_indices(input_features, cropping_indices, FEATURES, self.token_crop_size if self.fix_size else None, crop_method)
            # input_features = complete_features(input_features)
            # return DataSetOutput(input_features=input_features, ground_truth_features=ground_truth_features, cropped_ground_truth_features=cropped_ground_truth_features, empty_output_struc=empty_output_struc, cropping_indices=cropping_indices)
            input_features = complete_features(input_features)
            return DataSetOutput(input_features=input_features, empty_output_struc=empty_output_struc, cropping_indices=cropping_indices, prediction_seed=prediction_seed)
        elif self.mode == 'eval':
            # no crop actually, just to get the ground truth features
            input_features, ground_truth_features, cropped_ground_truth_features = aggregate_crop_features_by_indices(input_features, cropping_indices, FEATURES, self.token_crop_size if self.fix_size else None, crop_method)
            input_features = complete_features(input_features)
            return DataSetOutput(input_features=input_features, ground_truth_features=ground_truth_features, cropped_ground_truth_features=cropped_ground_truth_features, empty_output_struc=empty_output_struc, cropping_indices=cropping_indices)
        else: # train
            cropped_features, ground_truth_features, cropped_ground_truth_features = aggregate_crop_features_by_indices(input_features, cropping_indices, FEATURES, self.token_crop_size if self.fix_size else None, crop_method)
            cropped_features = complete_features(cropped_features)
            return DataSetOutput(input_features=cropped_features, ground_truth_features=ground_truth_features, cropped_ground_truth_features=cropped_ground_truth_features, empty_output_struc=empty_output_struc, cropping_indices=cropping_indices)


    def collate_fn(self, batch):
        """Collate the batch of items.

        Args:
            batch (List): A list of items to collate. List of DataSetOutput objects.

        Returns:
            DataLoaderOutput: A DataLoaderOutput object containing the collated features in torch tensors.
        """
        
        # 1. First stack the input features anyway
        input_features = {k: np.stack([item.input_features[k] for item in batch], axis=0) for k in batch[0].input_features.keys()}
        input_features = {k: torch.tensor(v) for k, v in input_features.items()}
        
        # 2. Then make a list of all the other features
        
        
        # Making ground truth features
        try:
            cropped_ground_truth_features = [
                {k: torch.tensor(v) for k, v in item.cropped_ground_truth_features.items()} 
                for item in batch
            ]
        except:
            cropped_ground_truth_features = [None for _ in batch]
            
        try:
            ground_truth_features = [
                {k: torch.tensor(v) for k, v in item.ground_truth_features.items()} 
                for item in batch
            ]
        except:
            ground_truth_features = [None for _ in batch]
            
            
        empty_output_struc = [item.empty_output_struc for item in batch]
        cropping_indices = [item.cropping_indices for item in batch]
        idx = [item.idx for item in batch]
        sampled_ids = [item.sampled_ids for item in batch]
        prediction_seed = [item.prediction_seed for item in batch]
        
        return dict(
            input_features=input_features,
            cropped_ground_truth_features=cropped_ground_truth_features,
            ground_truth_features=ground_truth_features,
            empty_output_struc=empty_output_struc,
            cropping_indices=cropping_indices,
            idx=idx,
            sampled_ids=sampled_ids, 
            prediction_seed=prediction_seed
        )
        
        


# if __name__=='__main__':
#     mmcif_path = "${DATA_ROOT}/benchmarks/posebusters/test_mmcif_v2"
#     alignment_path = "${DATA_ROOT}/benchmarks/posebusters/colabfold_msa"
#     seq_to_msa_mapping_path = "${DATA_ROOT}/benchmarks/posebuster/seq_to_subdir_mapping.json"
#     sample_targets_path = "${DATA_ROOT}/benchmarks/posebuster/sample_targets.json"
#     input_json_dir = "${DATA_ROOT}/benchmarks/posebuster/debug_jsons"
    
#     dataset = AlphaFoldDataSet(
#                             mmcif_path=None,
#                             sample_targets_path=None,
#                             alignment_path=None,
#                             seq_to_msa_mapping_path=seq_to_msa_mapping_path,
#                             token_crop_size=None,
#                             msa_crop_size=16384,
#                             mode = 'predict',
#                             input_json_dir= input_json_dir,
#                             prediction_seeds=[42],
#                             )
#     dataloader = DataLoader(dataset, 
#                             batch_size=1,
#                             shuffle=False,
#                             num_workers=0,
#                             collate_fn=dataset.collate_fn)
#     max_len = 0
#     for i, features in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
#         breakpoint()
#         max_len = max(max_len, features['input_features']['msa'].shape[-1])
#     print(max_len)
