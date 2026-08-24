import hashlib
import os
import sys
sys.path.insert(0, './')
import random
from torch.utils.data import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from alphafold3.common import folding_input
import numpy as np
import pandas as pd
import json
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.model.pipeline.pipeline import WholePdbPipeline
from alphafold3.model.pipeline.pipeline import compute_template_features
from alphafold3.structure.structure import Structure
import datetime
import torch

from alphafold3.data.affinity_data_utils import (
    _BUCKETS,
    FEATURES,
    DataSetOutput,
    pad_at_dim,
    complete_features,
    aggregate_crop_features_by_indices,
    aggregate_crop_features_by_indices_no_tempalte,
    t_weighted_sampling_no_replacement_optimized,
    normalize_pocket_index,
    pocket_crop_expand,
    pocket_crop_expandV2,
)


def _optional_row_value(row, *columns):
    for column in columns:
        if column not in row:
            continue
        value = row.get(column)
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        return value
    return None


def _optional_row_float(row, *columns):
    value = _optional_row_value(row, *columns)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_row_bool(row, column, default):
    if column not in row:
        return bool(default)
    value = row.get(column)
    try:
        if pd.isna(value):
            return bool(default)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
        return bool(default)
    return bool(value)


def _binary_source_code(source):
    if source is None:
        return 0
    value = str(source).strip().lower()
    if value == "true_aqaffinity_class":
        return 1
    if value == "derived_regression_threshold_7_5":
        return 2
    return 0


def _normalize_affinity_bound_type(value):
    if value is None:
        return "exact"
    text = str(value).strip().lower()
    if not text:
        return "exact"
    if text in {"=", "==", "exact"}:
        return "exact"
    if text in {">", ">=", "lower", "lower_bound", "left_censored"}:
        return "lower_bound"
    if text in {"<", "<=", "upper", "upper_bound", "right_censored"}:
        return "upper_bound"
    if text in {"unknown", "na", "n/a", "none"}:
        return "unknown"
    return "unknown"


def _enforce_ligand_crop_budget(input_features, cropping_indices, token_crop_size):
    ligand_indices = np.flatnonzero(input_features["is_ligand"])
    if len(ligand_indices) > token_crop_size:
        raise ValueError(
            f"ligand has {len(ligand_indices)} tokens, exceeding "
            f"token_crop_size={token_crop_size}"
        )
    if cropping_indices is None:
        return None

    selected = np.unique(np.asarray(cropping_indices, dtype=np.int64))
    ligand_set = set(ligand_indices.tolist())
    non_ligand = [index for index in selected if index not in ligand_set]
    non_ligand = non_ligand[:token_crop_size - len(ligand_indices)]
    return np.asarray(sorted(ligand_set.union(non_ligand)), dtype=np.int32)


def _attach_affinity_supervision_features(features, row):
    regression_mask = _optional_row_bool(row, "REGRESSION_LABEL_MASK", True)
    binary_mask = _optional_row_bool(row, "BINARY_LABEL_MASK", False)
    binary_label = _optional_row_float(row, "BINARY_LABEL")
    if binary_label is None:
        binary_label = 0.0
        binary_mask = False
    binary_source = _optional_row_value(row, "BINARY_LABEL_SOURCE")
    features["regression_label_mask"] = np.asarray(regression_mask, dtype=np.float32)
    features["binary_label"] = np.asarray(float(binary_label), dtype=np.float32)
    features["binary_label_mask"] = np.asarray(binary_mask, dtype=np.float32)
    features["binary_label_source_code"] = np.asarray(_binary_source_code(binary_source), dtype=np.int64)
    return features


class AlphaFoldDataSet(Dataset):
    def __init__(self,
                 sampler_df=None,
                 input_cache_dir=None,
                 data_cache_path=None,
                 mmcif_path=None,
                 alignment_path=None,
                 seq_to_msa_mapping_path=None,
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
                 distillation_sampler=None,
                 recropping_tolerance=10,
                 only_sampled_chains=False,
                 error_dir=None,
                 only_protein_chains=False,
                 crop_ligand=True,
                 prediction_seeds=[42,43,44,45,46],
                 dataset_name='d7',
                 feats_output_dir=None,
                 max_extend=10,
                 crop_sigma=10.0,
                 mode="train",
                 batch_group_cols=None,
                 sampling_seed=42,
                 consistent_group_crop=False):
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
        ## affinity
        self.input_cache_dir = input_cache_dir
        self.data_cache_path = data_cache_path

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
        self.distillation_sampler = distillation_sampler
        self.crop_ligand = crop_ligand

        self.mode = mode
        self.prediction_seeds = prediction_seeds
        self.error_dir = error_dir
        self.dataset_name = dataset_name
        self.feats_output_dir = feats_output_dir
        self.max_extend = max_extend
        self.batch_group_cols = self._normalize_batch_group_cols(batch_group_cols)
        self.sampling_seed = int(sampling_seed)
        self.consistent_group_crop = bool(consistent_group_crop)
        if self.consistent_group_crop and not self.batch_group_cols:
            raise ValueError(
                "consistent_group_crop requires explicit batch_group_cols"
            )
        self.epoch = 0
        self._batch_group_to_indices = None
        self._batch_group_metadata = None
        self.uses_external_batch_sampler = sampler_df is not None
        # Object-storage reads were removed from this distribution; every path
        # is a local filesystem path.
        for path in (
            self.alignment_path,
            self.distillation_alignment_path,
            self.template_path,
        ):
            if path is not None and str(path).startswith('s3://'):
                raise ValueError(
                    f'object-storage paths are not supported in this build: {path}'
                )
        self.oss_petrel_backend = None
        if self.data_cache_path is not None:
            if sampler_df is None:
                self.data = pd.read_parquet(data_cache_path)
                self.data = self.data.sample(frac=1.0)
                # idx reset
                self.data = self.data.reset_index(drop=True)
                print(f"Not use sampler, so shuffle data and reset index.")
            else:
                self.data = sampler_df.reset_index(drop=True)
            print("pred", mode,self.data.shape)
            if self.mode == 'train':
                self.protein_chain_info = self.data
                self.epoch_len = min(self.protein_chain_info.shape[0], epoch_len)
                print(f"self.epoch_len = {self.epoch_len}!!!!!!!!!!!!!!!!!!!!!")
            else:
                self.protein_chain_info = self.data
                self.epoch_len = len(self.protein_chain_info)
            if 'pdbbind' in self.dataset_name:
                self.protein_chain_info = self.protein_chain_info[self.protein_chain_info['group'] == self.mode].reset_index(drop=True)
                self.epoch_len = len(self.protein_chain_info)

        else:
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
                    # distillation_sampled_targets = random.sample(self.distillation_targets, num_distillation_samples)
                    if self.distillation_sampler is None:
                        distillation_sampled_targets = random.sample(self.distillation_targets, num_distillation_samples)
                        print(f"Resampling distillation targets: {len(distillation_sampled_targets)} using random sampling.")
                    else:
                        distillation_sampled_targets = self.distillation_sampler.sample(num_samples=num_distillation_samples)
                        print(f"Resampling distillation targets: {len(distillation_sampled_targets)} using distillation sampling.")

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
                    self.sampled_targets = [f.split('.')[0] for f in os.listdir(input_json_dir) if f.endswith('.json')]
                    self.epoch_len = len(self.sampled_targets) * len(self.prediction_seeds)
                else:
                    self.input_json_dir = None
                    with open(sample_targets_path, 'r') as f:
                        self.sampled_targets = json.load(f)
                    self.epoch_len = len(self.sampled_targets) * len(self.prediction_seeds)

            else:
                raise ValueError(f"Invalid mode: {mode}.")

        if self.uses_external_batch_sampler and self.batch_group_cols:
            self._build_batch_group_index()

        data_pipeline_config = pipeline.DataPipelineConfig(
            use_precomputed_alignments=True,
            precomputed_alignments_path=self.alignment_path,
            sequence_to_precomputed_alignment_id_mapping_path=self.seq_to_msa_mapping_path,
            use_templates=self.use_templates,
            precomputed_templates_path=self.template_path,
            sequence_to_precomputed_template_id_mapping_path=self.seq_to_template_mapping_path,
            pdb_database_path=self.template_mmcif_path,
            oss_petrel_backend=self.oss_petrel_backend,
        )
        self.data_pipepine = pipeline.DataPipeline(data_pipeline_config)

        if self.distillation_sampling_prob > 0:
            distillation_data_pipeline_config = pipeline.DataPipelineConfig(
                use_precomputed_alignments=True,
                precomputed_alignments_path=self.distillation_alignment_path,
                sequence_to_precomputed_alignment_id_mapping_path=self.distillation_seq_to_msa_mapping_path,
                use_templates=False,
                oss_petrel_backend=self.oss_petrel_backend,
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


    @staticmethod
    def _normalize_batch_group_cols(batch_group_cols):
        if batch_group_cols is None:
            return ()
        if isinstance(batch_group_cols, str):
            return (batch_group_cols,)
        columns = tuple(batch_group_cols)
        if not columns:
            raise ValueError("batch_group_cols must not be empty")
        if len(set(columns)) != len(columns):
            raise ValueError(f"batch_group_cols must be unique, got {columns}")
        return columns

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _batch_group_key(self, row):
        values = tuple(_optional_row_value(row, column) for column in self.batch_group_cols)
        return values if values and all(value is not None for value in values) else None

    def _build_batch_group_index(self):
        group_to_indices = {}
        for row_idx, row in self.protein_chain_info.iterrows():
            group_key = self._batch_group_key(row)
            if group_key is None:
                raise ValueError(
                    f"row index {row_idx} has an incomplete composite batch-group key"
                )
            group_to_indices.setdefault(group_key, []).append(row_idx)
        self._batch_group_to_indices = group_to_indices

        group_metadata = {}
        for group_key, row_indices in group_to_indices.items():
            rows = self.protein_chain_info.iloc[row_indices]
            metadata = {"row_indices": tuple(row_indices)}
            if self.consistent_group_crop:
                if "PROTEIN_SEQUENCE" not in rows.columns:
                    raise ValueError(
                        "consistent_group_crop requires PROTEIN_SEQUENCE"
                    )
                sequence_values = rows["PROTEIN_SEQUENCE"].astype("string")
                if sequence_values.isna().any() or sequence_values.str.len().eq(0).any():
                    raise ValueError(
                        "consistent_group_crop requires nonempty protein sequences"
                    )
                sequence_signatures = {
                    hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()
                    for sequence in sequence_values
                }
                if len(sequence_signatures) != 1:
                    raise ValueError(
                        "composite batch group contains multiple protein sequences"
                    )
                if "LIGAND_TOKEN_COUNT" not in rows.columns:
                    raise ValueError(
                        "consistent_group_crop requires LIGAND_TOKEN_COUNT"
                    )
                ligand_counts = pd.to_numeric(
                    rows["LIGAND_TOKEN_COUNT"], errors="coerce"
                )
                if (
                    ligand_counts.isna().any()
                    or (ligand_counts < 0).any()
                    or not np.allclose(ligand_counts, np.floor(ligand_counts))
                ):
                    raise ValueError(
                        "LIGAND_TOKEN_COUNT must contain non-negative integers"
                    )
                max_ligand_tokens = int(ligand_counts.max())
                if max_ligand_tokens >= self.token_crop_size:
                    raise ValueError(
                        f"group ligand token count {max_ligand_tokens} leaves no "
                        f"protein budget at token_crop_size={self.token_crop_size}"
                    )

                if "PROTEIN_LENGTH" not in rows.columns:
                    raise ValueError(
                        "consistent_group_crop requires PROTEIN_LENGTH"
                    )
                protein_length_values = pd.to_numeric(
                    rows["PROTEIN_LENGTH"], errors="coerce"
                )
                if (
                    protein_length_values.isna().any()
                    or (protein_length_values <= 0).any()
                    or not np.allclose(
                        protein_length_values,
                        np.floor(protein_length_values),
                    )
                ):
                    raise ValueError(
                        "PROTEIN_LENGTH must contain positive integers"
                    )
                if protein_length_values.nunique() != 1:
                    raise ValueError(
                        "composite batch group contains multiple protein lengths"
                    )
                protein_length = int(protein_length_values.iloc[0])
                if not sequence_values.str.len().eq(protein_length).all():
                    raise ValueError(
                        "PROTEIN_LENGTH does not match PROTEIN_SEQUENCE"
                    )
                if "MODEL_TOKEN_COUNT" not in rows.columns:
                    raise ValueError(
                        "consistent_group_crop requires MODEL_TOKEN_COUNT"
                    )
                model_token_counts = pd.to_numeric(
                    rows["MODEL_TOKEN_COUNT"], errors="coerce"
                )
                if (
                    model_token_counts.isna().any()
                    or not np.allclose(
                        model_token_counts,
                        np.floor(model_token_counts),
                    )
                    or not np.array_equal(
                        model_token_counts.to_numpy(dtype=np.int64),
                        protein_length + ligand_counts.to_numpy(dtype=np.int64),
                    )
                ):
                    raise ValueError(
                        "MODEL_TOKEN_COUNT must equal protein plus ligand tokens"
                    )

                valid_protein_indices = np.arange(protein_length, dtype=np.int32)
                pocket_union = set()
                if "pocket_index" in rows.columns:
                    for pocket_value in rows["pocket_index"]:
                        pocket_union.update(
                            normalize_pocket_index(
                                pocket_value, valid_protein_indices
                            ).tolist()
                        )
                metadata.update({
                    "max_ligand_tokens": max_ligand_tokens,
                    "protein_length": protein_length,
                    "pocket_indices": np.asarray(
                        sorted(pocket_union), dtype=np.int32
                    ),
                })
            group_metadata[group_key] = metadata
        self._batch_group_metadata = group_metadata

    def _prediction_seed(self, row):
        group_key = None
        if self.mode == "train" and self.uses_external_batch_sampler:
            group_key = self._batch_group_key(row)
        if group_key is None:
            configured_seeds = getattr(self, "prediction_seeds", None)
            if isinstance(configured_seeds, (int, np.integer)):
                return int(configured_seeds)
            if configured_seeds:
                return int(configured_seeds[0])
            return self.sampling_seed
        payload = json.dumps(
            [self.sampling_seed, self.epoch, *group_key],
            ensure_ascii=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    def _retry_indices(self, idx):
        # A ranking batch is defined by the sampler. Replacing one failed row
        # here can duplicate another row already present in that batch and can
        # silently change its labels. Invalid rows belong in preprocessing.
        return [idx]

    def _consistent_group_cropping_indices(
        self,
        row,
        input_features,
        prediction_seed,
    ):
        if not self.consistent_group_crop:
            return None
        group_key = self._batch_group_key(row)
        metadata = self._batch_group_metadata.get(group_key)
        if metadata is None:
            raise ValueError("missing metadata for composite batch group")

        is_ligand = np.asarray(input_features["is_ligand"], dtype=bool)
        is_protein = np.asarray(input_features["is_protein"], dtype=bool)
        if is_ligand.ndim != 1 or is_protein.shape != is_ligand.shape:
            raise ValueError("protein and ligand token masks must be aligned vectors")
        if np.any(is_ligand & is_protein) or not np.all(is_ligand | is_protein):
            raise ValueError(
                "prepared affinity samples must contain only disjoint protein and ligand tokens"
            )
        ligand_indices = np.flatnonzero(is_ligand)
        protein_indices = np.flatnonzero(is_protein)
        declared_ligand_tokens = _optional_row_float(row, "LIGAND_TOKEN_COUNT")
        if declared_ligand_tokens is None or int(declared_ligand_tokens) != len(ligand_indices):
            raise ValueError(
                "LIGAND_TOKEN_COUNT does not match featurized ligand tokens"
            )
        declared_protein_tokens = _optional_row_float(row, "PROTEIN_LENGTH")
        if (
            declared_protein_tokens is None
            or int(declared_protein_tokens) != len(protein_indices)
        ):
            raise ValueError(
                "PROTEIN_LENGTH does not match featurized protein tokens"
            )
        declared_model_tokens = _optional_row_float(row, "MODEL_TOKEN_COUNT")
        if (
            declared_model_tokens is None
            or int(declared_model_tokens) != len(is_ligand)
        ):
            raise ValueError(
                "MODEL_TOKEN_COUNT does not match featurized model tokens"
            )

        protein_budget = self.token_crop_size - metadata["max_ligand_tokens"]
        if len(protein_indices) <= protein_budget:
            return np.asarray(
                sorted(
                    set(protein_indices.tolist()).union(ligand_indices.tolist())
                ),
                dtype=np.int32,
            )
        pocket_indices = metadata["pocket_indices"]
        pocket_indices = pocket_indices[
            np.isin(pocket_indices, protein_indices)
        ]
        if pocket_indices.size:
            return pocket_crop_expand(
                input_features,
                pocket_index=pocket_indices,
                token_crop_size=protein_budget + len(ligand_indices),
                max_extend=self.max_extend,
            )

        if len(protein_indices) > protein_budget:
            crop_rng = random.Random(prediction_seed)
            start = crop_rng.randint(0, len(protein_indices) - protein_budget)
            protein_indices = protein_indices[start:start + protein_budget]
        return np.asarray(
            sorted(set(protein_indices.tolist()).union(ligand_indices.tolist())),
            dtype=np.int32,
        )

    def resample(self):
        """
        Resample the targets for the dataset. This should be called at the start of each epoch.
        """
        # self.protein_chain_info = t_weighted_sampling_no_replacement_optimized(self.data, n_samples=self.epoch_len)
        if self.mode == 'train':
            if getattr(self, "uses_external_batch_sampler", False):
                self.protein_chain_info = self.data
            else:
                self.protein_chain_info = self.protein_chain_info.sample(frac=1).reset_index(drop=True)
            pass
        else:
            self.protein_chain_info = self.data
            self.epoch_len = len(self.protein_chain_info)


    max_getitem_retries = 16

    def __getitem__(self, idx, recropping_times=0):
        """Load one sample without silently substituting another training row."""
        del recropping_times  # Kept for compatibility with older callers.
        retry_indices = self._retry_indices(idx)[:self.max_getitem_retries]

        for candidate_idx in retry_indices:
            row = self.protein_chain_info.iloc[candidate_idx]
            prediction_seed = self._prediction_seed(row)
            try:
                dataset_output = self.process(row, prediction_seed=prediction_seed)
                if dataset_output.input_features['is_ligand'].sum() == 0:
                    raise ValueError("sample has no ligand tokens")
            except Exception as error:
                print(
                    "Error in processing sample "
                    f"index={candidate_idx}: {type(error).__name__}"
                )
                if self.error_dir is not None:
                    with open(os.path.join(self.error_dir, "error.txt"), 'a') as error_file:
                        error_file.write(
                            f"index={candidate_idx}, error={type(error).__name__}\n"
                        )
                continue

            dataset_output.idx = candidate_idx
            dataset_output.sampled_ids = _optional_row_value(
                row, "ID", "row_ordinal", "id"
            )
            dataset_output.file_ids = _optional_row_value(
                row, "id", "INPUT_CACHE_ID", "input_cache_id"
            )
            dataset_output.assay_batch_key = _optional_row_value(
                row, "ASSAY_BATCH_KEY", "TRUE_ASSAY_GROUP_ID"
            )
            dataset_output.sampler_target_group_id = _optional_row_value(row, "SAMPLER_TARGET_GROUP_ID", "target_group_id", "UNIPROT_ID")
            dataset_output.sampler_standard_type = _optional_row_value(row, "SAMPLER_STANDARD_TYPE", "standard_type")
            dataset_output.sampler_label = _optional_row_float(row, "SAMPLER_LABEL", "REG_LABEL")
            dataset_output.sampler_smiles = _optional_row_value(row, "SAMPLER_SMILES", "rdkit_canonical_smiles", "smiles")
            dataset_output.binary_label = _optional_row_float(row, "BINARY_LABEL")
            dataset_output.binary_label_mask = _optional_row_bool(row, "BINARY_LABEL_MASK", False)
            dataset_output.regression_label_mask = _optional_row_bool(row, "REGRESSION_LABEL_MASK", True)
            dataset_output.binary_label_source = _optional_row_value(row, "BINARY_LABEL_SOURCE")
            dataset_output.affinity_task_type = _optional_row_value(row, "AFFINITY_TASK_TYPE")
            affinity_bound_type = _normalize_affinity_bound_type(
                _optional_row_value(row, "AFFINITY_BOUND_TYPE", "affinity_bound_type")
            )
            sampler_affinity_bound_raw = _optional_row_value(
                row, "SAMPLER_AFFINITY_BOUND_TYPE", "sampler_affinity_bound_type"
            )
            sampler_affinity_bound_type = (
                _normalize_affinity_bound_type(sampler_affinity_bound_raw)
                if sampler_affinity_bound_raw is not None
                else affinity_bound_type
            )
            dataset_output.affinity_bound_type = affinity_bound_type
            dataset_output.sampler_affinity_bound_type = sampler_affinity_bound_type or affinity_bound_type
            return dataset_output

        raise RuntimeError(f"Failed to load preprocessed sample index={idx}") from None

    def __len__(self):
        return self.epoch_len

    def idx_to_sampled_target(self, idx):
        if self.mode == 'predict':
            return self.sampled_targets[idx // len(self.prediction_seeds)]
        else:
            return self.sampled_targets[idx]

    def process(self, row, prediction_seed=None):
        """Given the sampled pdb_id and chain_id(s), process the data and return the feature dictionary.

        Args:
            sampled_ids (List): A list containing the pdb_id and the chain_id(s) of the complex. The length of the list is 1, 2 or 3, depending on the sampling method used.
                                   When a chain is sampled, the list will contain only two element. When an interface is sampled, the list will contain three elements.
                                   When the entire complex is sampled, the list will contain only one element.

        Returns:
            dict: A dictionary containing the features of the complex.
        """
        # file_id = f"{self.dataset_name}_{row['id']}"
        file_id = f"{_optional_row_value(row, 'INPUT_CACHE_ID', 'input_cache_id', 'id')}"
        input_cache_path = os.path.join(self.input_cache_dir, f"{file_id}.json")
        # print(input_cache_path)
        if not os.path.exists(input_cache_path):
            input_cache_path = os.path.join(self.input_cache_dir, f"{self.dataset_name}_{file_id}.json")
        if not os.path.exists(input_cache_path):
            uniprot_id = row['UNIPROT_ID']
            input_cache_path = os.path.join(self.input_cache_dir, f"{self.dataset_name}_{file_id}.json")

        fold_input = folding_input.load_fold_input_from_json_path(
            input_cache_path,
            only_protein_chains=self.only_protein_chains,
            oss_petrel_backend=self.oss_petrel_backend,
        )
        atoms_iter = None
        resolution = None
        release_date = None
        filtered_struc = None
        sampled_chain_ids = []
        distillation = False

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
            crop_ligand=self.crop_ligand if self.mode == 'train' else False,
        )
        empty_output_struc = input_features['empty_output_struc']

        input_features = {k: v for k, v in input_features.items() if isinstance(v, np.ndarray) and v.dtype != np.dtype('O')}
        if self.mode == 'train':
            if self.consistent_group_crop:
                cropping_indices = self._consistent_group_cropping_indices(
                    row,
                    input_features,
                    prediction_seed,
                )
                crop_method = "ContiguousCropping"
            else:
                cropping_indices = _enforce_ligand_crop_budget(
                    input_features, cropping_indices, self.token_crop_size
                )
                if cropping_indices is None:
                    protein_indices = np.where(input_features['is_protein'])[0]
                    pocket_indices = normalize_pocket_index(row.get('pocket_index'), protein_indices)

                    if pocket_indices.size > 0:
                        cropping_indices = pocket_crop_expand(
                            input_features,
                            pocket_index=pocket_indices,
                            token_crop_size=self.token_crop_size,
                            max_extend=self.max_extend,
                        )
                    else:
                        # fallback: random contiguous cropping when no pocket info
                        cropping_indices = np.arange(len(input_features['token_index']))
                        if len(cropping_indices) > self.token_crop_size:
                            ligand_mask = input_features['is_ligand']
                            protein_mask = input_features['is_protein']

                            ligand_indices = np.where(ligand_mask)[0]
                            protein_indices = np.where(protein_mask)[0]

                            ligand_length = len(ligand_indices)
                            protein_length = len(protein_indices)
                            crop_protein_length = self.token_crop_size - ligand_length

                            if crop_protein_length <= 0:
                                raise ValueError(f"token_crop_size={self.token_crop_size} < num of ligand token {ligand_length}")

                            if protein_length > crop_protein_length:
                                crop_rng = random.Random(prediction_seed)
                                start_idx = crop_rng.randint(0, protein_length - crop_protein_length)
                                crop_protein_indices = protein_indices[start_idx: start_idx + crop_protein_length]
                            else:
                                crop_protein_indices = protein_indices

                            cropping_indices = np.sort(np.concatenate([crop_protein_indices, ligand_indices]))

            cropping_indices = _enforce_ligand_crop_budget(
                input_features, cropping_indices, self.token_crop_size
            )

            crop_method = "ContiguousCropping" if crop_method is None else crop_method
            input_features["pocket_mask"] = np.ones_like(input_features['seq_mask'], dtype=np.int32)
            cropped_features, ground_truth_features, cropped_ground_truth_features = aggregate_crop_features_by_indices(input_features, cropping_indices, FEATURES, self.token_crop_size if self.fix_size else None, crop_method)
            cropped_features = complete_features(cropped_features)
            cropped_features['affinity'] = row['REG_LABEL']
            _attach_affinity_supervision_features(cropped_features, row)
            cropped_features['pair_mask'] = cropped_features['seq_mask'][None, :] * cropped_features['seq_mask'][:, None]
            protein_mask = cropped_features['is_protein'].astype(bool)  # [num_tokens]
            pair_protein_mask = protein_mask[None, :] * protein_mask[:, None]
            cropped_features['pair_mask'] = cropped_features['pair_mask'] & ~pair_protein_mask

            eye_mask = np.eye(cropped_features['pair_mask'].shape[-1]).astype(bool)
            cropped_features['pair_mask'] = cropped_features['pair_mask'] * (~eye_mask)


            return DataSetOutput(input_features=cropped_features, empty_output_struc=empty_output_struc, cropping_indices=cropping_indices, prediction_seed=prediction_seed)
        else:

            input_features = complete_features(input_features)
            input_features['pair_mask'] = input_features['seq_mask'][None, :] * input_features['seq_mask'][:, None]
            protein_mask = input_features['is_protein'].astype(bool)  # [num_tokens]
            pair_protein_mask = protein_mask[None, :] * protein_mask[:, None]
            input_features['pair_mask'] = input_features['pair_mask'] & ~pair_protein_mask

            # Compute pocket_mask for AffinityModulePocketEmbedV2
            pocket_indices = normalize_pocket_index(row.get('pocket_index'), np.where(protein_mask)[0])
            pocket_mask = np.zeros(len(input_features['seq_mask']), dtype=np.int32)
            if pocket_indices.size > 0:
                pocket_mask[pocket_indices] = 1
            input_features['pocket_mask'] = pocket_mask
            _attach_affinity_supervision_features(input_features, row)

            return DataSetOutput(input_features=input_features, empty_output_struc=empty_output_struc, cropping_indices=cropping_indices, prediction_seed=prediction_seed)


    def _validate_composite_group_batch(self, batch):
        if not self.batch_group_cols:
            return

        actual_indices = [item.idx for item in batch]
        if len(set(actual_indices)) != len(actual_indices):
            raise ValueError(
                "same-group retry produced a duplicate sample inside a rank batch"
            )
        group_keys = [
            self._batch_group_key(self.protein_chain_info.iloc[item.idx])
            for item in batch
        ]
        if any(group_key is None for group_key in group_keys):
            raise ValueError("rank batch contains an incomplete composite group key")
        if len(set(group_keys)) != 1:
            raise ValueError("rank batch mixes composite assay/target groups")

        if not self.consistent_group_crop:
            return
        protein_crop_identities = []
        for item in batch:
            features = item.input_features
            required = {
                "is_protein",
                "seq_mask",
                "asym_id",
                "residue_index",
                "per_chain_token_index",
            }
            missing = required.difference(features)
            if missing:
                raise ValueError(
                    f"cannot validate group crop; missing features {sorted(missing)}"
                )
            protein_mask = (
                np.asarray(features["is_protein"]).astype(bool)
                & np.asarray(features["seq_mask"]).astype(bool)
            )
            identity = tuple(
                zip(
                    np.asarray(features["asym_id"])[protein_mask].tolist(),
                    np.asarray(features["residue_index"])[protein_mask].tolist(),
                    np.asarray(features["per_chain_token_index"])[protein_mask].tolist(),
                )
            )
            protein_crop_identities.append(identity)
        if any(
            identity != protein_crop_identities[0]
            for identity in protein_crop_identities[1:]
        ):
            raise ValueError(
                "composite rank batch does not share one protein crop"
            )


    def collate_fn(self, batch):
        """Collate the batch of items.

        Args:
            batch (List): A list of items to collate. List of DataSetOutput objects.

        Returns:
            DataLoaderOutput: A DataLoaderOutput object containing the collated features in torch tensors.
        """

        self._validate_composite_group_batch(batch)

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
        file_ids = [item.file_ids for item in batch]
        prediction_seed = [item.prediction_seed for item in batch]
        assay_batch_keys = [item.assay_batch_key for item in batch]
        sampler_target_group_ids = [item.sampler_target_group_id for item in batch]
        sampler_standard_types = [item.sampler_standard_type for item in batch]
        sampler_labels = [item.sampler_label for item in batch]
        sampler_smiles = [item.sampler_smiles for item in batch]
        binary_labels = [item.binary_label for item in batch]
        binary_label_masks = [item.binary_label_mask for item in batch]
        regression_label_masks = [item.regression_label_mask for item in batch]
        binary_label_sources = [item.binary_label_source for item in batch]
        affinity_task_types = [item.affinity_task_type for item in batch]
        affinity_bound_types = [
            _normalize_affinity_bound_type(item.affinity_bound_type)
            for item in batch
        ]
        sampler_affinity_bound_types = [
            _normalize_affinity_bound_type(item.sampler_affinity_bound_type or item.affinity_bound_type)
            for item in batch
        ]

        return dict(
            input_features=input_features,
            cropped_ground_truth_features=cropped_ground_truth_features,
            ground_truth_features=ground_truth_features,
            empty_output_struc=empty_output_struc,
            cropping_indices=cropping_indices,
            idx=idx,
            sampled_ids=sampled_ids,
            file_ids=file_ids,
            prediction_seed=prediction_seed,
            assay_batch_keys=assay_batch_keys,
            sampler_target_group_ids=sampler_target_group_ids,
            sampler_standard_types=sampler_standard_types,
            sampler_labels=sampler_labels,
            sampler_smiles=sampler_smiles,
            binary_labels=binary_labels,
            binary_label_masks=binary_label_masks,
            regression_label_masks=regression_label_masks,
            binary_label_sources=binary_label_sources,
            affinity_task_types=affinity_task_types,
            affinity_bound_types=affinity_bound_types,
            sampler_affinity_bound_types=sampler_affinity_bound_types,
        )
