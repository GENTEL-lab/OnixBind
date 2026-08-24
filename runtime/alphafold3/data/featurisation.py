# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 featurisation pipeline."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any
import datetime
import time

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model import features
from alphafold3.model.pipeline import pipeline
import numpy as np


def validate_fold_input(fold_input: folding_input.Input):
  """Validates the fold input contains MSA and templates for featurisation."""
  for i, chain in enumerate(fold_input.protein_chains):
    if chain.unpaired_msa is None:
      raise ValueError(f'Protein chain {i + 1} is missing unpaired MSA.')
    if chain.paired_msa is None:
      raise ValueError(f'Protein chain {i + 1} is missing paired MSA.')
    if chain.templates is None:
      raise ValueError(f'Protein chain {i + 1} is missing Templates.')
  for i, chain in enumerate(fold_input.rna_chains):
    if chain.unpaired_msa is None:
      raise ValueError(f'RNA chain {i + 1} is missing unpaired MSA.')


def featurise_input(
    fold_input: folding_input.Input,
    filtered_struc,
    ccd: chemical_components.Ccd,
    whole_pdb_pipeline_config: pipeline.WholePdbPipeline.Config,
    atoms_iter:  Iterator[Mapping[str, Any]] | None = None,
    resolution: float | None = None,
    sampled_chains: Sequence[str] | None = None,
    verbose: bool = False,
    prediction_seed: int | None = None,
    distillation_mode: bool = False,
    crop_ligand: bool = True,
) -> Sequence[features.BatchDict]:
  """Featurise the folding input.

  Args:
    fold_input: The input to featurise.
    atoms_iter:  The gruond truth atom tables.
    resolution: The resolution of the input data.
    ccd: The chemical components dictionary.
    verbose: Whether to print progress messages.
    sampled_chains: The sampled chains in the input data.

  Returns:
    A featurised batch for each rng_seed in the input.
    The sampled asym ids.
    
  """
  validate_fold_input(fold_input)

  data_pipeline = pipeline.WholePdbPipeline(config=whole_pdb_pipeline_config)
  if prediction_seed is None:
    rng_seed = fold_input.rng_seeds[0]
  else:
    rng_seed = prediction_seed
  if verbose:
    featurisation_start_time = time.time()
    print(f'Featurising {fold_input.name} with rng_seed {rng_seed}.')
  batch, cropping_indices, crop_method = data_pipeline.process_item(
      fold_input=fold_input,
      filtered_struc=filtered_struc,
      ccd=ccd,
      random_state=np.random.RandomState(rng_seed),
      random_seed=rng_seed,
      atoms_iter=atoms_iter,
      resolution=resolution,
      sampled_chains=sampled_chains,
      distillation_mode=distillation_mode,
      crop_ligand=crop_ligand,
  )
  if verbose:
    print(
        f'Featurising {fold_input.name} with rng_seed {rng_seed} '
        f'took {time.time() - featurisation_start_time:.2f} seconds.'
    )

  return batch, cropping_indices, crop_method