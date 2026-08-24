# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Model input dataclass."""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import json
from absl import logging
import os
import pathlib
import random
import re
import string
from typing import Any, Final, Self, TypeAlias, List

from alphafold3 import structure
from alphafold3.constants import chemical_components
from alphafold3.constants import mmcif_names
from alphafold3.model.pipeline import structure_cleaning
from alphafold3.model.atom_layout import atom_layout
from alphafold3.constants import residue_names
from alphafold3.structure import mmcif as mmcif_lib
import rdkit.Chem as rd_chem
import numpy as np
from scipy.spatial.distance import cdist
import itertools


BondAtomId: TypeAlias = tuple[str, int, str]

JSON_DIALECT: Final[str] = 'alphafold3'
JSON_VERSION: Final[int] = 1
JSON_VERSIONS: Final[List[int]] = [1, 2]

ALPHAFOLDSERVER_JSON_DIALECT: Final[str] = 'alphafoldserver'
ALPHAFOLDSERVER_JSON_VERSION: Final[int] = 1


def _validate_keys(actual: Collection[str], expected: Collection[str]):
  """Validates that the JSON doesn't contain any extra unwanted keys."""
  if bad_keys := set(actual) - set(expected):
    raise ValueError(f'Unexpected JSON keys in: {", ".join(sorted(bad_keys))}')


def tokenizer(
    flat_output_layout: atom_layout.AtomLayout,
    ccd: chemical_components.Ccd,
    max_atoms_per_token: int,
    flatten_non_standard_residues: bool,
    logging_name: str,
) -> tuple[atom_layout.AtomLayout, atom_layout.AtomLayout, np.ndarray]:
  """Maps a flat atom layout to tokens for evoformer.

  Creates the evoformer tokens as one token per polymer residue and one token
  per ligand atom. The tokens are represented as AtomLayouts all_tokens
  (1 representative atom per token) atoms per residue, and
  all_token_atoms_layout (num_tokens, max_atoms_per_token). The atoms in a
  residue token use the layout of the corresponding CCD entry

  Args:
    flat_output_layout: flat AtomLayout containing all atoms that the model
      wants to predict.
    ccd: The chemical components dictionary.
    max_atoms_per_token: number of slots per token.
    flatten_non_standard_residues: whether to flatten non-standard residues,
      i.e. whether to use one token per atom for non-standard residues.
    logging_name: logging name for debugging (usually the mmcif_id).

  Returns:
    A tuple (all_tokens, all_tokens_atoms_layout) with
      all_tokens: AtomLayout shape (num_tokens,) containing one representative
        atom per token.
      all_token_atoms_layout: AtomLayout with shape
        (num_tokens, max_atoms_per_token) containing all atoms per token.
      standard_token_idxs: The token index that each token would have if not
        flattening non standard resiudes.
  """
  # Select  the representative atom for each token.
  token_idxs = []
  single_atom_token = []
  standard_token_idxs = []
  current_standard_token_id = 0
  # Iterate over residues, and provide a group_iter over the atoms of each
  # residue.
  for key, group_iter in itertools.groupby(
      zip(
          flat_output_layout.chain_type,
          flat_output_layout.chain_id,
          flat_output_layout.res_id,
          flat_output_layout.res_name,
          flat_output_layout.atom_name,
          np.arange(flat_output_layout.shape[0]),
      ),
      key=lambda x: x[:3],
  ):

    # Get chain type and chain id of this residue
    chain_type, chain_id, _ = key

    # Get names and global idxs for all atoms of this residue
    _, _, _, res_names, atom_names, idxs = zip(*group_iter)

    # As of March 2023, all OTHER CHAINs in pdb are artificial nucleics.
    is_nucleic_backbone = (
        chain_type in mmcif_names.NUCLEIC_ACID_CHAIN_TYPES
        or chain_type == mmcif_names.OTHER_CHAIN
    )
    if chain_type in mmcif_names.PEPTIDE_CHAIN_TYPES:
      res_name = res_names[0]
      if (
          flatten_non_standard_residues
          and res_name not in residue_names.PROTEIN_TYPES_WITH_UNKNOWN
          and res_name != residue_names.MSE
      ):
        # For non-standard protein residues take all atoms.
        # NOTE: This may get very large if we include hydrogens.
        token_idxs.extend(idxs)
        single_atom_token += [True] * len(idxs)
        standard_token_idxs.extend([current_standard_token_id] * len(idxs))
      else:
        # For standard protein residues take 'CA' if it exists, else first atom.
        if 'CA' in atom_names:
          token_idxs.append(idxs[atom_names.index('CA')])
        else:
          token_idxs.append(idxs[0])
        single_atom_token += [False]
        standard_token_idxs.append(current_standard_token_id)
      current_standard_token_id += 1
    elif is_nucleic_backbone:
      res_name = res_names[0]
      if (
          flatten_non_standard_residues
          and res_name not in residue_names.NUCLEIC_TYPES_WITH_2_UNKS
      ):
        # For non-standard nucleic residues take all atoms.
        token_idxs.extend(idxs)
        single_atom_token += [True] * len(idxs)
        standard_token_idxs.extend([current_standard_token_id] * len(idxs))
      else:
        # For standard nucleic residues take C1' if it exists, else first atom.
        if "C1'" in atom_names:
          token_idxs.append(idxs[atom_names.index("C1'")])
        else:
          token_idxs.append(idxs[0])
        single_atom_token += [False]
        standard_token_idxs.append(current_standard_token_id)
      current_standard_token_id += 1
    elif chain_type in mmcif_names.NON_POLYMER_CHAIN_TYPES:
      # For non-polymers take all atoms
      token_idxs.extend(idxs)
      single_atom_token += [True] * len(idxs)
      standard_token_idxs.extend([current_standard_token_id] * len(idxs))
      current_standard_token_id += len(idxs)
    else:
      # Chain type that we don't handle yet.
      logging.warning(
          '%s: ignoring chain %s with chain type %s.',
          logging_name,
          chain_id,
          chain_type,
      )

  assert len(token_idxs) == len(single_atom_token)
  assert len(token_idxs) == len(standard_token_idxs)
  standard_token_idxs = np.array(standard_token_idxs, dtype=np.int32)

  # Create the list of all tokens, represented as a flat AtomLayout with 1
  # representative atom per token.
  all_tokens = flat_output_layout[token_idxs]

  return all_tokens, None, standard_token_idxs

def transform_list_to_dict(data, key_fields, value_fields):
    """
    Transforms a list of dictionaries into a single dictionary.
    
    Parameters:
        data (list or iterable of dict): The input list or iterable of dictionaries.
        key_fields (list of str): The list of keys to combine for the dictionary keys.
        value_fields (list of str): The list of keys to combine for the dictionary values.
    
    Returns:
        dict: A dictionary with combined keys and corresponding values.
    """
    result = {}
    for item in data:
        key = tuple(item[key_field] for key_field in key_fields)
        value = tuple(item[value_field] for value_field in value_fields)
        result[key] = value
    return result


def find_interface_token_and_closest_chains(
    all_chain_id, center_atom_positions, resolved_mask, query_chain_ids, threshold=15.0, top_chains_count=20
):
    """
    Identifies a random interface token on the query chain or selects the chain itself if no interface exists.
    Then finds the closest chains based on the minimum distance between the center atoms of tokens.

    Parameters:
    ----------
    all_chain_id : numpy.ndarray
        Array of shape (num_tokens,) containing the chain IDs of each token. dtype=object.
    center_atom_positions : numpy.ndarray
        Array of shape (num_tokens, 3) containing the 3D coordinates of the center atom for each token. dtype=float.
    resolved_mask : numpy.ndarray
        Boolean array of shape (num_tokens,) indicating whether a token is resolved (True) or not (False).
    query_chain_ids : List[str]
        The chain IDs for which the interface token is to be identified.
    threshold : float, optional
        Distance threshold in Å for determining interface tokens. Default is 15.0 Å.
    top_chains_count : int, optional
        Number of closest chains to return. Default is 20.

    Returns:
    -------
    top_chains : list
        List of up to `top_chains_count` chain IDs sorted by the minimum distance to the selected token
        in the query chain. If no interface tokens are found, returns the query chain itself in the list.

    Notes:
    -----
    - An interface token is defined as a token in the query chain with a center atom
      within the specified threshold of any center atom in another chain.
    - If no interface token is found, the query chain itself is returned as the closest chain.
    """
    # Apply the resolved_mask to filter valid tokens
    valid_positions = center_atom_positions[resolved_mask]
    valid_chain_ids = all_chain_id[resolved_mask]

    # Separate query chain tokens and other tokens
    # ramdom select one chain from the query chain
    if len(query_chain_ids) == 0:
      raise ValueError("query_chain_ids is empty")
    elif len(query_chain_ids) == 1:
      query_chain_id = query_chain_ids[0]
    elif len(query_chain_ids) == 2:
      query_chain_id = random.choice(query_chain_ids)
      remaining_query_chain_id = query_chain_ids[0] if query_chain_ids[1] == query_chain_id else query_chain_ids[1]
      if query_chain_id == remaining_query_chain_id:
        raise ValueError("query_chain_ids are the same")
    else:
      raise ValueError("query_chain_ids should have 1 or 2 elements")
    
    query_chain_mask = (valid_chain_ids == query_chain_id)
    other_chains_mask = (valid_chain_ids != query_chain_id) if len(query_chain_ids) == 1 else (valid_chain_ids != query_chain_id) & (valid_chain_ids != remaining_query_chain_id)
    query_positions = valid_positions[query_chain_mask]
    other_positions = valid_positions[other_chains_mask]
    other_chain_ids = valid_chain_ids[other_chains_mask]

    # Compute pairwise distances between query tokens and other tokens
    dist_matrix = cdist(query_positions, other_positions)
    # Identify interface tokens in the query chain (distance < threshold to any token in other chains)
    interface_mask = (dist_matrix < threshold).any(axis=1)
    interface_indices = np.where(interface_mask)[0]

    # If no interface tokens are found, return the query chain itself
    if interface_indices.size == 0:
        return [query_chain_id] if len(query_chain_ids) == 1 else [query_chain_id, remaining_query_chain_id]

    # Select a token from the interface tokens
    selected_index_in_query_positions = np.random.choice(interface_indices)

    # Find unique chain IDs in other chains
    other_chain_ids_unique = np.unique(other_chain_ids)
    dist_matrix_relative_to_selected = dist_matrix[selected_index_in_query_positions]
    
    # Calculate minimum distances from the selected token to each other chain
    chain_distances = [(query_chain_id, 0.0)] if len(query_chain_ids) == 1 else [(query_chain_id, 0.0), (remaining_query_chain_id, 0.0)]
    for chain_id in other_chain_ids_unique:
        chain_mask = (other_chain_ids == chain_id)
        min_distance = dist_matrix_relative_to_selected[chain_mask].min()
        chain_distances.append((chain_id, min_distance))

    # Sort chains by minimum distance and select the closest chains
    chain_distances.sort(key=lambda x: x[1])
    top_chains = [chain_id for chain_id, dist in chain_distances[:top_chains_count]]

    return top_chains

class Template:
  """Structural template input."""

  __slots__ = ('_mmcif', '_query_to_template')

  def __init__(self, mmcif: str, query_to_template_map: Mapping[int, int]):
    """Initializes the template.

    Args:
      mmcif: The structural template in mmCIF format. The mmCIF should have only
        one protein chain.
      query_to_template_map: A mapping from query residue index to template
        residue index.
    """
    self._mmcif = mmcif
    # Needed to make the Template class hashable.
    self._query_to_template = tuple(query_to_template_map.items())

  @property
  def query_to_template_map(self) -> Mapping[int, int]:
    return dict(self._query_to_template)

  @property
  def mmcif(self) -> str:
    return self._mmcif

  def __hash__(self) -> int:
    return hash((self._mmcif, tuple(sorted(self._query_to_template))))

  def __eq__(self, other: Self) -> bool:
    mmcifs_equal = self._mmcif == other._mmcif
    maps_equal = sorted(self._query_to_template) == sorted(
        other._query_to_template
    )
    return mmcifs_equal and maps_equal


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProteinChain:
  """Protein chain input.

  Attributes:
    id: Unique protein chain identifier.
    sequence: The amino acid sequence of the chain.
    ptms: A list of tuples containing the post-translational modification type
      and the (1-based) residue index where the modification is applied.
    paired_msa: Paired A3M-formatted MSA for this chain. This MSA is not
      deduplicated and will be used to compute paired features. If None, this
      field is unset and must be filled in by the data pipeline before
      featurisation. If set to an empty string, it will be treated as a custom
      MSA with no sequences.
    unpaired_msa: Unpaired A3M-formatted MSA for this chain. This will be
      deduplicated and used to compute unpaired features. If None, this field is
      unset and must be filled in by the data pipeline before featurisation. If
      set to an empty string, it will be treated as a custom MSA with no
      sequences.
    templates: A list of structural templates for this chain. If None, this
      field is unset and must be filled in by the data pipeline before
      featurisation. The list can be empty or contain up to 20 templates.
  """

  id: str
  sequence: str
  unmapped_sequence: str | None = None
  ptms: Sequence[tuple[str, int]]
  paired_msa: str | None = None
  unpaired_msa: str | None = None
  templates: Sequence[Template] | None = None

  def __post_init__(self):
    if not all(res.isalpha() for res in self.sequence):
      raise ValueError(
          f'Protein must contain only digits, got "{self.sequence}"'
      )
    if any(not 0 < mod[1] <= len(self.sequence) for mod in self.ptms):
      raise ValueError(f'Invalid protein modification index: {self.ptms}')

    # Use hashable types for ptms and templates.
    if self.ptms is not None:
      object.__setattr__(self, 'ptms', tuple(self.ptms))
    if self.templates is not None:
      object.__setattr__(self, 'templates', tuple(self.templates))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs ProteinChain from the AlphaFoldServer JSON dict."""
    _validate_keys(
        json_dict.keys(),
        {'sequence', 'glycans', 'modifications', 'count'},
    )
    sequence = json_dict['sequence']

    if 'glycans' in json_dict:
      raise ValueError(
          f'Specifying glycans in the `{ALPHAFOLDSERVER_JSON_DIALECT}` format'
          ' is not currently supported.'
      )

    ptms = [
        (mod['ptmType'].removeprefix('CCD_'), mod['ptmPosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(id=seq_id, sequence=sequence, ptms=ptms)

  @classmethod
  def from_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str | None = None
  ) -> Self:
    """Constructs ProteinChain from the AlphaFold JSON dict."""
    json_dict = json_dict['protein']
    _validate_keys(
        json_dict.keys(),
        {
            'id',
            'sequence',
            'modifications',
            'unpairedMsa',
            'pairedMsa',
            'templates',
        },
    )

    sequence = json_dict['sequence']
    ptms = [
        (mod['ptmType'], mod['ptmPosition'])
        for mod in json_dict.get('modifications', [])
    ]

    unpaired_msa = json_dict.get('unpairedMsa', None)
    paired_msa = json_dict.get('pairedMsa', None)

    raw_templates = json_dict.get('templates', None)

    if raw_templates is None:
      templates = None
    else:
      templates = [
          Template(
              mmcif=template['mmcif'],
              query_to_template_map=dict(
                  zip(template['queryIndices'], template['templateIndices'])
              ),
          )
          for template in raw_templates
      ]

    return cls(
        id=seq_id or json_dict['id'],
        sequence=sequence,
        ptms=ptms,
        paired_msa=paired_msa,
        unpaired_msa=unpaired_msa,
        templates=templates,
    )

  def to_dict(self) -> Mapping[str, Mapping[str, Any]]:
    """Converts ProteinChain to an AlphaFold JSON dict."""
    if self.templates is None:
      templates = None
    else:
      templates = [
          {
              'mmcif': template.mmcif,
              'queryIndices': list(template.query_to_template_map.keys()),
              'templateIndices': (
                  list(template.query_to_template_map.values()) or None
              ),
          }
          for template in self.templates
      ]
    contents = {
        'id': self.id,
        'sequence': self.sequence,
        'modifications': [
            {'ptmType': ptm[0], 'ptmPosition': ptm[1]} for ptm in self.ptms
        ],
        'unpairedMsa': self.unpaired_msa,
        'pairedMsa': self.paired_msa,
        'templates': templates,
    }
    return {'protein': contents}

  def to_ccd_sequence(self) -> Sequence[str]:
    """Converts to a sequence of CCD codes."""
    ccd_coded_seq = [
        residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(res, residue_names.UNK)
        for res in self.sequence
    ]
    for ptm_code, ptm_index in self.ptms:
      ccd_coded_seq[ptm_index - 1] = ptm_code
    return ccd_coded_seq

  def fill_missing_fields(self) -> Self:
    """Fill missing MSA and template fields with default values."""
    return dataclasses.replace(
        self,
        unpaired_msa=self.unpaired_msa or '',
        paired_msa=self.paired_msa or '',
        templates=self.templates or [],
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RnaChain:
  """RNA chain input.

  Attributes:
    id: Unique RNA chain identifier.
    sequence: The RNA sequence of the chain.
    modifications: A list of tuples containing the modification type and the
      (1-based) residue index where the modification is applied.
    unpaired_msa: Unpaired A3M-formatted MSA for this chain. This will be
      deduplicated and used to compute unpaired features. If None, this field is
      unset and must be filled in by the data pipeline before featurisation. If
      set to an empty string, it will be treated as a custom MSA with no
      sequences.
  """

  id: str
  sequence: str
  unmapped_sequence: str | None = None
  modifications: Sequence[tuple[str, int]]
  unpaired_msa: str | None = None

  def __post_init__(self):
    if not all(res.isalpha() for res in self.sequence):
      raise ValueError(f'RNA must contain only digits, got "{self.sequence}"')
    if any(not 0 < mod[1] <= len(self.sequence) for mod in self.modifications):
      raise ValueError(f'Invalid RNA modification index: {self.modifications}')

    # Use hashable types for modifications.
    object.__setattr__(self, 'modifications', tuple(self.modifications))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs RnaChain from the AlphaFoldServer JSON dict."""
    _validate_keys(json_dict.keys(), {'sequence', 'modifications', 'count'})
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'].removeprefix('CCD_'), mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(id=seq_id, sequence=sequence, modifications=modifications)

  @classmethod
  def from_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str | None = None
  ) -> Self:
    """Constructs RnaChain from the AlphaFold JSON dict."""
    json_dict = json_dict['rna']
    _validate_keys(
        json_dict.keys(), {'id', 'sequence', 'unpairedMsa', 'modifications'}
    )
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'], mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]
    unpaired_msa = json_dict.get('unpairedMsa', None)
    return cls(
        id=seq_id or json_dict['id'],
        sequence=sequence,
        modifications=modifications,
        unpaired_msa=unpaired_msa,
    )

  def to_dict(self) -> Mapping[str, Mapping[str, Any]]:
    """Converts RnaChain to an AlphaFold JSON dict."""
    contents = {
        'id': self.id,
        'sequence': self.sequence,
        'modifications': [
            {'modificationType': mod[0], 'basePosition': mod[1]}
            for mod in self.modifications
        ],
        'unpairedMsa': self.unpaired_msa,
    }
    return {'rna': contents}

  def to_ccd_sequence(self) -> Sequence[str]:
    """Converts to a sequence of CCD codes."""
    mapping = {r: r for r in residue_names.RNA_TYPES}  # Same 1-letter and CCD.
    ccd_coded_seq = [
        mapping.get(res, residue_names.UNK_RNA) for res in self.sequence
    ]
    for ccd_code, modification_index in self.modifications:
      ccd_coded_seq[modification_index - 1] = ccd_code
    return ccd_coded_seq

  def fill_missing_fields(self) -> Self:
    """Fill missing MSA fields with default values."""
    return dataclasses.replace(self, unpaired_msa=self.unpaired_msa or '')


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DnaChain:
  """Single strand DNA chain input.

  Attributes:
    id: Unique DNA chain identifier.
    sequence: The DNA sequence of the chain.
    modifications: A list of tuples containing the modification type and the
      (1-based) residue index where the modification is applied.
  """

  id: str
  sequence: str
  unmapped_sequence: str | None = None
  modifications: Sequence[tuple[str, int]]

  def __post_init__(self):
    if not all(res.isalpha() for res in self.sequence):
      raise ValueError(f'DNA must contain only digits, got "{self.sequence}"')
    if any(not 0 < mod[1] <= len(self.sequence) for mod in self.modifications):
      raise ValueError(f'Invalid DNA modification index: {self.modifications}')

    # Use hashable types for modifications.
    object.__setattr__(self, 'modifications', tuple(self.modifications))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs DnaChain from the AlphaFoldServer JSON dict."""
    _validate_keys(json_dict.keys(), {'sequence', 'modifications', 'count'})
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'].removeprefix('CCD_'), mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(id=seq_id, sequence=sequence, modifications=modifications)

  @classmethod
  def from_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str | None = None
  ) -> Self:
    """Constructs DnaChain from the AlphaFold JSON dict."""
    json_dict = json_dict['dna']
    _validate_keys(json_dict.keys(), {'id', 'sequence', 'modifications'})
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'], mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(
        id=seq_id or json_dict['id'],
        sequence=sequence,
        modifications=modifications,
    )

  def to_dict(self) -> Mapping[str, Mapping[str, Any]]:
    """Converts DnaChain to an AlphaFold JSON dict."""
    contents = {
        'id': self.id,
        'sequence': self.sequence,
        'modifications': [
            {'modificationType': mod[0], 'basePosition': mod[1]}
            for mod in self.modifications
        ],
    }
    return {'dna': contents}

  def to_ccd_sequence(self) -> Sequence[str]:
    """Converts to a sequence of CCD codes."""
    ccd_coded_seq = [
        residue_names.DNA_COMMON_ONE_TO_TWO.get(res, residue_names.UNK_DNA)
        for res in self.sequence
    ]
    for ccd_code, modification_index in self.modifications:
      ccd_coded_seq[modification_index - 1] = ccd_code
    return ccd_coded_seq


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ligand:
  """Ligand input.

  Attributes:
    id: Unique ligand "chain" identifier.
    ccd_ids: The Chemical Component Dictionary or user-defined CCD IDs of the
      chemical components of the ligand. Typically, this is just a single ID,
      but some ligands are composed of multiple components. If that is the case,
      a bond linking these components should be added to the bonded_atom_pairs
      Input field.
    smiles: The SMILES representation of the ligand.
  """

  id: str
  ccd_ids: Sequence[str] | None = None
  smiles: str | None = None

  def __post_init__(self):
    if (self.ccd_ids is None) == (self.smiles is None):
      raise ValueError('Ligand must have one of CCD ID or SMILES set.')

    if self.smiles is not None:
      mol = rd_chem.MolFromSmiles(self.smiles)
      if not mol:
        raise ValueError(f'Unable to make RDKit Mol from SMILES: {self.smiles}')

    # Use hashable types for ccd_ids.
    if self.ccd_ids is not None:
      object.__setattr__(self, 'ccd_ids', tuple(self.ccd_ids))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs Ligand from the AlphaFoldServer JSON dict."""
    # Ligand can be specified either as a ligand, or ion (special-case).
    _validate_keys(json_dict.keys(), {'ligand', 'ion', 'count'})
    if 'ligand' in json_dict:
      return cls(id=seq_id, ccd_ids=[json_dict['ligand'].removeprefix('CCD_')])
    elif 'ion' in json_dict:
      return cls(id=seq_id, ccd_ids=[json_dict['ion']])
    else:
      raise ValueError(f'Unknown ligand type: {json_dict}')

  @classmethod
  def from_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str | None = None
  ) -> Self:
    """Constructs Ligand from the AlphaFold JSON dict."""
    json_dict = json_dict['ligand']
    _validate_keys(json_dict.keys(), {'id', 'ccdCodes', 'smiles'})
    if json_dict.get('ccdCodes') and json_dict.get('smiles'):
      raise ValueError(
          'Ligand cannot have both CCD code and SMILES set at the same time, '
          f'got CCD: {json_dict["ccdCode"]} and SMILES: {json_dict["smiles"]}'
      )

    if 'ccdCodes' in json_dict:
      return cls(id=seq_id or json_dict['id'], ccd_ids=json_dict['ccdCodes'])
    elif 'smiles' in json_dict:
      return cls(id=seq_id or json_dict['id'], smiles=json_dict['smiles'])
    else:
      raise ValueError(f'Unknown ligand type: {json_dict}')

  def to_dict(self) -> Mapping[str, Any]:
    """Converts Ligand to an AlphaFold JSON dict."""
    contents = {'id': self.id}
    if self.ccd_ids is not None:
      contents['ccdCodes'] = self.ccd_ids
    if self.smiles is not None:
      contents['smiles'] = self.smiles
    return {'ligand': contents}


def _sample_rng_seed() -> int:
  """Sample a random seed for AlphaFoldServer job."""
  # See https://alphafoldserver.com/faq#what-are-seeds-and-how-are-they-set.
  return random.randint(0, 2**32 - 1)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Input:
  """AlphaFold input.

  Attributes:
    name: The name of the target.
    chains: Protein chains, RNA chains, DNA chains, or ligands.
    protein_chains: Protein chains.
    rna_chains: RNA chains.
    dna_chains: Single strand DNA chains.
    ligands: Ligand (including ion) inputs.
    rng_seeds: Random number generator seeds, one for each model execution.
    bonded_atom_pairs: A list of tuples of atoms that are bonded to each other.
      Each atom is defined by a tuple of (chain_id, res_id, atom_name). Chain
      IDs must be set if there are any bonded atoms. Residue IDs are 1-indexed.
      Atoms in ligands defined by SMILES can't be bonded since SMILES doesn't
      define unique atom names.
    user_ccd: Optional user-defined chemical component dictionary in the CIF
      format. This can be used to provide additional CCD entries that are not
      present in the default CCD and thus define arbitrary new ligands. This is
      more expressive than SMILES since it allows to name all atoms within the
      ligand which in turn makes it possible to define bonds using those atoms.
  """

  name: str
  chains: Sequence[ProteinChain | RnaChain | DnaChain | Ligand]
  rng_seeds: Sequence[int]
  bonded_atom_pairs: Sequence[tuple[BondAtomId, BondAtomId]] | None = None
  user_ccd: str | None = None

  def __post_init__(self):
    if not self.rng_seeds:
      raise ValueError('Input must have at least one RNG seed.')

    if not self.name.strip() or not self.sanitised_name():
      raise ValueError(
          'Input name must be non-empty and contain at least one valid'
          ' character (letters, numbers, dots, dashes, underscores).'
      )

    chain_ids = [c.id for c in self.chains]
    # allow for "-" and numbers in chain ids
    
    # Define the pattern for valid IDs
    pattern = re.compile('^[A-Z0-9-]+$')

    # Find invalid IDs using regular expressions
    invalid_ids = [c_id for c_id in chain_ids if not pattern.match(c_id)]

    if invalid_ids:
        raise ValueError(
            f'IDs must be uppercase letters, digits, or "-", got invalid IDs: {invalid_ids}'
        )
    if len(set(chain_ids)) != len(chain_ids):
      raise ValueError('Input JSON contains sequences with duplicate IDs.')

    # Use hashable types for chains, rng_seeds, and bonded_atom_pairs.
    object.__setattr__(self, 'chains', tuple(self.chains))
    object.__setattr__(self, 'rng_seeds', tuple(self.rng_seeds))
    if self.bonded_atom_pairs is not None:
      object.__setattr__(
          self, 'bonded_atom_pairs', tuple(self.bonded_atom_pairs)
      )

  @property
  def protein_chains(self) -> Sequence[ProteinChain]:
    return [chain for chain in self.chains if isinstance(chain, ProteinChain)]

  @property
  def rna_chains(self) -> Sequence[RnaChain]:
    return [chain for chain in self.chains if isinstance(chain, RnaChain)]

  @property
  def dna_chains(self) -> Sequence[DnaChain]:
    return [chain for chain in self.chains if isinstance(chain, DnaChain)]

  @property
  def ligands(self) -> Sequence[Ligand]:
    return [chain for chain in self.chains if isinstance(chain, Ligand)]

  @classmethod
  def from_alphafoldserver_fold_job(cls, fold_job: Mapping[str, Any]) -> Self:
    """Constructs Input from an AlphaFoldServer fold job."""

    # Validate the fold job has the correct format.
    _validate_keys(
        fold_job.keys(),
        {'name', 'modelSeeds', 'sequences', 'dialect', 'version'},
    )
    if 'dialect' not in fold_job and 'version' not in fold_job:
      dialect = ALPHAFOLDSERVER_JSON_DIALECT
      version = ALPHAFOLDSERVER_JSON_VERSION
    elif 'dialect' in fold_job and 'version' in fold_job:
      dialect = fold_job['dialect']
      version = fold_job['version']
    else:
      raise ValueError(
          'AlphaFold Server input JSON must either contain both `dialect` and'
          ' `version` fields, or neither. If neither is specified, it is'
          f' assumed that `dialect="{ALPHAFOLDSERVER_JSON_DIALECT}"` and'
          f' `version="{ALPHAFOLDSERVER_JSON_VERSION}"`.'
      )

    if dialect != ALPHAFOLDSERVER_JSON_DIALECT:
      raise ValueError(
          f'AlphaFold Server input JSON has unsupported dialect: {dialect}, '
          f'expected {ALPHAFOLDSERVER_JSON_DIALECT}.'
      )

    # For now, there is only one AlphaFold Server JSON version.
    if version != ALPHAFOLDSERVER_JSON_VERSION:
      raise ValueError(
          f'AlphaFold Server input JSON has unsupported version: {version}, '
          f'expected {ALPHAFOLDSERVER_JSON_VERSION}.'
      )

    # Parse the chains.
    chains = []
    for sequence in fold_job['sequences']:
      if 'proteinChain' in sequence:
        for _ in range(sequence['proteinChain'].get('count', 1)):
          chains.append(
              ProteinChain.from_alphafoldserver_dict(
                  sequence['proteinChain'],
                  seq_id=mmcif_lib.int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'rnaSequence' in sequence:
        for _ in range(sequence['rnaSequence'].get('count', 1)):
          chains.append(
              RnaChain.from_alphafoldserver_dict(
                  sequence['rnaSequence'],
                  seq_id=mmcif_lib.int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'dnaSequence' in sequence:
        for _ in range(sequence['dnaSequence'].get('count', 1)):
          chains.append(
              DnaChain.from_alphafoldserver_dict(
                  sequence['dnaSequence'],
                  seq_id=mmcif_lib.int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'ion' in sequence:
        for _ in range(sequence['ion'].get('count', 1)):
          chains.append(
              Ligand.from_alphafoldserver_dict(
                  sequence['ion'],
                  seq_id=mmcif_lib.int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'ligand' in sequence:
        for _ in range(sequence['ligand'].get('count', 1)):
          chains.append(
              Ligand.from_alphafoldserver_dict(
                  sequence['ligand'],
                  seq_id=mmcif_lib.int_id_to_str_id(len(chains) + 1),
              )
          )
      else:
        raise ValueError(f'Unknown sequence type: {sequence}')

    if 'modelSeeds' in fold_job and fold_job['modelSeeds']:
      rng_seeds = [int(seed) for seed in fold_job['modelSeeds']]
    else:
      rng_seeds = [_sample_rng_seed()]

    return cls(name=fold_job['name'], chains=chains, rng_seeds=rng_seeds)

  @classmethod
  def from_json(cls, json_str: str, only_protein_chains: bool = False, oss_petrel_backend=None) -> Self:
    """Loads the input from the AlphaFold JSON string."""

    raw_json = json.loads(json_str)
    raw_json = check_input_dict_from_bucket(raw_json, oss_petrel_backend=oss_petrel_backend)
    _validate_keys(
        raw_json.keys(),
        {
            'dialect',
            'version',
            'name',
            'modelSeeds',
            'sequences',
            'bondedAtomPairs',
            'userCCD',
        },
    )

    if 'dialect' not in raw_json or 'version' not in raw_json:
      raise ValueError(
          'AlphaFold 3 input JSON must contain `dialect` and `version` fields.'
      )

    if raw_json['dialect'] != JSON_DIALECT:
      raise ValueError(
          'AlphaFold 3 input JSON has unsupported dialect:'
          f' {raw_json["dialect"]}, expected {JSON_DIALECT}.'
      )

    # For now, there is only one AlphaFold 3 JSON version.
    if raw_json['version'] not in JSON_VERSIONS:
      raise ValueError(
          'AlphaFold 3 input JSON has unsupported version:'
          f' {raw_json["version"]}, expected {JSON_VERSIONS}.'
      )

    if 'sequences' not in raw_json:
      raise ValueError('AlphaFold 3 input JSON does not contain any sequences.')

    if 'modelSeeds' not in raw_json or not raw_json['modelSeeds']:
      raise ValueError(
          'AlphaFold 3 input JSON must specify at least one rng seed in'
          ' `modelSeeds`.'
      )

    sequences = raw_json['sequences']

    # Make sure sequence IDs are all set.
    raw_sequence_ids = [next(iter(s.values())).get('id') for s in sequences]
    if all(raw_sequence_ids):
      sequence_ids = []
      for sequence_id in raw_sequence_ids:
        if isinstance(sequence_id, list):
          sequence_ids.append(sequence_id)
        else:
          sequence_ids.append([sequence_id])
    else:
      raise ValueError(
          'AlphaFold 3 input JSON contains sequences with unset IDs.'
      )

    flat_seq_ids = []
    for seq_ids in sequence_ids:
      flat_seq_ids.extend(seq_ids)

    chains = []
    for seq_ids, sequence in zip(sequence_ids, sequences, strict=True):
      if len(sequence) != 1:
        raise ValueError(f'Chain {seq_ids} has more than 1 sequence.')
      for seq_id in seq_ids:
        if 'protein' in sequence:
          chains.append(ProteinChain.from_dict(sequence, seq_id=seq_id))
        elif only_protein_chains:
          continue
        elif 'rna' in sequence:
          chains.append(RnaChain.from_dict(sequence, seq_id=seq_id))
        elif 'dna' in sequence:
          chains.append(DnaChain.from_dict(sequence, seq_id=seq_id))
        elif 'ligand' in sequence:
          chains.append(Ligand.from_dict(sequence, seq_id=seq_id))
        else:
          raise ValueError(f'Unknown sequence type: {sequence}')

    ligands = [chain for chain in chains if isinstance(chain, Ligand)]
    bonded_atom_pairs = None
    if bonds := raw_json.get('bondedAtomPairs'):
      bonded_atom_pairs = []
      for bond in bonds:
        if len(bond) != 2:
          raise ValueError(f'Bond {bond} must have 2 atoms, got {len(bond)}.')
        bond_beg, bond_end = bond
        if (
            len(bond_beg) != 3
            or not isinstance(bond_beg[0], str)
            or not isinstance(bond_beg[1], int)
            or not isinstance(bond_beg[2], str)
        ):
          raise ValueError(
              f'Atom {bond_beg} in bond {bond} must have 3 components: '
              '(chain_id: str, res_id: int, atom_name: str).'
          )
        if (
            len(bond_end) != 3
            or not isinstance(bond_end[0], str)
            or not isinstance(bond_end[1], int)
            or not isinstance(bond_end[2], str)
        ):
          raise ValueError(
              f'Atom {bond_end} in bond {bond} must have 3 components: '
              '(chain_id: str, res_id: int, atom_name: str).'
          )
        if bond_beg[0] not in flat_seq_ids or bond_end[0] not in flat_seq_ids:
          raise ValueError(f'Invalid chain ID(s) in bond {bond}')
        if bond_beg[1] <= 0 or bond_end[1] <= 0:
          raise ValueError(f'Invalid residue ID(s) in bond {bond}')
        smiles_ligand_ids = set(l.id for l in ligands if l.smiles is not None)
        if bond_beg[0] in smiles_ligand_ids:
          raise ValueError(
              f'Bond {bond} involves an unsupported SMILES ligand {bond_beg[0]}'
          )
        if bond_end[0] in smiles_ligand_ids:
          raise ValueError(
              f'Bond {bond} involves an unsupported SMILES ligand {bond_end[0]}'
          )
        bonded_atom_pairs.append((tuple(bond_beg), tuple(bond_end)))

    return cls(
        name=raw_json['name'],
        chains=chains,
        rng_seeds=[int(seed) for seed in raw_json['modelSeeds']],
        bonded_atom_pairs=bonded_atom_pairs,
        user_ccd=raw_json.get('userCCD'),
    )

  @classmethod
  def from_mmcif(cls, mmcif_str: str,pdb_id :str, ccd: chemical_components.Ccd, whole_pdb_config, sampled_chains: List[str],interface_threshold: float = 15.0, max_num_chains:int = 20, only_sampled_chains: bool = False, only_protein_chains: bool = False, valid_chains: list[str] = None) -> Self:
    """Loads the input from an mmCIF string.

    WARNING: Since rng seeds are not stored in mmCIFs, an rng seed is sampled
    in the returned `Input`.

    Args:
      mmcif_str: The mmCIF string.
      ccd: The chemical components dictionary.
      whole_pdb_config: The data pipeline configuration.
      sampled_chains: The chains to be sampled.
      interface_threshold: The distance threshold to define the interface.
      max_num_chains: The maximum number of chains to be selected.

    Returns:
      The input in an Input format.
    """
    if len(sampled_chains) ==0:
      max_num_chains = 1896
      
    random_seed = _sample_rng_seed()
    struc = structure.from_mmcif(
        mmcif_str,
        include_water=False,
        fix_mse_residues=True,
        fix_arginines=True,
        fix_unknown_dna=True,
        include_bonds=True,
        include_other=True,
    )

    # name the structure, convert the case to upper
    struc._name = pdb_id.upper()
    
    # Create default bioassembly, expanding structures implied by stoichiometry.
    struc = struc.generate_bioassembly(None)
    if valid_chains is not None:
        if len(valid_chains) > max_num_chains:
          raise ValueError(f'Number of valid chains {len(valid_chains)} is greater than max_num_chains {max_num_chains}')
        struc = struc.filter(chain_id=valid_chains)
    
    cleaned_struc, cleaning_metadata = structure_cleaning.clean_structure(
      struc,
      ccd=ccd,
      drop_non_standard_atoms=True,
      drop_missing_sequence=True,
      filter_clashes=True,
      filter_crystal_aids=True,
      filter_waters=True,
      filter_hydrogens=True,
      filter_leaving_atoms=whole_pdb_config.drop_ligand_leaving_atoms,
      only_glycan_ligands_for_leaving_atoms=True,
      covalent_bonds_only=True,
      remove_polymer_polymer_bonds=True,
      remove_bad_bonds=True,
      remove_nonsymmetric_bonds=whole_pdb_config.remove_nonsymmetric_bonds,
      sampled_chains=sampled_chains if len(sampled_chains) > 0 else None
    )
    num_clashing_chains_removed = cleaning_metadata[
        'num_clashing_chains_removed'
    ]

    if num_clashing_chains_removed:
      logging.info(
          'Removed %d clashing chains from %s',
          num_clashing_chains_removed,
          pdb_id,
      )
    # filter all polymer chains with less than 4 resolved residues
    chain_ids_to_keep = []
    resolved_sequences_dict = cleaned_struc.chain_single_letter_sequence(
            include_missing_residues=False
        )

    
    for chain in cleaned_struc.iter_chains():
      chain_id = chain['chain_id']
      chain_type = chain['chain_type']
      if only_protein_chains and chain_type != mmcif_names.PROTEIN_CHAIN:
        continue
      if chain_type in mmcif_names.POLYMER_CHAIN_TYPES:
        if len(resolved_sequences_dict[chain_id]) < 4:
          continue
      else:
        if len(resolved_sequences_dict[chain_id]) < 1:
          continue
      chain_ids_to_keep.append(chain_id)
      
    chain_ids_to_keep = sampled_chains if only_sampled_chains else chain_ids_to_keep
    
    cleaned_struc = cleaned_struc.filter(chain_id=chain_ids_to_keep)        
    if cleaned_struc.num_chains == 0:
      raise ValueError(f'No chains in the structure after filtering, pdb_id: {pdb_id}')
    
    residues = atom_layout.residues_from_structure(
      cleaned_struc, include_missing_residues=True
   )

    flat_output_layout = atom_layout.make_flat_atom_layout(
        residues,
        ccd=ccd,
        with_hydrogens=False,
        skip_unk_residues=False,
        polymer_ligand_bonds=None,
        ligand_ligand_bonds=None,
        drop_ligand_leaving_atoms=False,
    )

    # Select the tokens for Evoformer.
    # Each token (e.g. a residue) is encoded as one representative atom. This
    # is flexible enough to allow the 1-token-per-atom ligand representation
    # in the future.
    
    all_tokens, _, standard_token_idxs = (
        tokenizer(
            flat_output_layout,
            ccd=ccd,
            max_atoms_per_token=whole_pdb_config.max_atoms_per_token,
            flatten_non_standard_residues=whole_pdb_config.flatten_non_standard_residues,
            logging_name=f'{cleaned_struc.name}, random_seed={random_seed}',
        )
    )
    all_chain_id = all_tokens.chain_id
    if not set(sampled_chains).issubset(set(all_chain_id)):
      raise ValueError(f'Chains {sampled_chains} not found in the structure. All chains: {set(all_chain_id)}, pdb_id: {pdb_id}')
    
    chain_res_atom_to_pos = transform_list_to_dict(cleaned_struc.iter_atoms(), ['chain_id', 'res_id', 'atom_name'], ['atom_x', 'atom_y', 'atom_z'])

    center_atom_positions = np.zeros((all_tokens.shape[0], 3), dtype=np.float32)
    resolved_mask = np.zeros(all_tokens.shape[0], dtype=bool)
    for idx in np.ndindex(all_tokens.shape[0]):
      chain_id = all_tokens.chain_id[idx]
      res_id = all_tokens.res_id[idx]
      atom_name = all_tokens.atom_name[idx]
      atom_pos = chain_res_atom_to_pos.get((chain_id, res_id, atom_name))
      if atom_pos is not None:
          center_atom_positions[idx] = atom_pos
          resolved_mask[idx] = True

    num_unique_chains = len(np.unique(all_chain_id))
    if num_unique_chains > max_num_chains:
      seleted_chains = find_interface_token_and_closest_chains(
        all_chain_id=all_chain_id,
        center_atom_positions=center_atom_positions,
        resolved_mask=resolved_mask,
        query_chain_ids=sampled_chains,
        threshold=interface_threshold,
        top_chains_count=max_num_chains,
      )
    
      logging.info(f'There are {num_unique_chains} chains in the structure, {len(seleted_chains)} chains are selected.')
    # only keep the selected chains in the structure
    
      filtered_struc = cleaned_struc.filter(chain_id=seleted_chains)
    else:
      filtered_struc = cleaned_struc
      
    sequences = filtered_struc.chain_single_letter_sequence(
        include_missing_residues=True
    )
    unmapped_sequences = filtered_struc.unmapped_chain_single_letter_sequence(
        include_missing_residues=True
    )

    chains = []
    for chain_id, chain_type in zip(
        filtered_struc.group_by_chain.chain_id, filtered_struc.group_by_chain.chain_type
    ):
      sequence = sequences[chain_id]
      unmapped_sequence = unmapped_sequences[chain_id]

      if chain_type in mmcif_names.NON_POLYMER_CHAIN_TYPES:
        residues = list(filtered_struc.chain_res_name_sequence()[chain_id])
        if all(ccd.get(res) is not None for res in residues):
          chains.append(Ligand(id=chain_id, ccd_ids=residues))
        elif len(residues) == 1:
          comp_name = residues[0]
          comps = filtered_struc.chemical_components_data
          if comps is None:
            raise ValueError(
                'Missing mmCIF chemical components data - this is required for '
                f'a non-CCD ligand {comp_name} defined using SMILES string.'
            )
          chains.append(
              Ligand(id=chain_id, smiles=comps.chem_comp[comp_name].pdbx_smiles)
          )
        else:
          raise ValueError(
              'Multi-component ligand must be defined using CCD IDs, defining'
              ' using SMILES is supported only for single-component ligands. '
              f'Got {residues}'
          )
      else:
        residues = filtered_struc.chain_res_name_sequence()[chain_id]
        fixed = filtered_struc.chain_res_name_sequence(
            fix_non_standard_polymer_res=True
        )[chain_id]
        modifications = [
            (orig, i + 1)
            for i, (orig, fixed) in enumerate(zip(residues, fixed, strict=True))
            if orig != fixed
        ]
        if chain_type == mmcif_names.PROTEIN_CHAIN:
          chains.append(
              ProteinChain(id=chain_id, sequence=sequence,unmapped_sequence=unmapped_sequence, ptms=modifications)
          )
        elif chain_type == mmcif_names.RNA_CHAIN:
          chains.append(
              RnaChain(
                  id=chain_id, sequence=sequence,unmapped_sequence=unmapped_sequence, modifications=modifications
              )
          )
        elif chain_type == mmcif_names.DNA_CHAIN:
          chains.append(
              DnaChain(
                  id=chain_id, sequence=sequence,unmapped_sequence=unmapped_sequence, modifications=modifications
              )
          )

    bonded_atom_pairs = []
    chain_ids = set(c.id for c in chains)
    for atom_a, atom_b, _ in filtered_struc.iter_bonds():
      if atom_a['chain_id'] in chain_ids and atom_b['chain_id'] in chain_ids:
        beg = (atom_a['chain_id'], int(atom_a['res_id']), atom_a['atom_name'])
        end = (atom_b['chain_id'], int(atom_b['res_id']), atom_b['atom_name'])
        bonded_atom_pairs.append((beg, end))

    return filtered_struc.iter_atoms(),filtered_struc.resolution,filtered_struc, cls(
        name=filtered_struc.name,
        chains=chains,
        # mmCIFs don't store rng seeds, so we need to sample one here.
        rng_seeds=[random_seed],
        bonded_atom_pairs=bonded_atom_pairs or None,
    )

  def to_structure(self, ccd: chemical_components.Ccd) -> structure.Structure:
    """Converts Input to a Structure.

    WARNING: This method does not preserve the rng seeds.

    Args:
      ccd: The chemical components dictionary.

    Returns:
      The input in a structure.Structure format.
    """
    ids: list[str] = []
    sequences: list[str] = []
    poly_types: list[str] = []
    formats: list[structure.SequenceFormat] = []
    for chain in self.chains:
      ids.append(chain.id)
      match chain:
        case ProteinChain():
          sequences.append('(' + ')('.join(chain.to_ccd_sequence()) + ')')
          poly_types.append(mmcif_names.PROTEIN_CHAIN)
          formats.append(structure.SequenceFormat.CCD_CODES)
        case RnaChain():
          sequences.append('(' + ')('.join(chain.to_ccd_sequence()) + ')')
          poly_types.append(mmcif_names.RNA_CHAIN)
          formats.append(structure.SequenceFormat.CCD_CODES)
        case DnaChain():
          sequences.append('(' + ')('.join(chain.to_ccd_sequence()) + ')')
          poly_types.append(mmcif_names.DNA_CHAIN)
          formats.append(structure.SequenceFormat.CCD_CODES)
        case Ligand():
          if chain.ccd_ids is not None:
            sequences.append('(' + ')('.join(chain.ccd_ids) + ')')
            if len(chain.ccd_ids) == 1:
              poly_types.append(mmcif_names.NON_POLYMER_CHAIN)
            else:
              poly_types.append(mmcif_names.BRANCHED_CHAIN)
            formats.append(structure.SequenceFormat.CCD_CODES)
          elif chain.smiles is not None:
            # Convert to `<unique ligand ID>:<smiles>` format that is expected
            # by structure.from_sequences_and_bonds.
            sequences.append(f'LIG_{chain.id}:{chain.smiles}')
            poly_types.append(mmcif_names.NON_POLYMER_CHAIN)
            formats.append(structure.SequenceFormat.LIGAND_SMILES)
          else:
            raise ValueError('Ligand must have one of CCD ID or SMILES set.')

    # Remap bond chain IDs from chain IDs to chain indices and convert to
    # 0-based residue indexing.
    bonded_atom_pairs = []
    chain_indices = {cid: i for i, cid in enumerate(ids)}
    if self.bonded_atom_pairs is not None:
      for bond_beg, bond_end in self.bonded_atom_pairs:
        bonded_atom_pairs.append((
            (chain_indices[bond_beg[0]], bond_beg[1] - 1, bond_beg[2]),
            (chain_indices[bond_end[0]], bond_end[1] - 1, bond_end[2]),
        ))

    struc = structure.from_sequences_and_bonds(
        sequences=sequences,
        chain_types=poly_types,
        sequence_formats=formats,
        bonded_atom_pairs=bonded_atom_pairs,
        ccd=ccd,
        name=self.sanitised_name(),
        bond_type=mmcif_names.COVALENT_BOND,
        release_date=None,
    )
    # Rename chain IDs to the original ones.
    return struc.rename_chain_ids(dict(zip(struc.chains, ids, strict=True)))

  def to_json(self) -> str:
    """Converts Input to an AlphaFold JSON."""
    alphafold_json = json.dumps(
        {
            'dialect': JSON_DIALECT,
            'version': JSON_VERSION,
            'name': self.name,
            'sequences': [chain.to_dict() for chain in self.chains],
            'modelSeeds': self.rng_seeds,
            'bondedAtomPairs': self.bonded_atom_pairs,
            'userCCD': self.user_ccd,
        },
        indent=2,
    )
    # Remove newlines from the query/template indices arrays. We match the
    # queryIndices/templatesIndices with a non-capturing group. We then match
    # the entire region between the square brackets by looking for lines
    # containing only whitespace, number, or a comma.
    return re.sub(
        r'("(?:queryIndices|templateIndices)": \[)([\s\n\d,]+)(\],?)',
        lambda mtch: mtch[1] + re.sub(r'\n\s+', ' ', mtch[2].strip()) + mtch[3],
        alphafold_json,
    )

  def fill_missing_fields(self) -> Self:
    """Fill missing MSA and template fields with default values."""
    with_missing_fields = [
        c.fill_missing_fields()
        if isinstance(c, (ProteinChain, RnaChain))
        else c
        for c in self.chains
    ]
    return dataclasses.replace(self, chains=with_missing_fields)

  def sanitised_name(self) -> str:
    """Returns sanitised version of the name that can be used as a filename."""
    lower_spaceless_name = self.name.lower().replace(' ', '_')
    allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
    return ''.join(l for l in lower_spaceless_name if l in allowed_chars)


def check_unique_sanitised_names(fold_inputs: Sequence[Input]) -> None:
  """Checks that the names of the fold inputs are unique."""
  names = [fi.sanitised_name() for fi in fold_inputs]
  if len(set(names)) != len(names):
    raise ValueError(
        f'Fold inputs must have unique sanitised names, got {names}.'
    )

def load_fold_input_from_mmcif_path(mmcif_path: pathlib.Path, ccd: chemical_components.Ccd, whole_pdb_config, sampled_chains:List[str], interface_threshold: float = 15.0, max_num_chains: int = 20, only_sampled_chains: bool = False, only_protein_chains: bool = False, valid_chains: list[str] = None) -> Input:
  """Loads the input from an mmCIF string."""
  # Extract the filename without the directory path
  file_name = os.path.basename(mmcif_path)
  # Use a regular expression to extract the mmCIF name
  pdb_id = re.match(r'^([a-zA-Z0-9]+)', file_name).group(1)
  with open(mmcif_path, 'r') as f:
    mmcif_str = f.read()
  return Input.from_mmcif(mmcif_str,pdb_id, ccd, whole_pdb_config,sampled_chains,interface_threshold,max_num_chains,only_sampled_chains,only_protein_chains,valid_chains)

def load_fold_input_from_json_path(json_path: pathlib.Path, only_protein_chains: bool = False, oss_petrel_backend=None) -> Input:
  """Loads the input from a JSON string."""
  with open(json_path, 'r') as f:
    json_str = f.read()
  return Input.from_json(json_str, only_protein_chains, oss_petrel_backend=oss_petrel_backend)

def load_fold_inputs_from_path(json_path: pathlib.Path) -> Sequence[Input]:
  """Loads multiple fold inputs from a JSON string."""
  with open(json_path, 'r') as f:
    json_str = f.read()

  # Parse the JSON string, so we can detect its format.
  raw_json = json.loads(json_str)

  fold_inputs = []
  if isinstance(raw_json, list):
    # AlphaFold Server JSON.
    logging.info(
        'Detected %s is an AlphaFold Server JSON since the top-level is a'
        ' list.',
        json_path,
    )

    logging.info('Loading %d fold jobs from %s', len(raw_json), json_path)
    for fold_job_idx, fold_job in enumerate(raw_json):
      try:
        fold_inputs.append(Input.from_alphafoldserver_fold_job(fold_job))
      except ValueError as e:
        raise ValueError(
            f'Failed to load fold job {fold_job_idx} from {json_path}. The JSON'
            f' at {json_path} was detected to be an AlphaFold Server JSON since'
            ' the top-level is a list.'
        ) from e
  else:
    logging.info(
        'Detected %s is an AlphaFold 3 JSON since the top-level is not a list.',
        json_path,
    )
    # AlphaFold 3 JSON.
    try:
      fold_inputs.append(Input.from_json(json_str))
    except ValueError as e:
      raise ValueError(
          f'Failed to load fold input from {json_path}. The JSON at'
          f' {json_path} was detected to be an AlphaFold 3 JSON since the'
          ' top-level is not a list.'
      ) from e

  check_unique_sanitised_names(fold_inputs)

  return fold_inputs


def load_fold_inputs_from_dir(input_dir: pathlib.Path) -> Sequence[Input]:
  """Loads multiple fold inputs from all JSON files in a given input_dir.

  Args:
    input_dir: The directory containing the JSON files.

  Returns:
    The fold inputs from all JSON files in the input directory.

  Raises:
    ValueError: If the fold inputs have non-unique sanitised names.
  """
  fold_inputs = []
  for file_path in input_dir.glob('*.json'):
    if not file_path.is_file():
      continue

    fold_inputs.extend(load_fold_inputs_from_path(file_path))

  check_unique_sanitised_names(fold_inputs)

  return fold_inputs


def check_input_dict(raw_json):
    for seqs in raw_json['sequences']:
      if seqs.get('protein') is not None:
        if seqs['protein'].get('unpairedMsa') is None:
            unpairedMsaPath = seqs['protein'].get('unpairedMsaPath')
            templatePath = seqs['protein'].get('templatesPath')
            # unpairedMsaPath = unpairedMsaPath.
            if unpairedMsaPath is None or not os.path.exists(unpairedMsaPath):
                raise ValueError('unpairedMsaPath is missing or does not exist')
            else:
                with open(unpairedMsaPath, 'r') as f:
                    seqs['protein']['unpairedMsa'] = f.read()
                seqs['protein']['pairedMsa'] = seqs['protein']['unpairedMsa']
                seqs['protein'].pop('unpairedMsaPath')
            if templatePath is not None:
              if not os.path.exists(templatePath):
                print(f'{templatePath} is not exist. Template = []')
                seqs['protein'].pop('templatesPath')
              else:
                with open(templatePath, 'r') as f:
                  seqs['protein']['templates'] = json.loads(f.read())
                  # seqs['protein']['templates'] = []
                seqs['protein'].pop('templatesPath')
    return raw_json

def check_input_dict_from_bucket(raw_json, oss_petrel_backend=None):
    for seqs in raw_json['sequences']:
      if seqs.get('protein') is not None:
        if seqs['protein'].get('unpairedMsa') is None:
            unpairedMsaPath = seqs['protein'].get('unpairedMsaPath')
            pairedMsaPath = seqs['protein'].get('pairedMsaPath')
            templatePath = seqs['protein'].get('templatesPath')
            if unpairedMsaPath is None:
                raise ValueError('unpairedMsaPath is missing')

            has_bucket_path = any(
                str(path).startswith('s3://')
                for path in (unpairedMsaPath, pairedMsaPath, templatePath)
                if path is not None
            )
            if has_bucket_path:
                raise ValueError(
                    'object-storage paths are not supported in this build; '
                    'give a local path for unpairedMsaPath / pairedMsaPath / '
                    'templatePath, or inline the MSA'
                )

            if str(unpairedMsaPath).startswith('s3://'):
                unpaired_msa = oss_petrel_backend.get_text(unpairedMsaPath)
            elif os.path.exists(unpairedMsaPath):
                with open(unpairedMsaPath, 'r') as f:
                    unpaired_msa = f.read()
            else:
                raise ValueError('unpairedMsaPath is missing or does not exist')

            if pairedMsaPath is None:
                paired_msa = unpaired_msa
            elif str(pairedMsaPath).startswith('s3://'):
                paired_msa = oss_petrel_backend.get_text(pairedMsaPath)
            elif os.path.exists(pairedMsaPath):
                with open(pairedMsaPath, 'r') as f:
                    paired_msa = f.read()
            else:
                paired_msa = unpaired_msa

            seqs['protein']['unpairedMsa'] = unpaired_msa
            seqs['protein']['pairedMsa'] = paired_msa
            seqs['protein'].pop('unpairedMsaPath', None)
            seqs['protein'].pop('pairedMsaPath', None)

            if templatePath is not None:
              if str(templatePath).startswith('s3://'):
                seqs['protein']['templates'] = json.loads(oss_petrel_backend.get_text(templatePath))
                seqs['protein'].pop('templatesPath', None)
              elif os.path.exists(templatePath):
                with open(templatePath, 'r') as f:
                  seqs['protein']['templates'] = json.loads(f.read())
                seqs['protein'].pop('templatesPath', None)
              else:
                print(f'{templatePath} is not exist. Template = []')
                seqs['protein'].pop('templatesPath', None)
    return raw_json
