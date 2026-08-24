# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Functions for running the MSA and template tools for the AlphaFold model."""

from concurrent import futures
import dataclasses
import datetime
import functools
from absl import logging
import time
import json
import os
import io

from alphafold3.common import folding_input
from alphafold3.constants import mmcif_names
from alphafold3.data import msa
from alphafold3.data import msa_config
from alphafold3.data import structure_stores
from alphafold3.data import templates
from alphafold3.data import parsers


# Cache to avoid re-running the MSA tools for the same sequence in homomers.
def _is_remote_path(path: str | None) -> bool:
    return path is not None and str(path).startswith('s3://')


def _read_text(path: str, oss_petrel_backend=None) -> str:
    if _is_remote_path(path):
        if oss_petrel_backend is None:
            raise ValueError(f'Petrel backend is required to read {path}')
        return oss_petrel_backend.get_text(path)
    with open(path, 'r') as f:
        return f.read()


def _convert_stockholm_path_to_a3m(path: str, oss_petrel_backend=None) -> str:
    if _is_remote_path(path):
        if oss_petrel_backend is None:
            raise ValueError(f'Petrel backend is required to read {path}')
        return parsers.convert_stockholm_to_a3m(io.StringIO(oss_petrel_backend.get_text(path)))
    with open(path, 'r') as f:
        return parsers.convert_stockholm_to_a3m(f)


def _list_alignment_files(path: str, oss_petrel_backend=None):
    if _is_remote_path(path):
        if oss_petrel_backend is None:
            raise ValueError(f'Petrel backend is required to list {path}')
        return list(oss_petrel_backend.list_dir_or_file(path, list_dir=False, list_file=True))
    return os.listdir(path)


def _path_is_file(path: str) -> bool:
    return _is_remote_path(path) and path.endswith(('.a3m', '.sto')) or os.path.isfile(path)


@functools.cache
def _get_protein_msa_and_templates(
    sequence: str,
    uniref90_msa_config: msa_config.RunConfig,
    mgnify_msa_config: msa_config.RunConfig,
    small_bfd_msa_config: msa_config.RunConfig,
    uniprot_msa_config: msa_config.RunConfig,
    templates_config: msa_config.TemplatesConfig,
    pdb_database_path: str,
) -> tuple[msa.Msa, msa.Msa, templates.Templates]:
  """Processes a single protein chain."""
  logging.info('Getting protein MSAs for sequence %s', sequence)
  msa_start_time = time.time()
  # Run various MSA tools in parallel. Use a ThreadPoolExecutor because
  # they're not blocked by the GIL, as they're sub-shelled out.
  with futures.ThreadPoolExecutor(max_workers=4) as executor:
    uniref90_msa_future = executor.submit(
        msa.get_msa,
        target_sequence=sequence,
        run_config=uniref90_msa_config,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
    )
    mgnify_msa_future = executor.submit(
        msa.get_msa,
        target_sequence=sequence,
        run_config=mgnify_msa_config,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
    )
    small_bfd_msa_future = executor.submit(
        msa.get_msa,
        target_sequence=sequence,
        run_config=small_bfd_msa_config,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
    )
    uniprot_msa_future = executor.submit(
        msa.get_msa,
        target_sequence=sequence,
        run_config=uniprot_msa_config,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
    )
  uniref90_msa = uniref90_msa_future.result()
  mgnify_msa = mgnify_msa_future.result()
  small_bfd_msa = small_bfd_msa_future.result()
  uniprot_msa = uniprot_msa_future.result()
  logging.info(
      'Getting protein MSAs took %.2f seconds for sequence %s',
      time.time() - msa_start_time,
      sequence,
  )

  logging.info(
      'Deduplicating MSAs and getting protein templates for sequence %s',
      sequence,
  )
  templates_start_time = time.time()
  with futures.ThreadPoolExecutor() as executor:
    unpaired_protein_msa_future = executor.submit(
        msa.Msa.from_multiple_msas,
        msas=[uniref90_msa, small_bfd_msa, mgnify_msa],
        deduplicate=True,
    )
    paired_protein_msa_future = executor.submit(
        msa.Msa.from_multiple_msas, msas=[uniprot_msa], deduplicate=False
    )
    filter_config = templates_config.filter_config
    templates_future = executor.submit(
        templates.Templates.from_seq_and_a3m,
        query_sequence=sequence,
        msa_a3m=uniref90_msa.to_a3m(),
        max_template_date=filter_config.max_template_date,
        database_path=templates_config.template_tool_config.database_path,
        hmmsearch_config=templates_config.template_tool_config.hmmsearch_config,
        max_a3m_query_sequences=None,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
        structure_store=structure_stores.StructureStore(pdb_database_path),
    )
  unpaired_protein_msa = unpaired_protein_msa_future.result()
  paired_protein_msa = paired_protein_msa_future.result()
  protein_templates = templates_future.result()
  logging.info(
      'Deduplicating MSAs and getting protein templates took %.2f seconds for'
      ' sequence %s',
      time.time() - templates_start_time,
      sequence,
  )

  logging.info('Filtering protein templates for sequence %s', sequence)
  filter_start_time = time.time()
  filtered_templates = protein_templates.filter(
      max_subsequence_ratio=filter_config.max_subsequence_ratio,
      min_align_ratio=filter_config.min_align_ratio,
      min_hit_length=filter_config.min_hit_length,
      deduplicate_sequences=filter_config.deduplicate_sequences,
      max_hits=filter_config.max_hits,
  )
  logging.info(
      'Filtering protein templates took %.2f seconds for sequence %s',
      time.time() - filter_start_time,
      sequence,
  )
  return unpaired_protein_msa, paired_protein_msa, filtered_templates

@functools.cache
def _get_protein_msa_and_templates_precomputed(
    sequence: str,
    precomputed_alignment_path: str,
    hmmsearch_a3m_path: str,
    use_templates: bool,
    query_release_date: str,
    templates_filter_config: msa_config.TemplateFilterConfig,
    pdb_database_path: str,
    oss_petrel_backend=None,
) -> tuple[msa.Msa, msa.Msa, templates.Templates]:
  """Processes a single protein chain."""
  logging.info('Getting protein MSAs for sequence %s', sequence)
  msa_start_time = time.time()
  # Load precomputed alignments
  # Iterate over all files in the directory
  other_msas = []       # .a3m
  uniprot_msa = msa.Msa.from_empty(query_sequence=sequence,chain_poly_type=mmcif_names.PROTEIN_CHAIN,) # uniprot_*.a3m    
  uniref_msa = msa.Msa.from_empty(query_sequence=sequence,chain_poly_type=mmcif_names.PROTEIN_CHAIN,) # uniref90_*.a3m
  bfd_msa = msa.Msa.from_empty(query_sequence=sequence,chain_poly_type=mmcif_names.PROTEIN_CHAIN,) # bfd_*.a3m
  mgnify_msa = msa.Msa.from_empty(query_sequence=sequence,chain_poly_type=mmcif_names.PROTEIN_CHAIN,) # mgnify_*.a3m
  colabfold_msa = msa.Msa.from_empty(query_sequence=sequence,chain_poly_type=mmcif_names.PROTEIN_CHAIN,) # colabfold_*.a3m
  
  # if it is already a file, not a directory
  if _path_is_file(precomputed_alignment_path):
    if precomputed_alignment_path.endswith('.a3m'):
        a3m_string = _read_text(precomputed_alignment_path, oss_petrel_backend)
        colabfold_msa = msa.Msa.from_a3m(
            query_sequence=sequence,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            a3m=a3m_string,
            max_depth=None,
            deduplicate=False,
        )
    else:
        raise ValueError('We only support a3m files for now')
  else:       
    # if it is a directory
    for file in _list_alignment_files(precomputed_alignment_path, oss_petrel_backend):
        if file.startswith('uniprot_'):
            if file.endswith('.a3m'):
                a3m_string = _read_text(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            elif file.endswith('sto'):
                a3m_string = _convert_stockholm_path_to_a3m(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
                
            uniprot_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=a3m_string,
                max_depth=None,
                deduplicate=False,
            )
        elif file.startswith('uniref90_'):
            if file.endswith('.a3m'):
                a3m_string = _read_text(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            elif file.endswith('.sto'):
                a3m_string = _convert_stockholm_path_to_a3m(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            
            uniref_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=a3m_string,
                max_depth=10000,
                deduplicate=False,
            )
            
        elif file.startswith('bfd_'):
            if file.endswith('.a3m'):
                a3m_string = _read_text(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            elif file.endswith('.sto'):
                a3m_string = _convert_stockholm_path_to_a3m(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            bfd_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=a3m_string,
                max_depth=5000,
                deduplicate=False,
            )

        elif file.startswith('mgnify_'):
            if file.endswith('.a3m'):
                a3m_string = _read_text(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            elif file.endswith('.sto'):
                a3m_string = _convert_stockholm_path_to_a3m(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            mgnify_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=a3m_string,
                max_depth=5000,
                deduplicate=False,
            )
        elif file.endswith('.a3m'):
            a3m_string = _read_text(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
            colabfold_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=a3m_string,
                max_depth=None,
                deduplicate=False,
            )
        
  logging.info(
    'Getting protein MSAs took %.2f seconds for sequence %s',
    time.time() - msa_start_time,
    sequence,
  )
  
  logging.info(
    'Deduplicating MSAs and getting protein templates for sequence %s',
    sequence,
  )

  if colabfold_msa.depth == 1:
      other_msas = [uniref_msa, bfd_msa, mgnify_msa]
  else:
      other_msas = [colabfold_msa]
      
  template_start_time = time.time()
  unpaired_protein_msa = msa.Msa.from_multiple_msas(
    msas = other_msas,
    deduplicate = True,
  )
  paired_protein_msa = msa.Msa.from_multiple_msas(
    msas = [uniprot_msa],
    deduplicate = False,
  )
  protein_templates = None
  if use_templates:
      if hmmsearch_a3m_path is None:
          protein_templates = templates.Templates(
              query_sequence = sequence,
              hits = [],
              max_template_date=templates_filter_config.max_template_date,
              structure_store=structure_stores.StructureStore(pdb_database_path),
          )
      else:
          hmmsearch_a3m_string = _read_text(os.path.join(hmmsearch_a3m_path, 'hmmsearch.a3m'), oss_petrel_backend)
              
          protein_templates = templates.Templates.from_hmmsearch_a3m(
              query_sequence = sequence,
              a3m = hmmsearch_a3m_string,
              max_template_date=templates_filter_config.max_template_date,
              query_release_date=datetime.datetime.strptime(query_release_date, "%Y-%m-%d").date(),
              chain_poly_type=mmcif_names.PROTEIN_CHAIN,
              structure_store=structure_stores.StructureStore(pdb_database_path),
              filter_config=templates_filter_config,
          )
  logging.info(
    'Deduplicating MSAs and getting protein templates took %.2f seconds for sequence %s',
    time.time() - template_start_time,
    sequence,
  )
  
  return unpaired_protein_msa, paired_protein_msa, protein_templates
  
      

            
# Cache to avoid re-running the Nhmmer for the same sequence in homomers.
@functools.cache
def _get_rna_msa(
    sequence: str,
    nt_rna_msa_config: msa_config.NhmmerConfig,
    rfam_msa_config: msa_config.NhmmerConfig,
    rnacentral_msa_config: msa_config.NhmmerConfig,
) -> msa.Msa:
  """Processes a single RNA chain."""
  logging.info('Getting RNA MSAs for sequence %s', sequence)
  rna_msa_start_time = time.time()
  # Run various MSA tools in parallel. Use a ThreadPoolExecutor because
  # they're not blocked by the GIL, as they're sub-shelled out.
  with futures.ThreadPoolExecutor() as executor:
    nt_rna_msa_future = executor.submit(
        msa.get_msa,
        target_sequence=sequence,
        run_config=nt_rna_msa_config,
        chain_poly_type=mmcif_names.RNA_CHAIN,
    )
    rfam_msa_future = executor.submit(
        msa.get_msa,
        target_sequence=sequence,
        run_config=rfam_msa_config,
        chain_poly_type=mmcif_names.RNA_CHAIN,
    )
    rnacentral_msa_future = executor.submit(
        msa.get_msa,
        target_sequence=sequence,
        run_config=rnacentral_msa_config,
        chain_poly_type=mmcif_names.RNA_CHAIN,
    )
  nt_rna_msa = nt_rna_msa_future.result()
  rfam_msa = rfam_msa_future.result()
  rnacentral_msa = rnacentral_msa_future.result()
  logging.info(
      'Getting RNA MSAs took %.2f seconds for sequence %s',
      time.time() - rna_msa_start_time,
      sequence,
  )

  return msa.Msa.from_multiple_msas(
      msas=[rfam_msa, rnacentral_msa, nt_rna_msa],
      deduplicate=True,
  )

@functools.cache
def _get_rna_msa_precomputed(
    sequence: str,
    precomputed_alignment_path: str,
    oss_petrel_backend=None,
) -> msa.Msa:
  """Processes a single RNA chain."""
  logging.info('Getting RNA MSAs for sequence %s', sequence)
  rna_msa_start_time = time.time()
  # Load precomputed alignments
  # Iterate over all files in the directory
  msas = []       # .a3m
  for file in _list_alignment_files(precomputed_alignment_path, oss_petrel_backend):
    if file.endswith('.a3m'):
        a3m_string = _read_text(os.path.join(precomputed_alignment_path, file), oss_petrel_backend)
        rna_msa = msa.Msa.from_a3m(
            query_sequence=sequence,
            chain_poly_type=mmcif_names.RNA_CHAIN,
            a3m=a3m_string,
            max_depth=None,
            deduplicate=False,
        )
        msas.append(rna_msa)
  logging.info(
    'Getting RNA MSAs took %.2f seconds for sequence %s',
    time.time() - rna_msa_start_time,
    sequence,
  )
  
  return msa.Msa.from_multiple_msas(
      msas=msas,
      deduplicate=True,
  )



@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DataPipelineConfig:
  """The configuration for the data pipeline.

  Attributes:
    jackhmmer_binary_path: Jackhmmer binary path, used for protein MSA search.
    nhmmer_binary_path: Nhmmer binary path, used for RNA MSA search.
    hmmalign_binary_path: Hmmalign binary path, used to align hits to the query
      profile.
    hmmsearch_binary_path: Hmmsearch binary path, used for template search.
    hmmbuild_binary_path: Hmmbuild binary path, used to build HMM profile from
      raw MSA in template search.
    small_bfd_database_path: Small BFD database path, used for protein MSA
      search.
    mgnify_database_path: Mgnify database path, used for protein MSA search.
    uniprot_cluster_annot_database_path: Uniprot database path, used for protein
      paired MSA search.
    uniref90_database_path: UniRef90 database path, used for MSA search, and the
      MSA obtained by searching it is used to construct the profile for template
      search.
    ntrna_database_path: NT-RNA database path, used for RNA MSA search.
    rfam_database_path: Rfam database path, used for RNA MSA search.
    rna_central_database_path: RNAcentral database path, used for RNA MSA
      search.
    seqres_database_path: PDB sequence database path, used for template search.
    pdb_database_path: PDB database directory with mmCIF files path, used for
      template search.
    jackhmmer_n_cpu: Number of CPUs to use for Jackhmmer.
    nhmmer_n_cpu: Number of CPUs to use for Nhmmer.
    use_precomputed_alignments: Whether to use precomputed alignments.
    precomputed_alignments_path: Path to the directory containing precomputed
    sequence_to_precomputed_alignment_id_mapping_path: Path to the file containing the mapping from sequence to precomputed alignment id.
    use_templates: Whether to use templates.
  """

  # Binary paths.
  jackhmmer_binary_path: str |None = None
  nhmmer_binary_path: str |None = None
  hmmalign_binary_path: str|None = None
  hmmsearch_binary_path: str|None = None
  hmmbuild_binary_path: str|None = None

  # Jackhmmer databases.
  small_bfd_database_path: str|None = None
  mgnify_database_path: str|None = None
  uniprot_cluster_annot_database_path: str|None = None
  uniref90_database_path: str|None = None
  # Nhmmer databases.
  ntrna_database_path: str|None = None
  rfam_database_path: str|None = None
  rna_central_database_path: str|None = None
  # Template search databases.
  seqres_database_path: str|None = None
  pdb_database_path: str|None = None

  # Optional configuration for MSA tools.
  jackhmmer_n_cpu: int = 8
  nhmmer_n_cpu: int = 8
  
  # Use Precomputed Alignments
  use_precomputed_alignments: bool = False
  precomputed_alignments_path: str = None
  sequence_to_precomputed_alignment_id_mapping_path: str = None
  use_templates: bool = False
  precomputed_templates_path: str = None
  sequence_to_precomputed_template_id_mapping_path: str = None
  oss_petrel_backend: object = None


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self, data_pipeline_config: DataPipelineConfig):
    """Initializes the data pipeline with default configurations."""
    self.use_precomputed_alignments = data_pipeline_config.use_precomputed_alignments
    self.oss_petrel_backend = data_pipeline_config.oss_petrel_backend
    if self.use_precomputed_alignments:
        self.precomputed_alignments_path = data_pipeline_config.precomputed_alignments_path
        self.sequence_to_precomputed_alignment_id_mapping_path = data_pipeline_config.sequence_to_precomputed_alignment_id_mapping_path
        self.sequence_to_precomputed_alignment_id_mapping = json.loads(
            _read_text(self.sequence_to_precomputed_alignment_id_mapping_path, self.oss_petrel_backend)
        )
        self.use_templates = data_pipeline_config.use_templates
        if self.use_templates:
            self.precomputed_templates_path = data_pipeline_config.precomputed_templates_path
            self.sequence_to_precomputed_template_id_mapping_path = data_pipeline_config.sequence_to_precomputed_template_id_mapping_path
            self.sequence_to_precomputed_template_id_mapping = json.loads(
                _read_text(self.sequence_to_precomputed_template_id_mapping_path, self.oss_petrel_backend)
            )
            self._templates_filter_config = msa_config.TemplateFilterConfig(
                max_subsequence_ratio=0.95,
                min_align_ratio=0.1,
                min_hit_length=10,
                deduplicate_sequences=True,
                max_hits=4,
                # By default, use the date from AF3 paper.
                max_template_date=datetime.date(2021, 9, 30),
            )
            self._pdb_database_path = data_pipeline_config.pdb_database_path
    
    else:
        raise ValueError('We use precomputed alignments only for now')
        self._uniref90_msa_config = msa_config.RunConfig(
            config=msa_config.JackhmmerConfig(
                binary_path=data_pipeline_config.jackhmmer_binary_path,
                database_config=msa_config.DatabaseConfig(
                    name='uniref90',
                    path=data_pipeline_config.uniref90_database_path,
                ),
                n_cpu=data_pipeline_config.jackhmmer_n_cpu,
                n_iter=1,
                e_value=1e-4,
                z_value=None,
                max_sequences=10_000,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )
        self._mgnify_msa_config = msa_config.RunConfig(
            config=msa_config.JackhmmerConfig(
                binary_path=data_pipeline_config.jackhmmer_binary_path,
                database_config=msa_config.DatabaseConfig(
                    name='mgnify',
                    path=data_pipeline_config.mgnify_database_path,
                ),
                n_cpu=data_pipeline_config.jackhmmer_n_cpu,
                n_iter=1,
                e_value=1e-4,
                z_value=None,
                max_sequences=5_000,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )
        self._small_bfd_msa_config = msa_config.RunConfig(
            config=msa_config.JackhmmerConfig(
                binary_path=data_pipeline_config.jackhmmer_binary_path,
                database_config=msa_config.DatabaseConfig(
                    name='small_bfd',
                    path=data_pipeline_config.small_bfd_database_path,
                ),
                n_cpu=data_pipeline_config.jackhmmer_n_cpu,
                n_iter=1,
                e_value=1e-4,
                # Set z_value=138_515_945 to match the z_value used in the paper.
                # In practice, this has minimal impact on predicted structures.
                z_value=None,
                max_sequences=5_000,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )
        self._uniprot_msa_config = msa_config.RunConfig(
            config=msa_config.JackhmmerConfig(
                binary_path=data_pipeline_config.jackhmmer_binary_path,
                database_config=msa_config.DatabaseConfig(
                    name='uniprot_cluster_annot',
                    path=data_pipeline_config.uniprot_cluster_annot_database_path,
                ),
                n_cpu=data_pipeline_config.jackhmmer_n_cpu,
                n_iter=1,
                e_value=1e-4,
                z_value=None,
                max_sequences=50_000,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )
        self._nt_rna_msa_config = msa_config.RunConfig(
            config=msa_config.NhmmerConfig(
                binary_path=data_pipeline_config.nhmmer_binary_path,
                hmmalign_binary_path=data_pipeline_config.hmmalign_binary_path,
                hmmbuild_binary_path=data_pipeline_config.hmmbuild_binary_path,
                database_config=msa_config.DatabaseConfig(
                    name='nt_rna',
                    path=data_pipeline_config.ntrna_database_path,
                ),
                n_cpu=data_pipeline_config.nhmmer_n_cpu,
                e_value=1e-3,
                alphabet='rna',
                max_sequences=10_000,
            ),
            chain_poly_type=mmcif_names.RNA_CHAIN,
            crop_size=None,
        )
        self._rfam_msa_config = msa_config.RunConfig(
            config=msa_config.NhmmerConfig(
                binary_path=data_pipeline_config.nhmmer_binary_path,
                hmmalign_binary_path=data_pipeline_config.hmmalign_binary_path,
                hmmbuild_binary_path=data_pipeline_config.hmmbuild_binary_path,
                database_config=msa_config.DatabaseConfig(
                    name='rfam_rna',
                    path=data_pipeline_config.rfam_database_path,
                ),
                n_cpu=data_pipeline_config.nhmmer_n_cpu,
                e_value=1e-3,
                alphabet='rna',
                max_sequences=10_000,
            ),
            chain_poly_type=mmcif_names.RNA_CHAIN,
            crop_size=None,
        )
        self._rnacentral_msa_config = msa_config.RunConfig(
            config=msa_config.NhmmerConfig(
                binary_path=data_pipeline_config.nhmmer_binary_path,
                hmmalign_binary_path=data_pipeline_config.hmmalign_binary_path,
                hmmbuild_binary_path=data_pipeline_config.hmmbuild_binary_path,
                database_config=msa_config.DatabaseConfig(
                    name='rna_central_rna',
                    path=data_pipeline_config.rna_central_database_path,
                ),
                n_cpu=data_pipeline_config.nhmmer_n_cpu,
                e_value=1e-3,
                alphabet='rna',
                max_sequences=10_000,
            ),
            chain_poly_type=mmcif_names.RNA_CHAIN,
            crop_size=None,
        )

        self._templates_config = msa_config.TemplatesConfig(
            template_tool_config=msa_config.TemplateToolConfig(
                database_path=data_pipeline_config.seqres_database_path,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                hmmsearch_config=msa_config.HmmsearchConfig(
                    hmmsearch_binary_path=data_pipeline_config.hmmsearch_binary_path,
                    hmmbuild_binary_path=data_pipeline_config.hmmbuild_binary_path,
                    filter_f1=0.1,
                    filter_f2=0.1,
                    filter_f3=0.1,
                    e_value=100,
                    inc_e=100,
                    dom_e=100,
                    incdom_e=100,
                    alphabet='amino',
                ),
            ),
            filter_config=msa_config.TemplateFilterConfig(
                max_subsequence_ratio=0.95,
                min_align_ratio=0.1,
                min_hit_length=10,
                deduplicate_sequences=True,
                max_hits=4,
                # By default, use the date from AF3 paper.
                max_template_date=datetime.date(2021, 9, 30),
            ),
        )
        self._pdb_database_path = data_pipeline_config.pdb_database_path

  def process_protein_chain(
      self, chain: folding_input.ProteinChain, query_release_date: str
  ) -> folding_input.ProteinChain:
    """Processes a single protein chain."""
    if chain.unpaired_msa:
      logging.info(f'Skipping MSA loading or searching for protein chain {chain.id} because it already has MSA.')
      if not chain.paired_msa:
          empty_msa = msa.Msa.from_empty(
          query_sequence=chain.sequence, chain_poly_type=mmcif_names.PROTEIN_CHAIN
      ).to_a3m()
          chain = dataclasses.replace(chain, paired_msa=empty_msa)
          logging.info(f'Paired MSA is empty for chain {chain.id}, set it to empty')
      return chain

    if self.use_precomputed_alignments:
        try:
            skip_loading = False
            try:
                precomputed_alignment_id = self.sequence_to_precomputed_alignment_id_mapping[chain.sequence]
            except KeyError:
                precomputed_alignment_id = self.sequence_to_precomputed_alignment_id_mapping[chain.unmapped_sequence]
        except KeyError:
            raise(ValueError(f'No precomputed alignment found for sequence {chain.sequence}'))
            logging.warning(f'No precomputed alignment found for sequence {chain.sequence}')
            unpaired_msa = msa.Msa.from_empty(
                query_sequence=chain.sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            )
            paired_msa = msa.Msa.from_empty(
                query_sequence=chain.sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            )
            template_hits = None
            skip_loading = True
        if self.use_templates:
            try:
                try:
                    precomputed_template_id = self.sequence_to_precomputed_template_id_mapping[chain.sequence]
                except KeyError:
                    precomputed_template_id = self.sequence_to_precomputed_template_id_mapping[chain.unmapped_sequence]
                use_templates_this_chain = True
            except KeyError:
                raise(ValueError(f'No precomputed template found for sequence {chain.sequence}'))
                logging.warning(f'No precomputed template found for sequence {chain.sequence}')
                template_hits = None
                use_templates_this_chain = False
                
        if not skip_loading:
            precomputed_alignment_path = os.path.join(self.precomputed_alignments_path, str(precomputed_alignment_id))
            precomputed_template_path = os.path.join(self.precomputed_templates_path, str(precomputed_template_id)) if (self.use_templates and use_templates_this_chain) else None
            unpaired_msa, paired_msa, template_hits = _get_protein_msa_and_templates_precomputed(
            sequence=chain.sequence,
            precomputed_alignment_path=precomputed_alignment_path,
            hmmsearch_a3m_path=precomputed_template_path,
            use_templates=self.use_templates and use_templates_this_chain,
            query_release_date=query_release_date,
            templates_filter_config=self._templates_filter_config if self.use_templates else None,
            pdb_database_path=self._pdb_database_path if self.use_templates else None,
            oss_petrel_backend=self.oss_petrel_backend,
        )

        
    else:
        unpaired_msa, paired_msa, template_hits = _get_protein_msa_and_templates(
            sequence=chain.sequence,
            uniref90_msa_config=self._uniref90_msa_config,
            mgnify_msa_config=self._mgnify_msa_config,
            small_bfd_msa_config=self._small_bfd_msa_config,
            uniprot_msa_config=self._uniprot_msa_config,
            templates_config=self._templates_config,
            pdb_database_path=self._pdb_database_path,
        )

    return dataclasses.replace(
        chain,
        unpaired_msa=unpaired_msa.to_a3m(),
        paired_msa=paired_msa.to_a3m(),
        templates=[
            folding_input.Template(
                mmcif=struc.to_mmcif(),
                query_to_template_map=hit.query_to_hit_mapping,
            )
            for hit, struc in template_hits.get_hits_with_structures()
        ] if template_hits is not None else [],
    )

  def process_rna_chain(
      self, chain: folding_input.RnaChain
  ) -> folding_input.RnaChain:
    """Processes a single RNA chain."""
    if chain.unpaired_msa is not None:
      # Don't run MSA tools if the chain already has an MSA.
      logging.info(
          'Skipping MSA loading or searching for RNA chain %s because it already has MSA.',
          chain.id,
      )
      return chain
    if self.use_precomputed_alignments:
        try:
            skip_loading = False
            try:
                precomputed_alignment_id = self.sequence_to_precomputed_alignment_id_mapping[chain.sequence]
            except KeyError:
                precomputed_alignment_id = self.sequence_to_precomputed_alignment_id_mapping[chain.unmapped_sequence]
        except KeyError:
            #! No strict check for rna chain
            logging.warning(f'No precomputed alignment found for sequence {chain.sequence}')
            rna_msa = msa.Msa.from_empty(
                query_sequence=chain.sequence,
                chain_poly_type=mmcif_names.RNA_CHAIN,
            )
            skip_loading = True
        if not skip_loading:  
            precomputed_alignment_path = os.path.join(self.precomputed_alignments_path, str(precomputed_alignment_id))
            rna_msa = _get_rna_msa_precomputed(
                sequence=chain.sequence,
                precomputed_alignment_path=precomputed_alignment_path,
                oss_petrel_backend=self.oss_petrel_backend,
            )
    else:
        rna_msa = _get_rna_msa(
            sequence=chain.sequence,
            nt_rna_msa_config=self._nt_rna_msa_config,
            rfam_msa_config=self._rfam_msa_config,
            rnacentral_msa_config=self._rnacentral_msa_config,
        )
    return dataclasses.replace(chain, unpaired_msa=rna_msa.to_a3m())

  def process(self, fold_input: folding_input.Input, release_date: str | None = None) -> folding_input.Input:
    """Runs MSA and template tools and returns a new Input with the results."""
    processed_chains = []
    for chain in fold_input.chains:
      logging.info(f'Processing chain {chain.id}')
      process_chain_start_time = time.time()
      match chain:
        case folding_input.ProteinChain():
          processed_chains.append(self.process_protein_chain(chain, release_date))
        case folding_input.RnaChain():
          processed_chains.append(self.process_rna_chain(chain))
        case _:
          processed_chains.append(chain)
      logging.info(
          f'Processing chain {chain.id} took'
          f' {time.time() - process_chain_start_time:.2f} seconds',
      )

    return dataclasses.replace(fold_input, chains=processed_chains)
