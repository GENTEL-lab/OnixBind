from __future__ import annotations

import os
from typing import Dict, Iterator, List, Literal, Tuple, Any

import numpy as np
import polars as pl
from absl import logging
from torch.utils.data import Sampler

# constants

CLUSTERING_RESIDUE_MOLECULE_TYPE = Literal["protein", "rna", "dna",'nucleic', "ligand", "peptide"]

def get_chain_count(molecule_type: CLUSTERING_RESIDUE_MOLECULE_TYPE) -> Tuple[int, int, int]:
    """
    Returns the number of protein (or `peptide`), nucleic acid (i.e., `rna` or `dna`), and
    ligand chains in a molecule based on its type.

    Example:
        n_prot, n_nuc, n_ligand = get_chain_count("protein")
    """
    match molecule_type:
        case "protein":
            return 1, 0, 0
        case "rna":
            return 0, 1, 0
        case "dna":
            return 0, 1, 0
        case "nucleic":
            return 0, 1, 0
        case "ligand":
            return 0, 0, 1
        case "peptide":
            return 1, 0, 0
        case _:
            raise ValueError(f"Unknown molecule type: {molecule_type}")

# TODO: We need to vectorized the weight Calculation

def calculate_weight(
    alphas: Dict[str, float],
    beta: float,
    n_prot: int,
    n_nuc: int,
    n_ligand: int,
    cluster_size: int,
) -> float:
    """
    Calculates the weight of a chain or an interface according to the formula
    provided in Section 2.5.1 of the AlphaFold 3 supplementary materials.
    """
    return (beta / cluster_size) * (
        alphas["prot"] * n_prot + alphas["nuc"] * n_nuc + alphas["ligand"] * n_ligand
    )



def get_chain_weight(
    molecule_type: CLUSTERING_RESIDUE_MOLECULE_TYPE,
    cluster_size: int,
    alphas: Dict[str, float],
    beta: float,
) -> float:
    """Calculates the weight of a chain based on its type."""
    n_prot, n_nuc, n_ligand = get_chain_count(molecule_type)
    return calculate_weight(alphas, beta, n_prot, n_nuc, n_ligand, cluster_size)



def get_interface_weight(
    molecule_type_1: CLUSTERING_RESIDUE_MOLECULE_TYPE,
    molecule_type_2: CLUSTERING_RESIDUE_MOLECULE_TYPE,
    cluster_size: int,
    alphas: Dict[str, float],
    beta: float,
) -> float:
    """Calculates the weight of an interface based on the types of the two molecules."""
    p1, n1, l1 = get_chain_count(molecule_type_1)
    p2, n2, l2 = get_chain_count(molecule_type_2)

    n_prot = p1 + p2
    n_nuc = n1 + n2
    n_ligand = l1 + l2

    return calculate_weight(alphas, beta, n_prot, n_nuc, n_ligand, cluster_size)



def get_cluster_sizes(
    mapping: pl.DataFrame,
    cluster_id_col: str,
) -> Dict[int, int]:
    """
    Returns a dictionary where keys are cluster IDs and values are the number
    of chains/interfaces in the cluster.
    """
    cluster_sizes = mapping.group_by(cluster_id_col).agg(pl.len()).sort(cluster_id_col)
    return {row[0]: row[1] for row in cluster_sizes.iter_rows()}



def compute_chain_weights(
    chains: pl.DataFrame, alphas: Dict[str, float], beta: float
) -> pl.Series:
    """Computes the weights of the chains based on the cluster sizes."""
    molecule_idx = chains.get_column_index("molecule_id")
    cluster_idx = chains.get_column_index("cluster_id")
    cluster_sizes = get_cluster_sizes(chains, "cluster_id")

    return (
        chains.map_rows(
            lambda row: get_chain_weight(
                row[molecule_idx].split("-")[0],
                cluster_sizes[row[cluster_idx]],
                alphas,
                beta,
            ),
            return_dtype=pl.Float32,
        )
        .to_series(0)
        .rename("weight")
    )



def compute_interface_weights(
    interfaces: pl.DataFrame, alphas: Dict[str, float], beta: float
) -> pl.Series:
    """Computes the weights of the interfaces based on the chain weights."""
    molecule_idx_1 = interfaces.get_column_index("interface_chain_type_1")
    molecule_idx_2 = interfaces.get_column_index("interface_chain_type_2")
    cluster_idx = interfaces.get_column_index("interface_cluster_id")
    cluster_sizes = get_cluster_sizes(interfaces, "interface_cluster_id")

    return (
        interfaces.map_rows(
            lambda row: get_interface_weight(
                row[molecule_idx_1].split("-")[0],
                row[molecule_idx_2].split("-")[0],
                cluster_sizes[row[cluster_idx]],
                alphas,
                beta,
            ),
            return_dtype=pl.Float32,
        )
        .to_series(0)
        .rename("weight")
    )


class WeightedPDBSampler():
    """
    Initializes a sampler for weighted sampling of PDB and chain/interface IDs.

    :param chain_mapping_paths: Path to the CSV file containing chain cluster
        mappings. If multiple paths are provided, they will be concatenated.
    :param interface_mapping_path: Path to the CSV file containing interface
        cluster mappings.
    :param batch_size: Number of PDB IDs to sample in each batch.
    :param beta_chain: Weighting factor for chain clusters.
    :param beta_interface: Weighting factor for interface clusters.
    :param alpha_prot: Weighting factor for protein chains.
    :param alpha_nuc: Weighting factor for nucleic acid chains.
    :param alpha_ligand: Weighting factor for ligand chains.
    
    Example:
    ```
    sampler = WeightedPDBSampler(...)
    for batch in sampler:
        print(batch)
    ```
    """

    def __init__(
        self,
        chain_mapping_paths: str | List[str],
        interface_mapping_paths: str | List[str],
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
        alpha_prot: float = 3.0,
        alpha_nuc: float = 3.0,
        alpha_ligand: float = 1.0,
        pdb_ids_to_skip: List[str] = [],
    ):
                # Calculate weights for chains and interfaces
        self.alphas = {"prot": alpha_prot, "nuc": alpha_nuc, "ligand": alpha_ligand}
        self.betas = {"chain": beta_chain, "interface": beta_interface}
            
        # Load chain and interface mappings
        if not isinstance(chain_mapping_paths, list):
            chain_mapping_paths = [chain_mapping_paths]
        
        chain_mapping = []
        for path in chain_mapping_paths:
            molecule_id = os.path.basename(path).split('_')[0]
            df = pl.read_csv(path, null_values=['nan', '']).with_columns([pl.lit(molecule_id).alias("molecule_id")])
            chain_mapping.append(df)
            
        # Increment chain cluster IDs to avoid overlap
        chain_cluster_nums = [mapping.get_column("cluster_id").max() for mapping in chain_mapping]
        for i in range(1, len(chain_mapping)):
            chain_mapping[i] = chain_mapping[i].with_columns(
                (pl.col("cluster_id") + sum(chain_cluster_nums[:i])).alias("cluster_id")
            )
        if len(chain_mapping) != 0:
            chain_mapping = pl.concat(chain_mapping)
            use_chain_mapping = True
        else:
            use_chain_mapping = False
        
        interface_mapping = []
        for path in interface_mapping_paths:
            molecule_id = os.path.basename(path).split('_')[0]
            df = pl.read_csv(path, null_values=['nan', '']).with_columns([pl.lit(molecule_id).alias("molecule_id")])
            interface_mapping.append(df)
            
        # Increment interface cluster IDs to avoid overlap
        interface_cluster_nums = [mapping.get_column("interface_cluster_id").max() for mapping in interface_mapping]
        for i in range(1, len(interface_mapping)):
            interface_mapping[i] = interface_mapping[i].with_columns(
                (pl.col("interface_cluster_id") + sum(interface_cluster_nums[:i])).alias("interface_cluster_id")
            )
        if len(interface_mapping) != 0:
            interface_mapping = pl.concat(interface_mapping)
            use_interface_mapping = True
        else:
            use_interface_mapping = False
            
        if use_interface_mapping == False and use_chain_mapping == False:
            raise ValueError("At least one of chain mapping and interface mapping must be provided")
    
        logging.info(
            "Precomputing chain and interface weights. This may take several minutes to complete."
        )
        
        self.mappings = []
        
        if use_chain_mapping:
            chain_mapping = chain_mapping.fill_null("NA")
            
            # Filter out unwanted PDB IDs
            if len(pdb_ids_to_skip) > 0:
                chain_mapping = chain_mapping.filter(pl.col("pdb_id").is_in(pdb_ids_to_skip).not_())

            chain_mapping.insert_column(
                len(chain_mapping.columns),
                compute_chain_weights(chain_mapping, self.alphas, self.betas["chain"]),
            )

            # Concatenate chain and interface mappings
            chain_mapping = chain_mapping.with_columns(
                [
                    pl.col("chain_id").alias("chain_id_1"),
                    pl.lit("").alias("chain_id_2"),
                ]
            )
            chain_mapping = chain_mapping.select(
                ["pdb_id", "chain_id_1", "chain_id_2", "cluster_id", "weight"]
            )
            
            
            self.mappings.append(chain_mapping)



        if use_interface_mapping:
            interface_mapping = interface_mapping.fill_null("NA")
            
            if len(pdb_ids_to_skip) > 0:
                interface_mapping = interface_mapping.filter(
                        pl.col("pdb_id").is_in(pdb_ids_to_skip).not_()
                    )
            
            interface_mapping.insert_column(
                len(interface_mapping.columns),
                compute_interface_weights(interface_mapping, self.alphas, self.betas["interface"]),
            )

            
            interface_mapping = interface_mapping.with_columns(
                [
                    pl.col("interface_chain_id_1").alias("chain_id_1"),
                    pl.col("interface_chain_id_2").alias("chain_id_2"),
                    (
                        pl.col("interface_cluster_id") + chain_mapping.get_column("cluster_id").max() if use_chain_mapping else pl.col("interface_cluster_id")
                    ).alias("cluster_id"),
                ]
            )
            interface_mapping = interface_mapping.select(
                ["pdb_id", "chain_id_1", "chain_id_2", "cluster_id", "weight"]
            )
            
            
            self.mappings.append(interface_mapping)
            
        logging.info("Finished precomputing chain and interface weights.")
        
        self.mappings = pl.concat(self.mappings)

        # Normalize weights
        self.weights = self.mappings.get_column("weight").to_numpy()
        self.weights = self.weights / self.weights.sum()
    
    def sample(self, num_samples: int) -> List[Tuple[str, str, str]]:
        """Samples a chain ID or interface ID based on the weights of the chains/interfaces."""
        indices = np.random.choice(len(self.mappings), size=num_samples, p=self.weights)
        return self.mappings[indices].select(["pdb_id", "chain_id_1", "chain_id_2"]).rows()
