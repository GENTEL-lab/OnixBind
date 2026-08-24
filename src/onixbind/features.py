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
"""Build model input features from an AlphaFold 3 JSON record.

This is the pipeline the released weights were fit on, vendored under
``runtime/`` and driven here with the settings the reference deployment uses.
Substituting a different featuriser was tried and measured: the model
reproduced the reference to within 0.03 pX when fed the reference's own
features, and disagreed by 0.23 pX when fed features built another way.  The
featuriser is the thing that has to match.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

# Cropping is off for inference; these are the reference deployment's values.
TOKEN_CROP_SIZE = 768
MSA_CROP_SIZE = 4096
MSA_SAMPLE_SIZE = 2048

# Applied to the raw features before the model sees them, exactly as the
# reference runtime does.
ONE_HOT_CLASSES = {
    "template_aatype": 31,
    "msa": 32,
    "aatype": 31,
    "ref_element": 128,
    "ref_atom_name_chars": 64,
}


def runtime_root() -> Path:
    return Path(__file__).resolve().parents[2] / "runtime"


def prepare_import_path(runtime_dir: Path | None = None) -> Path:
    """Put the vendored AF3 pipeline on sys.path before importing it."""
    runtime_dir = Path(runtime_dir) if runtime_dir else runtime_root()
    if not (runtime_dir / "alphafold3").is_dir():
        raise FileNotFoundError(f"vendored AF3 pipeline not found under {runtime_dir}")
    # the vendored cpp extension is built against a newer C++ ABI than some
    # hosts ship; load the newest available runtime before importing it
    from onixbind.native import preload_libstdcxx
    preload_libstdcxx()
    value = str(runtime_dir)
    if value not in sys.path:
        sys.path.insert(0, value)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return runtime_dir


class FeatureProcessor:
    """Turn one AF3 JSON file into the model's input features.

    Token and ligand cropping are refused rather than silently applied: a
    cropped record would score something other than the complex that was asked
    for.
    """

    def __init__(self, runtime_dir: Path | None = None,
                 alignment_dir: str | None = None,
                 seq_to_msa_mapping: str | None = None):
        root = prepare_import_path(runtime_dir)
        import pandas as pd
        from alphafold3.data.data_module_affinity import AlphaFoldDataSet

        self._pd = pd
        placeholder = pd.DataFrame(
            [{"ID": "placeholder", "INPUT_CACHE_ID": "placeholder", "UNIPROT_ID": "placeholder"}]
        )
        self._dataset = AlphaFoldDataSet(
            sampler_df=placeholder,
            data_cache_path="onixbind",
            input_cache_dir="/",
            # AF3 records carry their MSA inline; these only matter for the
            # legacy precomputed-alignment layout, but the constructor reads
            # the mapping file, so it must exist.
            alignment_path=str(alignment_dir or root),
            seq_to_msa_mapping_path=str(
                seq_to_msa_mapping or root / "empty_alignment_mapping.json"
            ),
            token_crop_size=TOKEN_CROP_SIZE,
            msa_crop_size=MSA_CROP_SIZE,
            mode="test",
            prediction_seeds=[0],
            dataset_name="onixbind",
            crop_ligand=False,
            only_sampled_chains=False,
            sampling_seed=0,
        )

    def build(self, json_path: Path, record_id: str, model_seed: int) -> dict[str, Any]:
        import numpy as np
        import torch

        # resolved because the vendored pipeline joins this onto its cache root
        json_path = Path(json_path).resolve()
        if json_path.suffix != ".json" or not json_path.is_file():
            raise FileNotFoundError(f"not an AF3 JSON file: {json_path}")
        row = self._pd.Series(
            {
                "ID": record_id,
                "id": str(json_path.with_suffix("")),
                "INPUT_CACHE_ID": str(json_path.with_suffix("")),
                "UNIPROT_ID": record_id,
            }
        )
        output = self._dataset.process(row, prediction_seed=int(model_seed))
        if output.cropping_indices is not None:
            raise RuntimeError(
                f"{record_id}: the record was cropped; scoring it would score a "
                "different complex than the input describes"
            )
        features = output.input_features
        if int(np.asarray(features["is_ligand"]).sum()) < 1:
            raise ValueError(f"{record_id}: the record produced no ligand tokens")
        return {
            key: torch.tensor(np.expand_dims(value, axis=0))
            for key, value in features.items()
        }


def to_model_inputs(features: Mapping[str, Any], device, record_id: str) -> dict[str, Any]:
    """Apply the reference runtime's one-hot expansions and move to the device."""
    import torch
    import torch.nn.functional as F

    batch = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in features.items()
    }
    for key, classes in ONE_HOT_CLASSES.items():
        if key not in batch:
            raise KeyError(f"{record_id}: feature {key} is missing")
        batch[key] = F.one_hot(batch[key].long(), num_classes=classes).float()
    batch["file_id"] = record_id
    return batch
