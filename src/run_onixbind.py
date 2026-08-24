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
"""OnixBind affinity inference.

Reads AlphaFold 3 input JSON, builds features with the vendored AF3 pipeline,
runs the shared trunk and diffusion path once per record, evaluates both
affinity heads on the same tensors, and writes the mean pX value.
"""

import argparse
import csv
import gc
import json
import logging
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# must happen before torch loads any compiled extension
from onixbind.native import preload_libstdcxx
preload_libstdcxx()

import torch
from tqdm import tqdm

from onixbind.features import FeatureProcessor, to_model_inputs
from onixbind.openfold.inference_config import get_model_config
from onixbind.openfold.model.model import OnixBind

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = ("record_id", "seed", "affinity", "status")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def _sole_package(candidates: list[Path], where: Path) -> Path:
    if not candidates:
        raise SystemExit(
            f"no weight package (*.pt) found in {where}\n"
            "It is distributed separately from the source tree; see docs/installation.md."
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise SystemExit(
            f"{len(candidates)} weight packages in {where} ({names}). "
            "Pass --weights to say which one."
        )
    return candidates[0]


def resolve_weights(explicit: str | None) -> Path:
    """Locate the weight package.

    The file name is not part of the contract: whatever it is called, the
    package is checked against the configured ensemble by its own `members`
    field before it is loaded. With no --weights, the single .pt sitting in
    src/weights/ is used.
    """
    if explicit is None:
        return _sole_package(sorted(WEIGHTS_DIR.glob("*.pt")), WEIGHTS_DIR)
    path = Path(explicit).expanduser()
    if path.is_dir():
        return _sole_package(sorted(path.glob("*.pt")), path)
    if not path.is_file():
        raise SystemExit(
            f"weight package not found: {path}\n"
            "It is distributed separately from the source tree; see docs/installation.md."
        )
    return path


def init_logging():
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)-4s] [%(filename)s:%(lineno)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def collect_inputs(data: Path, recursive: bool) -> list[Path]:
    if data.is_file():
        return [data]
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(data.glob(pattern))


def load_completed(predictions_csv: Path) -> set:
    if not predictions_csv.exists():
        return set()
    with predictions_csv.open(newline="") as handle:
        return {
            (row["record_id"], int(row["seed"]))
            for row in csv.DictReader(handle)
            if row.get("status") == "success"
        }


def append_prediction(predictions_csv: Path, row: dict) -> None:
    """Append one result. Written per record so a killed run keeps its work."""
    is_new = not predictions_csv.exists()
    with predictions_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n"
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def seeds_from_record(json_path: Path, override: str | None) -> list[int]:
    """AF3 records carry their own modelSeeds; only an explicit flag overrides."""
    if override:
        return [int(s) for s in override.split(",")]
    record = json.loads(json_path.read_text())
    seeds = record.get("modelSeeds")
    if not seeds:
        raise ValueError(f"{json_path.name}: no modelSeeds and no --seed given")
    return list(dict.fromkeys(int(s) for s in seeds))


def main(args) -> None:
    init_logging()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shard_count < 1:
        raise SystemExit("--shard_count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit(
            f"--shard_index must be in [0, {args.shard_count}); got {args.shard_index}. "
            "Every worker running the full set is the failure mode this catches."
        )

    weights = resolve_weights(args.weights)

    inputs = collect_inputs(Path(args.data).expanduser(), args.recursive)
    if not inputs:
        logger.warning("No AF3 JSON files found; exiting.")
        return
    stems = [p.stem for p in inputs]
    duplicates = {name for name in stems if stems.count(name) > 1}
    if duplicates:
        raise SystemExit(
            f"{len(duplicates)} input file name(s) collide once the directory is "
            f"flattened, e.g. {sorted(duplicates)[:3]}. Record ids come from the "
            "file name, so these would overwrite each other's output."
        )
    shard = inputs[args.shard_index :: args.shard_count] if args.shard_count > 1 else inputs
    logger.info(
        f"{len(inputs)} record(s); this worker takes {len(shard)} "
        f"(shard {args.shard_index}/{args.shard_count})"
    )

    predictions_csv = out_dir / f"predictions_rank-{args.shard_index}.csv"
    completed = load_completed(predictions_csv) if args.skip_completed else set()
    error_dir = out_dir / "errors"
    error_dir.mkdir(exist_ok=True)

    processor = FeatureProcessor(args.runtime_dir)
    config = get_model_config(argparse.Namespace(
        sampling_steps=args.sampling_steps, recycling_iters=args.recycling_iters))
    config.backbone.msa.msa_embedder.msa_depth = args.msa_depth

    if args.save_features:
        feat_output_dir = out_dir / "features"
        feat_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        feat_output_dir = None

    generator = torch.Generator(device=device)
    model = OnixBind(config, generator=generator).to(device).eval()
    package = torch.load(weights, map_location="cpu")
    if tuple(package["members"]) != tuple(config.ensemble["members"]):
        raise SystemExit(
            f"weight members {package['members']} do not match the configured "
            f"ensemble {list(config.ensemble['members'])}"
        )
    model.load_state_dict(package["state_dict"], strict=True)
    logger.info(f"Loaded {weights.name} ({package['members']}, {package['weights']})")
    logger.info(
        f"recycles={config.backbone.recycling_iters} "
        f"steps={config.sample.no_sample_steps_T} msa_depth={args.msa_depth} "
        f"diffusion_samples={args.num_diffusion_samples} "
        f"save_features={args.save_features}"
    )

    for json_path in tqdm(shard, desc="Predicting", disable=args.shard_index != 0):
        record_id = json_path.stem
        try:
            seeds = seeds_from_record(json_path, args.seed)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{record_id}: {exc}")
            error_dir.joinpath(f"{record_id}.txt").write_text(traceback.format_exc())
            continue

        for seed in seeds:
            if (record_id, seed) in completed:
                continue
            try:
                features = processor.build(json_path, record_id, seed)
                batch = to_model_inputs(features, device, record_id)
                batch["feat_output_dir"] = feat_output_dir
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                model.generator.manual_seed(seed)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                    outputs = model(batch, diffusion_batch_size=args.num_diffusion_samples)
                affinity = outputs["affinity_logits"].float().cpu().reshape(-1)
                heads = {
                    alias: value.float().cpu().reshape(-1).tolist()
                    for alias, value in outputs["head_affinity_logits"].items()
                }
                record_dir = out_dir / "predictions" / record_id
                record_dir.mkdir(parents=True, exist_ok=True)
                (record_dir / f"{record_id}_seed-{seed}_affinity.json").write_text(
                    json.dumps({
                        "record_id": record_id,
                        "seed": seed,
                        "affinity": affinity.tolist(),
                        "affinity_space": "pX",
                        "ensemble_members": list(model.head_aliases),
                        "ensemble_weights": list(model.head_weights),
                        "head_affinity": heads,
                        "pocket_fallback_code":
                            outputs["pocket_fallback_code"].cpu().reshape(-1).tolist(),
                    }, indent=1)
                )
                append_prediction(predictions_csv, {
                    "record_id": record_id, "seed": seed,
                    "affinity": f"{float(affinity.mean()):.6f}", "status": "success",
                })
            except Exception as exc:  # noqa: BLE001 - one bad record must not stop the batch
                logger.warning(f"{record_id} seed {seed}: {exc}")
                error_dir.joinpath(f"{record_id}_seed-{seed}.txt").write_text(
                    traceback.format_exc()
                )
                append_prediction(predictions_csv, {
                    "record_id": record_id, "seed": seed,
                    "affinity": "", "status": "failed",
                })
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    scored, failed = 0, 0
    with predictions_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "success":
                scored += 1
            else:
                failed += 1
    logger.info(f"{scored} scored, {failed} failed; predictions in {predictions_csv}")
    if failed:
        logger.warning(f"per-record tracebacks are in {error_dir}")
    if scored == 0:
        # a run that scored nothing must not look like a success to a pipeline
        raise SystemExit("no record was scored successfully")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OnixBind affinity inference")
    parser.add_argument("data", type=str,
                        help="AF3 JSON file, or a directory of them.")
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--weights", type=str, default=None,
                        help="Merged two-head weight package, or a directory "
                             "holding one. Defaults to the single .pt in src/weights/.")
    parser.add_argument("--runtime_dir", type=str, default=None,
                        help="Vendored AF3 pipeline; defaults to ../runtime.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--seed", type=str, default=None,
                        help="Override the record's modelSeeds, e.g. '42' or '42,43'.")
    parser.add_argument("--recycling_iters", type=int, default=10)
    parser.add_argument("--msa_depth", type=int, default=2048)
    parser.add_argument("--num_diffusion_samples", type=int, default=1)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Whether to save the trunk features (s, z) for OnixBind-Flash. Default is False.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
