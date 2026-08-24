# Installation

## Requirements

- Linux, x86-64
- NVIDIA GPU with compute capability 9.0 (H100/H200) or 8.0 (A100)
- CUDA 12.1 runtime
- Python 3.11 — the vendored `alphafold3/cpp` extension is built for this
  version specifically and will not import under another one
- ~600 MiB for the weight file, ~500 MiB for the chemical component dictionary

## Environment

```bash
conda env create --file src/environment.yaml
conda activate onixbind
```

`environment.yaml` is a pinned export of the environment these weights were
validated in: PyTorch 2.1.2+cu121 on Python 3.11. Newer PyTorch has not been
tested against them.

## Required environment variables

Two of these select which kernel runs, so they change the numbers rather than
just the speed. `primitives.py` reads the attention flag **at import time**, so
exporting it after the process starts has no effect.

```bash
export CUTLASS_PATH=/path/to/cutlass
export LAYERNORM_TYPE=fast_layernorm
export USE_DEEPSPEED_EVO_ATTENTION=true
export TORCH_EXTENSIONS_DIR="$HOME/.cache/onixbind/torch_extensions"
```

`src/predict.sh` sets all four for you. If you invoke `run_onixbind.py`
directly, set them yourself first.

## CUTLASS

Two CUDA kernels are compiled on first use and need CUTLASS headers:

```bash
git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git /path/to/cutlass
export CUTLASS_PATH=/path/to/cutlass
```

The first record of a run pays roughly three minutes for that build; later
records and later runs reuse it. Point `TORCH_EXTENSIONS_DIR` somewhere
writable and persistent to keep the cache across runs:

```bash
export TORCH_EXTENSIONS_DIR="$HOME/.cache/onixbind/torch_extensions"
```

## Large files

Two files are distributed separately from the source tree:

| file | size | where it goes |
|---|---|---|
| [`onixbind.pt`](https://drive.google.com/file/d/1g7xs8-Xe-L30iecCbzKxz5WQ3xAKMLYR/view?usp=sharing) | 518 MiB | `src/weights/` |
| [`ccd.pickle`](https://drive.google.com/file/d/10XBnUEdiw_AVeXlLqihRvnO7Z9HLPVgD/view?usp=sharing) | 462 MiB | `runtime/alphafold3/constants/converters/` |

Both are on [google drive](https://drive.google.com/drive/folders/1GcfLyZXnz4labCwyNKJnPwV_HICTTskB?usp=sharing). Put each in the directory above before
running.

The weight file's name does not matter. With no `--weights`, the single `.pt`
in `src/weights/` is used; the package is then checked against the configured
ensemble by the `members` field it carries, so a wrong or renamed file is
caught by its contents rather than by what it is called. Pass `--weights` with
a file or a directory to override. If `src/weights/` holds more than one `.pt`,
the run stops and asks which one. Keep the OnixBind-Flash checkpoint in
`src/onixbind-flash/` rather than `src/weights/`; `run_onixbind.py` only loads
the full-model ensemble package.

A missing weight file is reported immediately with a pointer to this page; a
missing `ccd.pickle` surfaces as a `FileNotFoundError` naming the path, from
inside the vendored pipeline.

## C++ runtime

The vendored `cpp` extension and the compiled CUDA kernels are built against a
newer C++ ABI than some hosts ship. `run_onixbind.py` scans the Conda
environment and the system library paths, and preloads whichever
`libstdc++.so.6` advertises the newest `GLIBCXX` version, before torch loads
anything. If that choice is wrong for your host, override it:

```bash
export ONIXBIND_LIBSTDCXX=/path/to/a/newer/libstdc++.so.6
```

An explicit override that cannot be loaded is an error rather than a silent
fallback, so a run never proceeds against a runtime other than the one asked
for.

## Check the install

```bash
python -m unittest discover -s tests -v     # CPU only, no GPU or weights needed
cd src && bash predict.sh                    # the bundled example, needs a GPU
```

## Usage

`src/predict.sh` takes an input path and an output directory, both optional:

```bash
bash predict.sh                                    # the bundled example
bash predict.sh /path/to/af3_inputs                # your own records
bash predict.sh /path/to/af3_inputs /path/to/out   # and your own output directory
```

Or call the entry point directly, after exporting the environment variables
above:

```bash
python run_onixbind.py /path/to/af3_inputs --out_dir /path/to/output --skip_completed
```

Add `--save_features` to dump the shared trunk representations (`s`, `z`) for OnixBind-Flash. Each record is written to `<out_dir>/features/<record_id>/features_<record_id>.pt`. Point `inference_onixbind_flash.ipynb`'s `trunk_file_dir` at that `features/` directory (or copy the `.pt` files into `src/onixbind-flash/feats/`).

The input path may be a single JSON file or a directory of them; add
`--recursive` to descend into subdirectories. Each record is written as soon as
it finishes, so a killed run keeps its completed work and `--skip_completed`
resumes without recomputing it. Seeds are read from each record's own
`modelSeeds`; `--seed 42` or `--seed 42,43` overrides that for every record.

### Multiple GPUs

One process per device, with records split by index so that none is scored
twice:

```bash
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$i python run_onixbind.py /path/to/af3_inputs \
    --out_dir /path/to/output --device cuda:0 \
    --shard_index "$i" --shard_count 6 --skip_completed &
done
wait
```

Resume reads only the worker's own output file, so resume with the same
`--shard_index` and `--shard_count` the run started with.

### Output

`predictions_rank-<i>.csv` holds one row per record and seed:

```text
record_id,seed,affinity,status
```

Each record additionally gets
`predictions/<record_id>/<record_id>_seed-<seed>_affinity.json`, containing the
ensemble value, both individual head scores, and the pocket fallback code
(0 = the 8 A pocket was found, 1 = widened to 12 A, 2 = no contact, fell back to
the 64 nearest residues).

### Inference settings

`--recycling_iters` (10), `--sampling_steps` (200), `--msa_depth` (2048) and
`--num_diffusion_samples` (1) default to the values these weights were deployed
with. The pocket cutoff (8 A), the affinity distogram range (2-22 A in 64 bins),
the token and MSA crop sizes and the equal-weight two-head reduction are fixed;
changing them invalidates the weights. A record that would need cropping is
refused rather than silently scored.
