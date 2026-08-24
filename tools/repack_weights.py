#!/usr/bin/env python3
"""Build the OnixBind weight file from the four-head source package.

The source package stores one shared upstream state and four affinity heads.
OnixBind ships two of them, so this rewrites the selected heads under the
``affinity_heads.<i>.`` prefix the model expects and drops everything else.
The source file is only read.

``--members`` names the heads to pull out of the source package; ``--aliases``
names them in the output. They are kept separate so the released package does
not have to carry the source's internal labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch



def _digest(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].contiguous()
        h.update(key.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="source four-head weight package (.pt)")
    parser.add_argument("--out", required=True, help="output .pt path")
    parser.add_argument("--members", nargs="+", required=True,
                        help="head names to take from the source package")
    parser.add_argument("--aliases", nargs="+", default=None,
                        help="what to call them in the output; defaults to "
                             "head_0, head_1, ...")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    aliases = args.aliases or [f"head_{i}" for i in range(len(args.members))]
    if len(aliases) != len(args.members):
        raise SystemExit(
            f"{len(args.members)} member(s) but {len(aliases)} alias(es); "
            "they name the same heads and must line up"
        )
    package = torch.load(source, map_location="cpu")
    for alias in args.members:
        if alias not in package["heads"]:
            raise SystemExit(f"head {alias!r} not in {sorted(package['heads'])}")
        arch = package["head_architectures"][alias]
        if arch not in ("AffinityModulePocket", "AffinityModulePocket_v1_pool"):
            raise SystemExit(
                f"head {alias!r} is {arch}; OnixBind only implements the "
                "AffinityModulePocket v1_pool head"
            )

    state = dict(package["shared_upstream"])
    for index, alias in enumerate(args.members):
        for key, value in package["heads"][alias].items():
            state[f"affinity_heads.{index}.{key}"] = value

    weight = 1.0 / len(args.members)
    # the id follows the file it is written to, so the manifest and the package
    # never disagree about what to call themselves
    package_id = out.stem
    payload = {
        "format_version": 1,
        "package_id": package_id,
        "members": list(aliases),
        "weights": [weight] * len(args.members),
        "state_dict": state,
    }
    payload["logical_state_sha256"] = _digest(state)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    manifest = {k: v for k, v in payload.items() if k != "state_dict"}
    manifest.update(
        {
            "tensors": len(state),
            "parameters": sum(v.numel() for v in state.values()),
            "size_bytes": out.stat().st_size,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }
    )
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
