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
"""Preload a C++ runtime new enough for the compiled attention kernel.

The DeepSpeed evo-attention extension is built by the container's compiler but
loaded into a process whose library path starts with the Conda environment.  If
that environment ships an older ``libstdc++.so.6`` the import fails with a
missing ``GLIBCXX_*`` symbol even though the build succeeded.  Loading a new
enough copy first, globally, fixes the resolution order.

Call :func:`preload_libstdcxx` before ``import torch``.  Once torch has loaded
a ``libstdc++.so.6`` the loader keeps that one for the soname and a later
preload does nothing, so ordering is the whole point.  When python is launched
from a shell, ``LD_PRELOAD`` is the more reliable lever.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
from pathlib import Path
from typing import Optional

ENV_VAR = "ONIXBIND_LIBSTDCXX"

_CANDIDATE_PATHS = (
    "native/lib/libstdc++.so.6",  # relative to the package parent, if vendored
)

_SYSTEM_PATHS = (
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    "/usr/lib64/libstdc++.so.6",
    "/usr/local/lib/libstdc++.so.6",
)


def _candidates() -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    found = [here / rel for rel in _CANDIDATE_PATHS]
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        found.append(Path(conda) / "lib" / "libstdc++.so.6")
    # the interpreter's own environment, even when CONDA_PREFIX is unset
    found.append(Path(sys.executable).resolve().parent.parent / "lib" / "libstdc++.so.6")
    found.extend(Path(p) for p in _SYSTEM_PATHS)
    seen, unique = set(), []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _abi_version(path: Path) -> tuple[int, ...]:
    """Highest GLIBCXX_3.4.N the library advertises, as a sortable tuple."""
    try:
        blob = path.read_bytes()
    except OSError:
        return ()
    best = ()
    for match in re.finditer(rb"GLIBCXX_(\d+)\.(\d+)\.(\d+)", blob):
        version = tuple(int(g) for g in match.groups())
        if version > best:
            best = version
    return best


def preload_libstdcxx(verbose: bool = False) -> Optional[Path]:
    """Load the newest available libstdc++ globally, before torch does.

    Picking the first match is not enough: a host's system copy is often older
    than the Conda environment's, and preloading the older one pins it for the
    process and breaks any extension built against a newer toolchain.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        # an explicit choice that silently falls back is worse than a crash:
        # the run would proceed against a different runtime than intended
        candidate = Path(override)
        if not candidate.is_file():
            raise FileNotFoundError(f"{ENV_VAR} does not point at a file: {override}")
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise FileNotFoundError(f"{ENV_VAR} is not loadable: {override} ({error})") from error
        if verbose:
            print(f"preloaded {candidate}")
        return candidate

    ranked = sorted(
        ((_abi_version(p), p) for p in _candidates() if p.is_file()),
        key=lambda item: item[0],
        reverse=True,
    )
    for version, candidate in ranked:
        if not version:
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if verbose:
            print(f"preloaded {candidate} (GLIBCXX up to {'.'.join(map(str, version))})")
        return candidate
    return None
