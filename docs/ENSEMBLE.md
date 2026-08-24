# Which two heads, and why

The source package holds four affinity heads over one shared trunk: one from an
earlier training run and three epoch checkpoints — `e0`, `e2`, `e3` — from a
later one. OnixBind ships two of the three later checkpoints, equally weighted.

Restricting to those three keeps the release simple: they share one head
architecture, so the inference path is a single class evaluated twice. The
fourth head is a different class and would have doubled the model code for one
more member.

## e0 + e2

Scored on OpenBind (494 records, single target, zero leakage), Boltz-BM (1103
records, 4 targets) and a 8789-record validation set, as Pearson against the
experimental label:

| pair | OpenBind | Boltz-BM | held-out set |
|---|---:|---:|---:|
| **e0+e2** | **0.5188** | **0.5210** | 0.2694 |
| e0+e3 | 0.5033 | 0.5119 | 0.2908 |
| e2+e3 | 0.4954 | 0.5080 | 0.3007 |
| e0 alone | 0.5098 | 0.5129 | 0.2480 |

`e0+e2` is the strongest pair on the two structure-based benchmarks. On
the held-out set the ordering reverses and pairs containing `e3` do better: across
`e0 → e2 → e3` the score rises monotonically on the held-out set and falls monotonically on
OpenBind, which is what a later checkpoint fitting its training distribution
more tightly looks like. If your deployment target resembles that set more than it
resembles OpenBind, `e0+e3` is the better pair. The held-out set is an internal 8789-record
benchmark; the two named benchmarks are public.

Switching is a repack; the config does not change, because the two heads are
called `head_0` and `head_1` in the released package whichever pair is packed:

```bash
python tools/repack_weights.py --source <four-head package>.pt \
  --out src/weights/onixbind_e0_e3.pt \
  --members <e0's name in the source> <e3's name in the source>
```

`--members` names the heads inside the source package; `--aliases` names them in
the output and defaults to `head_0`, `head_1`. Keep only one `.pt` in
`src/weights/`, or pass `--weights` to say which one to run; a directory holding
two packages stops the run rather than guessing.

The four-head source package is not part of this distribution, so this path is
only open to whoever holds it, who is also the only one who needs its head names.

The runner refuses to start if the weight file's members disagree with the
configured ensemble, so the two cannot drift apart silently.

Throughout this page `e0`, `e2` and `e3` are the training checkpoints the heads
came from. In the released package they are simply `head_0` and `head_1`, in the
order given to `--members`; for this release that order is `e0` then `e2`.

## Reproduction

Measured on OpenBind against the reference implementation's own per-head
predictions, this release scores 0.4521 where the reference scores 0.5188 for
the same pair. Per head: `e0` reaches 0.4913 against the reference's 0.5098,
while `e2` reaches 0.3824 against 0.5014. Correlation between our per-head
output and the reference's is 0.931 for `e0` and 0.865 for `e2`.

In other words `e0` reproduces closely and `e2` does not, and the gap is not
yet explained. If reproducing the reference matters more to you than the
benchmark score, `e0` alone is currently the closest configuration.
