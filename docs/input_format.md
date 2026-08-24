# Input format

OnixBind reads AlphaFold 3 input JSON. One file describes one complex.

```json
{
  "dialect": "alphafold3",
  "version": 1,
  "name": "5S8I_A",
  "modelSeeds": [42],
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "SMSYDIQAWKKQ...",
        "unpairedMsa": ">query\nSMSYDIQAWKKQ...\n>hit1\nSMSYEIQAWKKQ...\n",
        "pairedMsa": "",
        "templates": []
      }
    },
    { "ligand": { "id": ["B"], "ccdCodes": ["2LY"] } }
  ]
}
```

- **`modelSeeds`** — the record carries its own seeds and one prediction is
  written per seed. `--seed 42` on the command line overrides them for every
  record.
- **protein `sequence`** — one entry per chain. A multi-chain complex lists
  several protein entries, each with its own `id` and its own MSA.
- **`unpairedMsa`** — A3M text inline. `unpairedMsaPath` takes a file path
  instead, and `pairedMsa` / `pairedMsaPath` are the paired equivalents for
  multi-chain complexes. Paths must be local; object-storage URLs are rejected. An MSA is required: these weights were
  fit with real alignments and a query-only input is not equivalent. Deeper
  alignments are cropped to 4096 rows, of which the model samples 2048 per
  recycle with the query row pinned first.
- **ligand** — either `ccdCodes` for a chemical component dictionary entry, or
  `smiles` for an arbitrary molecule:

```json
{ "ligand": { "id": ["B"], "smiles": "C#C[C@H](NCCc1cocn1)c1nn(C)c2ccccc12" } }
```

A record whose token count would exceed the crop size is refused rather than
scored, because a cropped complex is not the complex the file describes.

See `src/examples/5S8I_A.json` for a complete working record.
