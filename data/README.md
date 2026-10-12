# Data

Raw downloads go in `raw/<source>/` (`scripts/download_data.sh`). Prepared dry
audio goes in `dry/`, rendered pairs in `dataset/`. Only this file,
`splits.json` and `manifest.csv` are tracked in git.

| Source | Role | URL | Licence |
|--------|------|-----|---------|
| vocalset | pool | https://zenodo.org/records/1442513 | CC BY 4.0 |
| idmt_guitar | pool | https://zenodo.org/records/7544110 | CC BY-NC-ND 4.0 |
| guitarset | pool | https://zenodo.org/record/3371780 | MIT (to confirm) |
| synth | pool | own MIDI renders | own |
| own | heldout | own recordings | own |
| vctk | heldout | https://datashare.ed.ac.uk/handle/10283/3443 | CC BY 4.0 |

IDMT-SMT-Guitar is NC-ND: use it for training and evaluation only, never in
demo outputs (`demo_ok` is false in `dataset/params.csv`).

GuitarSet: MIT per the mirdata loader. To confirm on Zenodo.

## Steps

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
scripts/download_data.sh                        # plus VCTK, synth and own by hand
.venv/bin/python python/inspect_raw.py <source> # agree on group_id parsing first
.venv/bin/python python/prepare_dry.py
.venv/bin/python python/qc.py                   # listen to data/qc/
.venv/bin/python python/make_dataset.py
.venv/bin/pytest                                # includes checks on the real data
```
