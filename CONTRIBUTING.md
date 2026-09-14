# Contributing

## Set up

```bash
git clone <repository>
cd neutral-pose
python -m venv .venv && source .venv/bin/activate    # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff format src tests scripts validation/verify_exports.py
ruff check  src tests scripts validation/verify_exports.py
python -m pytest -n auto
```

CI runs the same three commands on Linux, macOS and Windows for every supported
Python version.

## What kind of change is it?

This code produces scientific results that people compare across versions, so
please be explicit about which of these a change is:

- **Behaviour-preserving** (refactor, docs, packaging). Say so in the PR and, if
  you touched fitting/discovery/selection code, include a function-level AST
  comparison (`scripts/` has an example of how the 1.7.0 audit was produced).
- **Model change** (any edit to the objective, thresholds, discovery policy,
  spacing statistics, selection rules or export precision). These need:
  - a regression test that fails before and passes after;
  - an entry in `CHANGELOG.md` describing what changes for users;
  - a fresh run on at least one real specimen with the receipt added to
    `validation/`, following the existing `fresh_run_receipt.json` pattern.

Named constants in `core.py`, `discovery.py` and `contacts.py` are model
parameters, not tuning knobs; changing one is a model change.

Anything run through `parallel_map` must be a pure function of its shared
inputs and index. If a task needs to write state on a shared object (as
contact discovery does with cached symmetry planes), it is not independent
and must run serially; `tests/test_parallel.py` checks serial and parallel
outputs are bitwise equal.

## Layout

```
src/neutral_pose/
  core.py        Mesh, Options, Joint, fitting, column refinement, export, manual CLI
  landmarks.py   bilateral landmark correspondence and reflection-plane fitting
  anatomy.py     symmetry planes, bilateral contact choice, neutral frames
  surfaces.py    closest-point queries and patch descriptors
  discovery.py   automatic contact discovery between adjacent bones
  contacts.py    specimen-local contact model learning and role identity
  recovery.py    recovery of incomplete joints
  support.py     contact support, pose-selection reasons, dense feasibility
  parallel.py    process-pool helper for independent per-joint tasks
  auto.py        automatic pipeline and CLI
tests/           pytest suite (fixtures are plain helper functions in test_neutral_pose.py / test_realistic.py)
validation/      release run records for real specimens (not distributed in the package)
scripts/         reproducibility scripts
docs/            method description and provenance
```

## Style

`ruff format` decides formatting; don't argue with it. Keep functions
commented at the level of *why*, not *what*. Report strings that end up in
JSON outputs are part of the interface: change them deliberately.
