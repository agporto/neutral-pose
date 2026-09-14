# neutral-pose

Estimate an **osteological neutral articulation** of a vertebral column from
segmented bone meshes and landmarks.

Given ordered vertebrae exported from 3D Slicer (or any mesh tool) that still sit
in an approximate articulation, `neutral-pose` finds each bone's midsagittal plane,
discovers the centrum and zygapophyseal contacts between neighbours, estimates
joint spacing from the input geometry, and fits **proper rigid transforms only**
that remove lateral bending and axial twist while allowing sagittal curvature.
Bones are never deformed or rescaled. Every inference is recorded, and results
that cannot be supported are flagged `needs_review` rather than silently accepted.

> Results are osteological references fitted from bone surfaces. They are not
> measurements of cartilage thickness or of physiological resting posture.

## Install

Python 3.10 or newer.

```bash
pip install neutral-pose
```

From a checkout:

```bash
pip install -e ".[dev]"     # editable install with test and lint tools
```

Dependencies: numpy, scipy, VTK ≥ 9.2 and matplotlib. VTK wheels are large
(~150 MB) but install without a compiler on Linux, macOS and Windows.

## Quick start

```bash
neutral-pose-auto path/to/specimen.zip
```

That is the whole automatic workflow. No configuration, role map, patch radius or
spacing entry is required. A folder of meshes, or a parent folder containing
several specimens, works the same way. `python -m neutral_pose` is an alias.

Results are written beside the input in a folder ending in `_neutral`
(`specimen_neutral/` for the example above). Useful flags:

| Flag | Effect |
| --- | --- |
| `--out PATH` | Write results somewhere else. |
| `--overwrite` | Replace an existing result. The previous result is only removed after the new one is written; inputs are never modified. |
| `--workers N` | Worker processes for independent per-joint work (default `auto` = all CPUs). Results are identical for any value; `--workers 1` runs serially. |
| `--log-level DEBUG` | More console detail. |

Open `articulation_preview.png` first. For Slicer/VTK load `column_neutral.vtp`
or the individual files in `meshes_vtp/`; these keep double-precision coordinates
even at large world offsets. When the status is `needs_review`, read the reasons
in `neutral_report.json` before using the result.

### Input contract

- One mesh per bone (`.ply`, `.vtp`, `.stl` or `.obj`), in millimetres.
- Filenames sort into column order under natural numeric sorting
  (`C1.ply, C2.ply, …, C10.ply`).
- Landmarks (`.fcsv` or `.mrk.json`) share the mesh basename and live in `LMKs/`,
  `LMKs_json/`, or beside the meshes. Landmark *labels are identifiers only*;
  no anatomical role is ever inferred from a label number.
- Meshes must retain an approximate articulation. The method does not assemble
  arbitrarily scattered bones.
- Analysis tables, existing `*_neutral` result folders and macOS metadata inside
  a ZIP are ignored; no reorganisation is needed.

### Outputs

| File | Contents |
| --- | --- |
| `column_neutral.vtp`, `meshes_vtp/` | Posed meshes, double precision (recommended for Slicer/VTK). |
| `column_neutral.ply`, `neutral_*.ply` | Double-precision PLY counterparts. Some readers downcast to float32. |
| `LMKs_json/` | Transformed landmarks with original labels and row order. |
| `neutral_transforms.npy`, `transforms.json` | Rigid 4×4 matrices mapping input world coordinates to the neutral pose, plus the coordinate convention. |
| `patches/` | Triangle ids and anchors of every contact patch in cleaned-mesh coordinates. |
| `neutral_report.json`, `joint_metrics.csv` | Quality, tolerances, contacts, sensitivity and pose-selection diagnostics. |
| `automatic_setup.json`, `automatic_inference.json` | Every generated setting and the evidence behind it, including spacing replacements and contact recovery. |
| `articulation_preview.png` | Dorsal and lateral projections on shared scales. Orange/blue mark contact patches, not collisions. |

## How it works

The full method is described in [docs/method.md](docs/method.md). In brief:

1. Read and clean meshes; resolve RAS vs LPS by testing which convention places
   the landmarks on the surfaces.
2. Infer bilateral landmark correspondences by specimen-wide agreement and fit
   robust reflection planes; fall back to a checked surface-symmetry estimate.
3. Discover connected opposing surfaces between neighbours and classify one
   centrum contact plus a bilaterally paired facet contact.
4. Where discovery is incomplete, learn contact landmark groups from complete
   joints and use them to re-seed, partition merged regions, and retry; defer
   joints until neighbouring contacts provide a supported frame.
5. Estimate spacing from signed surface gaps with robust outlier handling.
6. Fit sagittal rotation and in-plane translation per joint with a Huber
   least-squares objective (surface gap, coverage, normal alignment, anchor
   centring, penetration), then refine the whole column with a smoothness prior
   and nonadjacent collision terms.
7. Compare the fitted pose against the input reference by convergence, named
   contact support, coverage and dense penetration; refine under a hard
   penetration ceiling; report everything.

## Manual, configuration-driven workflow

For specimens with hand-declared anatomical roles, the `neutral-pose` command
accepts a JSON configuration (see [examples/manual_config.json](examples/manual_config.json)
and [docs/method.md](docs/method.md#manual-configuration)).

```bash
neutral-pose specimens_root --config my_config.json --mesh-coordinates RAS
```

## Python API

```python
from neutral_pose import auto

results = auto.run("specimen.zip", workers="auto")   # same as the CLI
print(results[0]["status"], results[0]["output"])
```

The lower-level pieces (`Mesh`, `Options`, `fit_column`, `save_result`) are
exported from `neutral_pose`; discovery, recovery and selection live in the
`neutral_pose.discovery`, `neutral_pose.recovery` and `neutral_pose.auto` modules.

## Validation

```bash
python -m pytest -n auto
```

142 tests cover synthetic geometry with known curvature, landmark symmetry,
contact ambiguity, named-contact loss, spacing outliers, dense feasibility,
rigid transforms, merged-contact partitions, recovery dependencies and export
precision. The `validation/` directory records the actual release runs on real
specimens, with dependency versions and source hashes; see
[validation/README.md](validation/README.md). Geometric validation does not
establish biological resting-pose ground truth.

## Performance

Contact discovery, joint fitting and pose selection are evaluated per joint
and run in parallel worker processes by default. Each worker executes exactly
the code the serial loop would for its own joint, so results are the same for
any `--workers` value; only wall-clock time changes. Discovery falls back to
serial execution when a bone's symmetry plane had to be estimated from the
surface rather than from landmarks, because in that case neighbouring joints
share the estimated plane (the log says so). The whole-column refinement,
export and preview are serial.

## Limitations

- Inputs must retain an approximate articulation.
- Bone surfaces do not determine cartilage thickness or resting posture; spacing
  is a modelling assumption supported by the input geometry.
- Self-intersections, open or non-manifold surfaces and disconnected components
  are retained and reported, never removed to make a result pass. Unverified
  surface signs cannot certify absence of penetration.
- Automatic contact identities are provisional. Inspect `needs_review` results.

## Project history

Version 1.7.0 is a packaging and code-quality release with no change to the
fitted model; see [CHANGELOG.md](CHANGELOG.md). The provenance of the
reconstructed 1.6.x line is documented in [docs/reconstruction.md](docs/reconstruction.md).

## License

BSD-2
