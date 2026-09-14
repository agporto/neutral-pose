# Changelog

## 1.7.0 (2026-09-14) — packaging and code-quality release

No change to the fitted model. Every function of the 1.6.2 fitting, discovery,
symmetry, spacing, selection and export code is AST-identical except where
listed under "Refactoring" below; see `validation/refactor_audit_1.7.0.json`.
All 142 tests pass unchanged in behaviour.

### Packaging
- Installable package `neutral-pose` with a `src/` layout (`pyproject.toml`,
  setuptools backend, Python 3.10–3.13).
- Console commands `neutral-pose-auto` (automatic) and `neutral-pose` (manual);
  `python -m neutral_pose` aliases the automatic command. Both accept `--version`;
  the automatic command accepts `--log-level`.
- Single version source in `neutral_pose/version.py`.
- CI workflow: lint, format check, tests on Linux/macOS/Windows × 3.10–3.13,
  build and `twine check`.

### Performance
- Independent per-joint work — contact discovery, multistart joint fitting
  with sensitivity analysis, and pose selection — can run in worker
  processes (`--workers N|auto`, default `auto`; `workers=` in
  `auto.run`, `fit_column`, `select_supported_poses`, `prepare_specimen`).
  Every worker executes exactly the serial code for its own joint, so
  results do not depend on the worker count. Verified bitwise identical
  to 1.6.2 on six synthetic specimens under serial, `fork` and `spawn`
  execution (transforms, meshes, patches, landmarks, metrics, reports).
- Contact discovery runs serially whenever any bone lacks an accepted
  landmark symmetry plane: in that case discovery of joint *i* seeds the
  surface-symmetry plane that joint *i+1* reuses, so the pairs are not
  independent. This dependency existed in 1.6.2 and is preserved exactly;
  it is now documented and gated. Fitting and selection do not depend on it.
- `Mesh` objects are picklable; VTK distance and polydata objects are
  rebuilt from the stored geometry on unpickle with identical query results.
- The serial arithmetic is untouched: signed-distance queries and
  closest-point matching (the dominant costs) still use the same VTK and
  SciPy calls in the same order.

### Refactoring
- Flat modules became a package: `core`, `landmarks`, `anatomy`, `surfaces`,
  `discovery`, `contacts`, `recovery`, `support`, `auto`.
- Circular imports removed. Surface helpers moved from `neutral_pose_auto` to
  `surfaces`; `infer_joint` moved to `discovery`; `recover_joint` and
  `complete_contacts` moved to `recovery`. The bodies of these functions are
  unchanged apart from removing function-local imports and referencing
  `contacts.*` through the module namespace.
- `save_result`: an unused loop index removed.
- Inline thresholds in the fitting objective, coverage QC, contact discovery
  and role identity are now named module constants with the same values
  (`core.COVERAGE_*`, `core.DENSE_PENETRATION_*`, `discovery.*`,
  `contacts.ROLE_IDENTITY_*`, `contacts.PARTITION_*`).
- `fit_column`, `select_supported_poses` and `prepare_specimen` loop bodies
  were extracted verbatim into `_fit_joint_task`, `_select_joint_task` and
  `_discover_joint_task`; the chaining of poses happens in the caller.
- Whole codebase formatted with `ruff format` (line length 110); lint clean
  under `ruff check`. Unused imports removed. Every public function and class
  has a docstring. Loggers use module names.

### Migration from 1.6.2
- `python neutral_pose_auto.py X` → `neutral-pose-auto X`.
- `python neutral_pose.py …` → `neutral-pose …`.
- `import neutral_pose as n` → `from neutral_pose import core as n` (the
  high-level API is also exported from `neutral_pose` directly).
- `neutral_pose_auto.infer_joint` → `neutral_pose.discovery.infer_joint`;
  `neutral_pose_contacts.complete_contacts` → `neutral_pose.recovery.complete_contacts`.
- Tests no longer need the repository root on `sys.path`.
- Tests replaced deprecated VTK calls (`vtkTransformPolyDataFilter`,
  `GetPolys().GetData()`) so the suite runs warning-free on VTK 9.7.

## 1.6.2 — merged contact regions and dependent recovery

- Partition merged opposing regions with specimen-learned landmark seeds and geodesic surface distance.
- Match anatomical roles jointly with distinct surfaces; retain distance and ambiguity limits.
- Use the original arrangement and a finer fallback search within the existing angle window.
- Retry deferred joints after neighboring contacts provide supported frames; keep the original training model and spacing statistics fixed.
- Permit recovery trials while unrelated bones still lack a complete contact reference.
- Record partition boundaries, landmark match distances, and recovery passes.
- Retain fitting equations, penetration tolerances, coverage QC, and input geometry.
- Add regressions and fresh Rhineura/Ouroborus validation; see validation/release_checks.json.

## 1.6.1 — reconstructed Ouroborus correction

- Learn missing contacts from specimen-local landmarks, then rediscover and validate opposing surfaces.
- Reject incomplete references without reliable recovery.
- Use joint-level median/MAD evidence for isolated high facet spacing.
- Select by named contacts and penetration; keep uncertainty as a review flag.
- Constrain dense penetration while retaining the original residual objective.
- Export double PLY and VTP; avoid compressed VTP offset failures.
- Add new regressions and fresh complete specimen runs; see RECONSTRUCTION.md.

## 1.5.1 (2026-09-12) — exact reuse and conservative collision bounds

- Reuse byte-identical joint residuals and nonadjacent collision terms within
  each whole-column solve, using bounded caches with no pose rounding.
- Precompute local box corners and batch the existing box-overlap comparisons.
  Preserve every residual slot and finite-difference dependency.
- Avoid signed-distance queries for points provably outside a checked closed
  surface's expanded bounding box, where the penetration penalty is zero.
  Preserve the original VTK query for potentially interior points and
  unverified geometry. Keep the general signed-distance routine unchanged.
- Reuse the identical inverse transform within each joint residual call.
- Share an isolated internal configuration snapshot across joints and
  sensitivity trials, materializing independent mutable public copies only
  when accessed. Avoid repeated copies of the full column's patch lists.
- Add equivalence regressions for constrained and unconstrained solves,
  smoothness, active nonadjacent collisions, cache eviction, boundary points,
  concave meshes, scales and uncertified geometry.
- Report progress during refinement, dense validation and pose selection.
- Keep the mathematical objective, landmark method, spacing assumptions,
  sampling, starts, tolerances, sensitivity checks and output conventions.

## 1.5.0 (2026-09-12) — landmark-first anatomical planes

- Discover bilateral landmark reflection correspondences directly from point
  geometry, without mesh-plane or contact-patch seeds and without a fixed
  landmark-count schema. Require correspondence agreement across bones.
- Fit robust reflection planes using paired Cauchy weights and the exact
  weighted eigenvector update. Check fit residuals, identifiability and
  leave-one-pair-out sensitivity. Use complete pairs from partial files.
- Check whole-mesh reflection error without refining accepted landmark planes.
  Record missing or unreliable landmarks and use the existing checked surface
  fallback when necessary. Record original labels and inferred pair identities.
- Preserve landmark labels and row order in all output files. Normalize
  uniform exporter numbering only for internal correspondence matching.
- Validate unequal mirrored contact footprints directly when their centroids
  differ, while retaining opposite-side, normal and area checks. Reject
  contacts that span the midline as bilateral facets.
- Add known-plane, scale/orientation, row-order, outlier, missing-point,
  ambiguity, mesh-disagreement, cache-reset and unequal-footprint regressions.
- Preserve the one-command interface, inferred spacing, rigid transforms and
  constrained neutral-pose fitting.

## 1.4.1 (2026-09-12) — bilateral contact selection

- Correct the automatic assignment that could treat an extra near-midline
  arch contact and one lateral contact as the two facets, leaving the opposite
  contact unclassified. This reproduced the Bipes C3 plane-search-boundary error.
- Infer whole-bone symmetry from multiple intrinsic shape directions and
  unlabelled contact-pair directions before assigning bilateral contacts.
- Require the selected contacts to lie on opposite sides with compatible
  reflected positions, normals and areas on both adjacent bones.
- Retain the local search bounds and symmetry-quality checks. Near-optimal
  unresolved alternative planes contribute to ambiguity checks instead of
  being discarded in favor of a worse converged solution.
- Add regressions for extra central contacts, reordered candidates, a missing
  opposite contact and a shape with multiple equally valid planes.
- Include both mesh names in automatic contact-discovery errors.

## 1.4.0 (2026-09-12) — anatomical neutral constraints

- Correct the automatic neutral definition: remove lateral bending, axial
  twist and lateral displacement while fitting sagittal rotation and spacing.
- Infer bilateral planes from paired facets and robust whole-surface symmetry
  fits; record independent-sample residuals, conditioning and seed agreement.
- Constrain pairwise fits, global refinement and sensitivity trials to proper
  rigid transforms with coincident inferred midsagittal planes.
- Replace whole-input-pose fallback with a constrained sagittal reference whose
  in-plane translations are refitted. Lateral bending cannot reappear.
- Pool signed left/right facet spacing estimates equally before clamping.
- Label dorsal/lateral previews from anatomical axes and export numerical plane
  alignment and lateral-offset checks.
- Add known-pose recovery, scale, world-orientation, whole-column and fallback
  regressions; fail clearly when automatic anatomy cannot support the constraint.

## 1.3.0 (2026-09-12) — automatic specimen workflow

- Add `neutral_pose_auto.py`: a specimen folder, parent folder or ZIP is the only
  required input. Discover mesh/landmark pairs, filename order, unit metadata,
  coordinate convention, opposing contact patches and observed clearances.
- Infer candidate centrum/facet identities from contact geometry, with landmark
  support recorded and no hard-coded meaning for numbered labels.
- Estimate patch-specific spacing from input surface separation and record its
  dispersion. Estimate a per-joint error allowance from quadratic surface-fit
  residuals; retain the distinction from independent cartilage/noise measurements.
- Freeze projected inferred anchors during automatic patch-boundary sensitivity.
- Retain imperfect inputs in automatic mode with explicit review flags and
  unverified signed-distance constraints disabled. Strict manual behavior remains.
- Add automatic setup/inference receipts and a geometric before/after preview.
- Retain original local articulations when automatic fitting is unstable,
  ambiguous, or worsens penetration/coverage; record rejected proposals and
  recheck the selected column. A retained input pose is not certified neutral.
- Exercise the workflow on the supplied 22-vertebra Draco specimen; this is a
  real-data workflow check, not validation against known biological neutral poses.
- Add tests for coordinates, spacing, ZIP processing, preserved landmarks,
  self-intersection reporting, discovery and overwrite protection.

## 1.2.0 (2026-09-12) — robustness corrections

- Correct self-intersection candidate bounds: bounding-box centers with
  half-diagonal radii, conservative rounding padding, and bounded candidate
  batches. Includes the elongated-triangle counterexample and comparisons
  against exhaustive VTK triangle tests.
- Preserve disconnected components by default. Opt-in `keep_largest_component`
  selects greatest physical surface area instead of cell count. Record component
  areas, selections, removed area/fraction and output component count. Removal
  or retained disconnected geometry requires review.
- `noise_floor_mm: null` means unspecified; explicit values, including zero,
  override automatic estimation. `auto_noise_floor` now defaults to false.
  The optional edge heuristic reports its provenance and always requires review.
- Median edge length uses all unique edges. Sphere-fit noise allowances are
  normalized by mesh scale rather than joint scale.
- Adjacent crossings with unverified signed-distance signs always require review.
- Regression coverage for all three review findings and configuration/CLI
  precedence. Noisy recovery tests use a declared 0.02 mm fixture allowance;
  recovery results are regenerated for this version.
- Clarify cleanup effects, shallow-contact checks, migration and synthetic-only
  validation. Retain 1.1's anatomical anchors and centering sensitivity.

Migration: component retention and automatic noise defaults changed. Explicit
`noise_floor_mm: 0` now disables noise-based widening even when auto is enabled;
use `null` to request the optional heuristic. Legacy small-component report
keys remain aliases for the new removal counts.

## 1.1.0 (2026-09-11) — segmentation-oriented synthetic validation

Fixes for findings in the review of 1.0.0:

- Centering acts on anatomical anchors (landmark seeds projected onto the
  surface), with `centering_tolerance_fraction` (0.03) and
  `centering_inner_weight` (0.4). Fixes a 5-degree yaw error caused by
  asymmetric patch growth on planar facets that passed every 1.0 check.
- Noise floor per joint (`noise_floor_mm`, `auto_noise_floor`,
  `noise_edge_factor`): widens acceptance, Huber, penetration (2x) and
  ambiguity (1.5x) thresholds; the fitting dead zone is unchanged.
- `Mesh.read`/`Mesh.from_polydata`: relative-tolerance point merging, degenerate
  cell removal, sliver collapse, largest-shell retention
  (`keep_largest_component`), cleaning statistics in the report.
- Patch growth: one-ring smoothed normals on fine meshes and a neighbourhood
  cone axis; sphere acceptance loosened with noise.
- Self-intersection preflight vectorized (about 30x faster) and capped by
  `preflight_triangles`; skipping is a review reason.
- New review reasons: `pose_depends_on_patch_centering`,
  `anchor_offset_exceeds_tolerance`, `self_intersection_preflight_skipped`;
  `triangle_intersections_require_review` now only for crossings deeper than
  the penetration tolerance or on open meshes (`contact_within_noise_tolerance`
  otherwise).
- Report additions: `effective_tolerances`, per-patch `anchor_source`,
  `spherical_centrum`, `anchor_tangential_offset_mm`, `acceptance_gap_tolerance`,
  `mesh_quality[].median_edge_mm`, `.cleaning`, `.self_intersection_preflight`.
- Outputs: `patches/*_centers.npy` (original coordinates) alongside face ids.
- CLI: `--noise-floor-mm`. Warning when numeric ordering ignores a mesh.
- Modern `vtkCellArray` API (no deprecated `GetData`/`SetCells`).
- Tests: `tests/test_realistic.py` (noisy marching-cubes ball-and-socket pair
  and chain, wrong-start recovery, yaw regression, preflight scaling, vertex
  merging, island removal). Validation script extended with noisy cases and an
  explicit note that the wedge rows confirm a fixed point, not recovery.

Behavioral notes: cleaning may renumber vertices, so exported face ids refer to
the cleaned mesh. In 1.1, results on coarse hand-made meshes needed
`auto_noise_floor: false`; explicit values did not reliably override the
automatic floor. Version 1.2 corrects that precedence.

## 1.0.0

Initial anatomical articulation fitting replacing the straightening script.
