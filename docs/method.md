# Method

This page describes what `neutral-pose` computes and how to read its outputs.
The README covers installation and everyday use.

## Automatic method

1. Read meshes, normalize landmark coordinate conventions, and compare surface
   matches under RAS and LPS. Read declared mesh units; absent declarations use
   the recorded Slicer millimeter convention.
2. Infer bilateral landmark correspondence by specimen-wide agreement. Robust
   reflection fitting estimates each plane, checked against the surface.
   Missing or unreliable landmark planes use the checked surface method.
   Numbered labels never have hard-coded anatomical roles.
3. Find connected opposing surfaces. A centrum and two facets require geometric
   support. Facets must form a bilateral pair on both bones.
4. For an incomplete joint, learn landmark groups, pairing, normal directions
   and patch sizes from at least three complete joints with accepted landmark
   planes. Use these only to seed alignment and rediscover opposing surfaces.
   Require unique landmark identities and bilateral geometric agreement.
   If ordinary discovery fails, partition merged opposing regions by geodesic
   distance from the learned landmark seeds. Only the original proximity/normal
   mask is eligible. Assign the three roles jointly with distinct candidates,
   retaining the distance and ambiguity checks. Try the input arrangement and,
   if needed, a finer search within the existing 12-degree seed window.
   Defer failed joints while neighboring contacts are recovered, then retry;
   the original training model and spacing statistics remain fixed. Stop if a
   pass makes no progress. Complete joints retain their original boundaries
   and anchors; the recovery never relabels them.
5. Estimate signed surface gaps and pool the two facets equally. With at least
   five complete joints, isolated high facet gaps beyond a robust median/MAD
   bound use the specimen median. Each joint has one statistical vote.
   Consistently wide spacing is preserved; original observations stay in the
   report. Recovered contacts use complete-joint spacing medians.
6. Fit rotation about the lateral axis and two in-plane translations. The first
   bone stays fixed and inferred midsagittal planes coincide. Bones keep their
   shape and anatomical asymmetry.
7. Preserve multistart and sensitivity diagnostics. Compare fitted and
   projected-input reference poses by convergence, named contact support,
   coverage and dense penetration. Uncertainty alone does not restore the
   input bend. Lost contacts cannot be hidden by improved average coverage.
   A penetrating reference cannot win solely on coverage against a feasible
   fitted candidate supporting all contacts.
8. Where needed, refine the same residual objective under an absolute
   penetration ceiling. An active set accelerates constraints; every acceptance
   checks all vertices, triangle centers and independent area samples.
   Exclude unreliable surface signs and report them. Recheck nonadjacent
   contacts on the final selected column.

The 10% bidirectional support test detects absent contacts during selection;
the separate coverage QC threshold stays at 60%. Passing the first does not
certify satisfactory anatomy. A flagged sagittal reference may be retained
when a fit cannot be supported; consult the selected-pose report.

The fitting residual equations and v1.5.1 exact-reuse optimizations remain.
Contact inference, spacing assumptions, constraints and selection change the
automatic model. No math-equivalence or general speedup over v1.5.1 is claimed.
Manual configurations without the automatic contact policy retain legacy
selection behavior. The manual entry point remains the `neutral-pose` command.

## Outputs

- `column_neutral.vtp`: combined posed mesh, recommended for Slicer/VTK.
- `meshes_vtp/`: individual posed meshes with double coordinates.
- `column_neutral.ply` and `neutral_*.ply`: double-coordinate PLY counterparts.
  Some readers downcast doubles; use VTP to retain precision.
- `LMKs_json/`: transformed landmarks with original labels and row order.
- `neutral_transforms.npy` and `transforms.json`: rigid matrices and coordinate
  convention. Matrices map original input world coordinates.
- `patches/`: triangle IDs and input-coordinate centers in cleaned meshes.
- `neutral_report.json` and `joint_metrics.csv`: quality, tolerances, contacts,
  sensitivity and selected-pose diagnostics.
- `automatic_setup.json` and `automatic_inference.json`: generated settings and
  their evidence, including spacing replacements and contact recovery.
- `articulation_preview.png`: dorsal/lateral projections with shared scales.

Orange and blue mark patches, not collisions. The first bone defines preview
axes. The centrum reference line should be straight in the dorsal view;
sagittal curvature can remain in the lateral view.

## Interpretation and limits

Inputs must retain an approximate articulation. This method does not assemble
arbitrary scattered bones. Bone surfaces do not determine cartilage thickness
or physiological resting posture. Spacing is a modeling assumption supported
by the input geometry; results are osteological references.

Automatic identities remain provisional. Self-intersections, open/nonmanifold
surfaces and disconnected components are retained and reported, never removed
to make a result pass. Unverified signs cannot certify absence of penetration.
Inspect `needs_review` results before downstream analyses.

## Manual configuration

The `neutral-pose` command runs the same fitting with hand-declared anatomy.
A configuration is a JSON file merged over the built-in defaults; a
`neutral_config.json` inside a specimen folder is merged over that.

```json
{
  "mesh_coordinate_system": "RAS",
  "length_unit": "mm",
  "require_anatomy": true,
  "roles": {
    "anterior_centrum": "<landmark label>",
    "posterior_centrum": "<landmark label>",
    "left_prezygapophysis": "<landmark label>",
    "right_prezygapophysis": "<landmark label>",
    "left_postzygapophysis": "<landmark label>",
    "right_postzygapophysis": "<landmark label>"
  },
  "patch_defaults": {
    "centrum":   {"radius_fraction": 0.22, "normal_angle_deg": 80},
    "facet":     {"radius_fraction": 0.16, "normal_angle_deg": 55},
    "accessory": {"radius_fraction": 0.12, "normal_angle_deg": 65}
  },
  "joint_gaps": {"centrum": 0.01, "facet": 0.01, "accessory": 0.01},
  "options": {"starts": 5, "samples": 160, "target_samples": 1600, "sensitivity": true}
}
```

- `roles` map anatomical roles to the landmark labels that carry them. Patches
  are grown as geodesic discs around those landmarks (`radius_fraction` of the
  bone scale, limited by `normal_angle_deg` from the seed normal).
- `joint_gaps` are assumed clearances as fractions of the joint scale.
- `options` correspond to the fields of `neutral_pose.Options`; every field is
  validated. Command-line flags (`--starts`, `--samples`, `--seed`,
  `--noise-floor-mm`, `--no-sensitivity`) override the file.
- Per-joint overrides (`"joints": [{"a": "...", "b": "...", "patches": [...],
  "initial_transform": [...]}]`) are recorded in `automatic_setup.json` of
  automatic runs and can be copied into a manual configuration.

Without `require_anatomy`, joints lacking declared roles fall back to
geometry-only candidate discovery with legacy pose selection.
