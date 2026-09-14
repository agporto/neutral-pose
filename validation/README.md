# Validation record

## v1.7.0

Packaging release; the model is unchanged. `refactor_audit_1.7.0.json` lists
every function whose AST differs from 1.6.2 and the exact diff.
`test_execution_1.7.0.json` records the test run of the packaged code.
`PROVENANCE.json` (repository root) records the bitwise equivalence runs
against 1.6.2, including parallel execution; `scripts/make_synthetic_specimen.py`
and `scripts/compare_results.py` reproduce them.
The real-specimen results below were produced by 1.6.2 and were not rerun.

## v1.6.2

All 142 tests passed. Fresh automatic CLI runs completed for the full original
Rhineura upload (121 meshes, 3388 landmarks) and recovered Ouroborus (21 meshes,
588 landmarks). Every exported mesh vertex, triangle set, landmark label/order,
and rigid transform was audited. Double PLY and VTP coordinates agree exactly.
Ouroborus matrices are bitwise identical to the validated v1.6.1 result.

## What failed in Rhineura

The old code reproduced the exact T1-T2 error. On T2, its original connected
candidate combined centrum and one facet; in some trial alignments it combined
all three roles. Pairing that region with the centrum left no distinct region
for the facets. Landmark files and coordinate matching were available.

The update partitions merged opposing masks using specimen-learned landmark
seeds, assigns anatomical roles jointly, and retries deferred joints after
neighboring contacts provide supported frames. The original training model and
spacing statistics stay fixed. A finer fallback search stays within the
existing 12-degree angular window. Penetration limits and coverage QC remain.

Eight incomplete joints were recovered: 6, 7, 11, 105, 107, 108, 109 and 111
(one-based joint indices). All 112 originally complete boundaries/anchors were
preserved. The original meshes and landmarks were never edited.

## Final T1-T2 contact coverage

| Contact | T1 to T2 | T2 to T1 |
| --- | ---: | ---: |
| centrum / inferred_centrum | 86.5% | 83.3% |
| facet / inferred_facet_1 | 96.9% | 96.9% |
| facet / inferred_facet_2 | 90.6% | 71.9% |

The full result remains `needs_review` and is supplied as a diagnostic output,
not an accepted neutral pose for downstream analyses. The full-column preview
still shows substantial sagittal coiling. Joints with penetration flags:
[33, 62, 67, 68]. Joints with a contact below the 10%
presence threshold: [14, 22, 27, 34, 36, 42, 52, 57, 59, 60, 63, 64, 66, 73, 82, 85, 86, 93, 96, 100, 101, 102, 110, 112, 115, 116, 120]. Read
`rhineura/summary.json` for all reasons, including coverage, fit and uncertainty.
Passing the software/export checks does not certify every anatomical fit.

`release_checks.json` contains the runtime and measured checks. Exact commands,
timings and source hashes are in each case's `fresh_run_receipt.json`. The
fitting-algorithm AST audit documents unchanged functions; candidate discovery
and its recovery policy intentionally changed. Generated log paths refer to
the validation machine; reproduce with your own input path.

```bash
neutral-pose-auto "thinruts.zip"
python validation/verify_exports.py "extracted_input/New" "thinruts_neutral"
```
