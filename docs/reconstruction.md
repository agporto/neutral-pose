# Reconstruction provenance of the 1.6.x line

This document is retained verbatim from release 1.6.2. Version 1.7.0 repackaged
the same code (see `CHANGELOG.md` and `validation/refactor_audit_1.7.0.json`).


Version 1.6.2 updates the retained v1.6.1 implementation after reproducing a
Rhineura T1-T2 failure. The original `thinruts.zip` was available for this
validation. The correction separates merged contact masks, resolves joint
assignments, and retries failures after neighboring contacts are recovered.
No uploaded mesh or landmark was edited. This update leaves the original
fitting and dense-penetration objective and acceptance limits unchanged.

The history below describes reconstruction of the preceding v1.6.1 release.

The original v1.6.0 source archive could not be recovered. This v1.6.1 release
was reconstructed from retained v1.5.1 source, documented corrections, and the
saved v1.6.0 Ouroborus outputs. It is not claimed to be the original source or
numerically identical to it. The retained v1.5.1 source matches its saved hashes.

The saved Ouroborus result archive SHA-256 is:

`c7565bd472714f9dde221cfe43812f47401abd169d154c42c8085affe45fb561`

The uploaded Ouroborus archive was also unavailable. For validation, all 21
loaded/cleaned meshes and 588 landmarks were recovered with saved inverse
rigid transforms applied to their double-precision outputs. Every recovered
mesh coordinate lies within 1e-8 mm of the original float32 coordinate lattice.
Rounding to that lattice recovers the coordinates loaded by the old PLY reader;
the maximum forward roundtrip difference is approximately 4.4e-11 mm. Raw input
bytes, discarded degenerate cells and unmodified colors cannot be recovered.
Colors do not enter the fitting objective.

A fresh v1.5.1 run reproduced the critical baseline: 16 of 20 sagittal reference
fallbacks, a single unclassified C4-C5 contact, and 14 meshes with unreliable
surface signs. Fresh full automatic runs validate the reconstructed code on
this recovered Ouroborus input and retained original Draco and Bipes C3-C5
inputs. Results and limitations are in `validation/release_checks.json`.

New runs recompute discovery, landmark planes, fitting, sensitivity, selection,
collision checks and exports. Saved v1.6.0 poses and patches are comparison
evidence only; the algorithm does not depend on them. No saved transforms are
substituted for new fits. Search seeds and recovered patches may differ from
v1.6.0. An additional VTP writer correction avoids an observed compressed
binary-offset failure. Dependency versions and source hashes are recorded.
