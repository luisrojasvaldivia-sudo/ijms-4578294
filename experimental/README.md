# External molecular-validation gate (EXP-6)

No experimental datum is bundled or simulated in this directory. This is an
intentional safeguard: the revised manuscript does not relabel synthetic NRTL
outputs as molecular evidence.

To execute EXP-6, the authors must deposit a redistribution-permitted,
traceable dataset and complete `dataset_manifest.json` from the supplied
template. Every row must cite a resolvable DOI or database record and preserve
the reported measurement uncertainty. Suggested targets from the reviews are a
binary LLE system at two or more temperatures and a ternary type-I LLE system
with tie-line data. Access-controlled NIST TDE or Dortmund Data Bank records
must not be redistributed without the corresponding licence.

Required anti-leakage protocol:

1. Canonicalize species by InChIKey, not by display name.
2. Assign entire components, binary pairs, and ternary systems to exactly one
   of train, validation, or test; never split rows from one mixture across
   partitions when claiming chemical transfer.
3. Fit all hyperparameters on training/validation systems only.
4. Freeze the model before revealing test temperatures and compositions.
5. Compare against refitted NRTL/UNIQUAC and a molecular baseline on the same
   measurements, weights, and split.
6. Archive source identifiers, licence, row exclusions, unit conversions,
   uncertainty weights, split hashes, and falsifiers declared before fitting.

`experiments/run_external_validation.py` performs schema/provenance checks and
stops with an explicit error until real, cited rows and a frozen split are
provided. It is a protocol scaffold, not a claim that EXP-6 has been run.

