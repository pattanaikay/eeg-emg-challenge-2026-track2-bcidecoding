# Source provenance

The selected official competition source under `vendor/competition/` is copied without code edits from:

- Repository: https://github.com/neural-interfaces26/2026-competition
- Revision: `a66bc23d1a0e686b895c093a642b45c74970758b`
- Retrieved: 20 October 2026
- Selection: `benchmark_utils/`, `tracks/bci_decoding/`, root `README.md`, root `requirements.txt`

Original comments and notices are retained. The pinned repository did not include a root license file in the inspected checkout. This notice records origin and does not grant or replace upstream rights. Review upstream terms before redistributing its code publicly. Data use is separately governed by the dataset license (the public PROTEUS release declares CC BY 4.0) and competition rules.

The project-owned workflow code adds data auditing, explicit validation selection, artifact recording, smoke tests and packaging around those sources. `submission.py` implements the reference linear design and a Braindecode EEGNet wrapper. It is kept self-contained for the worker's one-Python-file inference contract.
