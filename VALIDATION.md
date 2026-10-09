# Release preparation validation

Locally verified on macOS ARM64 with Python 3.12.14 and the versions in `requirements-reproduction.txt`, 2026-10-10:

- 20 scientific and engineering tests passed for the published experiment suite.
- `python demo.py` completed; the coherent-source maximum-similarity identity had zero error, and weighted duplicate collapse error was 2.22e-16.
- `python run_fidelity.py` completed from scratch in a separate clean export: 20 calibration and 40 confirmation seeds, no reused case caches. Chinese report and six figures were generated.
- All 10 result CSVs matched the original exactly; runtime measurements were excluded from the resource table comparison.
- All 7920 raw arrays in 60 case NPZ files matched the original exactly.
- Frozen threshold values, summary, audit and mechanism diagnostic matched exactly. Creation times/run durations differ as expected.
- Public text was scanned for the original local user path, email address and private runtime paths; none was included.

The full confirmation result comparison establishes local repeatability under the same dependency environment. [GitHub Actions](https://github.com/keymen6303-lgtm/memory-replay-lab/actions/runs/37976039027) also passed both Linux and macOS jobs, installing dependencies and running the 20 checks and demo. Full confirmation experiments were not run in CI, and independent external replication remains to be done.

Version 0.1.1 changes project presentation and report bylines only. Scientific model files, protocols, seeds and numerical reference data are unchanged from 0.1.0.
