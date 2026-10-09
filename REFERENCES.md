# Attribution and prior work

This repository implements equations and experiments in original Python code; it does not redistribute another project's source or papers.

- [Takeda, Oizumi and Karakida, 2026 preprint, arXiv:2605.27975 v2](https://arxiv.org/html/2605.27975v2): basis for the closed-form energy/replay identities, drift geometry and energy-guided replay motivation. Stage 7 reproduces the specified closed-form setting, not all training or diffusion experiments in that paper.
- [Hopfield Networks is All You Need, official implementation](https://github.com/ml-jku/hopfield-layers): related established modern Hopfield memory work and software. Our small NumPy model is not a replacement for that library.
- [SQHN, Nature Communications 2024](https://www.nature.com/articles/s41467-024-46976-4) and [author code](https://github.com/nalonso2/SQHN): prior associative memory work including recall/recognition. Joint evaluation is not a newly invented task here.

More primary references, search rounds, established findings, proposed extensions and actual reading scope are recorded in the original dated snapshots under `outputs/literature/`. These snapshots describe candidate status at the time of planning; current outcomes are in `RESULTS.md` and `reference_results/`.

The incremental question here is how replay-source geometry and fixed-memory recognition readouts interact under the explicitly declared budget and synthetic task. A finite search cannot establish that nobody has studied a related question. The strongest present value is transparent controls and reproducibility, with negative primary results.
