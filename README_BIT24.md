# Bit 24 — Two targeted 4D eigenpacket nonlinear interaction

**Scientific result:** The exact four-state XOR nonlinear difference is identically
zero before causal contact and for all free-streaming ticks, but turns nonzero
under the unchanged reversible Bit8 collision. At reference separation 16,
all eight seeds first differ at tick 4; at tick 14 the interaction affects on
average 6,721.4 of 12,288 physical D4 nodes. This is *not* a demonstration
of stable particle scattering or isolated outgoing eigenmodes.

- Full report: `docs/RESULTS_Bit24_TwoEigenpacketNonlinearity_de.md`
- Standalone preparation matrix: `data/bit24_collision_matrix.json`
- Main executable: `src/phase_bit_s4_open43_pair_collision.py`
- Full scans: `src/phase_bit_s4_open43_batch.py` and `src/phase_bit_s4_open43_branch_controls.py`
- Synthesis and figures: `src/phase_bit_s4_open43_analyze.py`, `src/phase_bit_s4_open43_plot.py`
- Independent checks: `src/phase_bit_s4_open43_validate.py`, `data/bit24_validation.json`
- Summary: `data/bit24_summary.json`
- Full per-seed time series: `data/bit24_*.json` (11 cases)
- Figures: `figures/bit24_*.svg` and PNG

Core experiment with provided archive:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open43_validate.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python src/phase_bit_s4_open43_pair_collision.py \
  --matrix data/bit24_collision_matrix.json --seeds 1 --ticks 6 \
  --shape 32 4 4 16 --out data/check.json
```

IMPORTANT: Initial eigenvectors and Gaussian thresholds use analysis-only
floating-point arithmetic and 16-bit *integer* threshold quantization.
Every subsequent physical state consists of exactly 24 Boolean bits per
physical D4 node and the pre-existing Bit8 collision rule.
