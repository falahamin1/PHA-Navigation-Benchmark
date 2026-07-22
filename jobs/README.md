# Nav-Benchmark Slurm jobs (Alpine)

## Step 5b-cal calibration (current)

Two standalone long-running jobs, not an array — this is calibration (depth:
many iterations, frequent eval checkpoints on one encoder), not the sweep
(breadth: many short-ish runs across encoders/tiers/seeds).

- `nav-calibrate-easy-hrep-e05.slurm` — H-Rep, EASY, entropy_coef=0.05, up to
  5000 iterations, eval every 100 iterations (dual-mode: greedy + stochastic,
  10 resets/instance on the full 203-instance test set), model checkpoint
  every 250 iterations.
- `nav-calibrate-easy-hrep-e01.slurm` — identical, entropy_coef=0.01.

Both write to `calibration_results/` (relative to `WORK_DIR` in the script):
`<run_tag>_curve.json` (the growing three-curve data: iteration, entropy,
greedy_solve, stochastic_solve, elapsed_s) and `<run_tag>_checkpoint.pt`
(resumable model/optimizer/RNG state).

### Before submitting

1. **Edit `WORK_DIR`** at the top of each script to wherever this repo
   actually lives on Alpine (currently a placeholder:
   `/projects/amfa5003/nav-benchmark/Nav-Benchmark/envs`).
2. **Run `acompile` first.** `sbatch` inherits `MODULEPATH` from the shell
   you call it from — on the plain login node `module load anaconda` isn't
   visible there, so the job dies in under a second. This bit Rush-Hour's
   array jobs too; see `Rush-hour-git/jobs/submit_arrays.sh`'s header.
3. Confirm the `amfa-custom-env` conda environment on Alpine has this
   benchmark's dependencies (`gymnasium`, `torch`, `scipy` for
   `Voronoi`/`ConvexHull`) — it should already, since Tangram/Rush-Hour jobs
   use the same env, but this benchmark is new.
4. **Pool provenance — confirm the job trains on the same pool as your
   laptop, not a fresh regeneration that happens to differ.** `calibrate_cluster.py`
   now logs this at startup (`[pool] tier=easy n_train=... n_test=...
   train_fp=... test_fp=...`) and stores it in `<run_tag>_curve.json`'s
   `pool_provenance` field; on resume it raises immediately if the fingerprint
   doesn't match what the existing curve file recorded, rather than silently
   continuing on a different instance set. The reference values, computed
   locally (`pool_fingerprint` in `fairness_harness.py`, SHA-256-based, NOT
   Python's salted `hash()` — stable across machines/processes):

   ```
   EASY train: 812 instances, fingerprint 5273f45be21869b2
   EASY test:  203 instances, fingerprint 601ae1a38f60f29f
   ```

   **Safest option:** copy `envs/pool_cache/easy_pool.pkl` from your laptop
   to the same relative path on Alpine, so the job *loads* the pool instead
   of regenerating it. Regeneration should be deterministic (EASY is pure
   int/numpy arithmetic, no scipy geometry), but if you skip the copy, at
   least check the job's first `[pool]` log line against the values above
   before trusting any results.

### Submit

```bash
sbatch jobs/nav-calibrate-easy-hrep-e05.slurm
sbatch jobs/nav-calibrate-easy-hrep-e01.slurm
```

Monitor with `squeue -u $USER`. If either times out before reaching its
iteration ceiling, just resubmit the same script — it resumes from the last
saved checkpoint automatically (same convention as
`tangram-git/jobs/hrep.slurm`).

### If the job is crawling: eval frequency is the knob to turn

At the defaults (`--eval-every 100 --n-eval-resets 10`, 203 test instances,
dual-mode), each checkpoint is ~4,060 eval episodes (2,030 x2 modes); over
5000 iterations that's ~50 checkpoints ≈ 200k eval episodes total —
comparable to or larger than the training cost itself. Not worth
pre-optimizing before seeing real cluster timing, but if the first job's log
shows eval dominating wall-clock, drop `--eval-every` to 250 (fewer,
still-frequent-enough checkpoints) and raise `--n-eval-resets` later once
the interesting region (where entropy starts moving) is known, rather than
paying full resolution everywhere from iteration 1.

### Getting results back

Send back both `calibration_results/easy_hrep_entropy*_seed0_curve.json`
files — that's all that's needed to build the three-curve convergence plot
and the entropy comparison.

## Step 5b sweep (next, after calibration gates clean)

The 180-run array (6 encoders x 3 tiers x 10 seeds) is built and tested
locally (`envs/launch_sweep.py`, `envs/fairness_harness.py`,
`envs/test_fairness_harness.py`) but not yet turned into `.slurm` files here
-- that's the next step once the calibrated budget/entropy are frozen from
this calibration's results.
