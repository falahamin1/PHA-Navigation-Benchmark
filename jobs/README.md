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
   actually lives on Alpine.
2. **Run `acompile` first.** `sbatch` inherits `MODULEPATH` from the shell
   you call it from — on the plain login node `module load anaconda` isn't
   visible there, so the job dies in under a second. This bit Rush-Hour's
   array jobs too; see `Rush-hour-git/jobs/submit_arrays.sh`'s header.
3. **Partition/QoS naming and the 24h cap.** Both scripts use
   `--partition=acpu`/`--qos=cpu-normal` (Alpine's renamed `amilan`/`normal`,
   already accepted ahead of the Aug 5 2026 cutover). `cpu-normal` caps
   runtime at 24h — `--time=24:00:00` is the max allowed, not a real estimate
   of total need for 5000 iterations. If a job hits the wall before finishing,
   that's expected: just resubmit (see below) and it resumes from checkpoint,
   spanning as many 24h submissions as it takes.
4. Confirm the `amfa-custom-env` conda environment on Alpine has this
   benchmark's dependencies (`gymnasium`, `torch`, `scipy` for
   `Voronoi`/`ConvexHull`) — it should already, since Tangram/Rush-Hour jobs
   use the same env, but this benchmark is new.
5. **Pool provenance — confirm the job trains on the same pool as your
   laptop, not a fresh regeneration that happens to differ.** `calibrate_cluster.py`
   now logs this at startup (`[pool] tier=easy n_train=... n_test=...
   train_fp=... test_fp=...`) and stores it in `<run_tag>_curve.json`'s
   `pool_provenance` field; on resume it raises immediately if the fingerprint
   doesn't match what the existing curve file recorded, rather than silently
   continuing on a different instance set. The reference values, computed
   locally (`pool_fingerprint` in `fairness_harness.py`, SHA-256-based, NOT
   Python's salted `hash()` — stable across machines/processes):

   ```
   EASY   train: 812 instances, fingerprint 5273f45be21869b2
   EASY   test:  203 instances, fingerprint 601ae1a38f60f29f
   MEDIUM train: 708 instances, fingerprint 46cc988a095f337b
   MEDIUM test:  175 instances, fingerprint 1c494eb02a001f06
   HARD   train: 694 instances, fingerprint 764cc4977f8b6665
   HARD   test:  168 instances, fingerprint 9516aee04fea91a8
   ```

   MEDIUM/HARD were computed the same way as EASY (`get_final_pool(tier)` +
   `pool_fingerprint(...)`, both in `fairness_harness.py`) and are pinned in
   code as `fairness_harness.REFERENCE_POOL_FINGERPRINTS`. The reduced-sweep
   jobs (below) call `verify_pool_fingerprint(tier, ...)` automatically at
   startup and abort before training on any mismatch -- unlike the
   calibration jobs above, which only print the fingerprint for a human to
   compare by eye.

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

## Step 5b reduced sweep (current -- AAAI submission)

Calibration's conclusion: **entropy_coef=0.01**, uniform across every
encoder/tier. At 0.05 the policy stayed pinned near maximum entropy
(~1.9-2.0) for its entire 2600-iteration run and never sharpened; at 0.01
entropy decayed and greedy/stochastic solve rates climbed to 68.5%/72.0% by
iteration 4900 (see `calibration_results/easy_hrep_entropy*_seed0_curve.
json`). 0.05 is dead; do not re-open it.

This is the real run for the paper, not another calibration -- **90 runs**
(6 encoders x 3 tiers x **5 seeds**, not the 180-run/10-seed design
`envs/launch_sweep.py` was built for; that scaffold is untouched and still
the eventual full sweep once bandwidth allows). Fixed for every run:
entropy_coef=0.01, 3000-iteration budget (iterations, not wall-clock --
different encoders do different iterations/hour, so a wall-clock-fixed
budget would be an unfair comparison), both eval modes logged every
checkpoint.

**Code**: `envs/launch_reduced_sweep.py` (array_index -> (encoder, seed)
within a tier, 30 tasks/tier -- `python envs/launch_reduced_sweep.py` prints
the full 90-row mapping) and `envs/sweep_cluster.py` (the actual per-task
entry point `jobs/nav-sweep-*.slurm` call). `sweep_cluster.py`:

- Verifies the tier's pool fingerprint against
  `fairness_harness.REFERENCE_POOL_FINGERPRINTS` before any training and
  raises immediately on mismatch (the fingerprints recorded above).
- Checkpoints (model/optimizer/RNG state) and a progress curve every 300
  iterations by default; resuming (rerun the identical command -- the
  Slurm scripts already do this on resubmit) continues from the last
  checkpoint's iteration/episode_count, same contract as
  `calibrate_cluster.py`. If the tagged result file already exists, the
  task is a no-op -- safe to resubmit an array after it's fully finished.
- Logs entropy + greedy + stochastic solve rate at every checkpoint (a
  sampled 100-instance subset for cheap intermediate checkpoints, the full
  held-out set for the last 2 checkpoints and always at iteration 3000) --
  the three-curve record, not just an aggregate.
- Writes `sweep_results/<tier>_<encoder>_seed<seed>.json`: the curve, full
  per-instance held-out outcomes for both eval modes (from the iteration-
  3000 full eval), the frozen config (including `commit_hash`, via `git
  rev-parse HEAD`), and pool provenance -- the schema the statistics module
  consumes.

### Submit

Three per-tier arrays, 30 tasks each (`--array=0-29`):

```bash
sbatch jobs/nav-sweep-easy.slurm
sbatch jobs/nav-sweep-medium.slurm
sbatch jobs/nav-sweep-hard.slurm
```

Same preconditions as the calibration jobs above: run `acompile` first,
edit `WORK_DIR` if needed, confirm `amfa-custom-env` has this benchmark's
deps. Monitor with `squeue -u $USER`. A task that hits the 24h wall before
reaching 3000 iterations is expected, not a failure -- resubmitting the same
array (`sbatch jobs/nav-sweep-<tier>.slurm`) resumes every still-incomplete
task from its checkpoint and is a no-op for any task that already finished.

### Known issue -- skipped-cell physics violations on MEDIUM/HARD

**First real run against MEDIUM/HARD surfaced a pre-existing physics gap**:
`nav_env.py`'s hard `assert` against skipped-cell transitions (see its "STEP
A -- dt recalibration" docstring) fires during training/eval on these
tiers -- 0% of EASY runs, ~23% of MEDIUM runs, ~90% of HARD runs hit it in
the first submission. Root cause (strong hypothesis, not numerically
proven): the prior dt=0.008 recalibration was validated against the
closed-loop oracle's smooth, near-optimal trajectories; an RL policy
(especially high-entropy/early-training) reverses direction far more
erratically, and the coupled 2D velocity dynamics (`A_v` has off-diagonal
terms) can transiently overshoot past either direction's steady-state speed
when the input flips -- large enough to skip a cell on MEDIUM/HARD's
smaller cells, essentially never on EASY's larger ones.

This is a frozen-layer (dt/drift_coupling) issue, so per the launch prompt
it is **not** fixed by touching `nav_env.py`/`integrator.py`. Fix adopted
(2026-07-26, confirmed with the person running the sweep): harness-level
catch. `fairness_harness._is_sim_violation` matches exactly this assertion's
message; `fairness_harness._evaluate` and `sweep_cluster.py`'s training
rollout both catch it, end that episode/decision-step with a new
`"sim_violation"` outcome (counted as not-solved, same bucket as
hazard/timeout/truncated), and continue -- a fresh episode for training, the
next reset for eval. Nothing in `nav_env.py`/`integrator.py` changed.

Counts are surfaced, not silently dropped: `sweep_results/*.json` now
carries `train_sim_violations` (cumulative over the whole run) and
`final_eval_sim_violations` (from the final full eval only), and the curve
carries both per-checkpoint. **A run with a nonzero count is not invalid**
-- those episodes already count as not-solved in the reported solve rates --
but a high count relative to total episodes is a real signal worth checking
before trusting a given (tier, encoder, seed)'s numbers, and worth a
footnote in the paper's methodology for MEDIUM/HARD specifically.

**Any array submitted before this fix landed must be resubmitted** --
`sweep_cluster.py`'s idempotent-completion check means already-finished
tasks (mostly EASY, which was never affected) are a no-op on resubmit, and
failed HARD/MEDIUM tasks simply restart from iteration 0 (they crashed
before ever reaching a checkpoint in the large majority of cases; any that
had reached one checkpoint before failing will resume from it as normal).

### Getting results back

Send back `sweep_results/*.json` (all 90, or however many have completed --
but per the launch prompt, do not interpret partial results mid-sweep; the
statistics module runs once all three arrays are fully done). Flag any file
with a high `train_sim_violations`/`final_eval_sim_violations` relative to
its training length or eval sample size when handing off to the statistics
module.

## Step 5b full sweep (later, if the reduced sweep isn't enough)

The 180-run array (6 encoders x 3 tiers x 10 seeds) is built and tested
locally (`envs/launch_sweep.py`, `envs/fairness_harness.py`,
`envs/test_fairness_harness.py`) but not yet turned into `.slurm` files here
-- this is the larger design the reduced sweep above was cut down from, kept
around in case 5 seeds/encoder turns out not to be enough statistical power
for the paper.
