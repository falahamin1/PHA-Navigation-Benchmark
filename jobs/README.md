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

## B0e -- protocol-matched chance baseline (evaluation only, no training)

Diagnosis follow-up, not part of the sweep proper. Answers: is a given
(tier, encoder)'s trained solve rate actually above chance, on the SAME
protocol (same fixed 100-instance subsample, same 10 resets, same eval
code path) the plateau values and Stage A's interval-protocol curves were
measured on -- rather than against an untrained baseline measured under a
different, unmatched protocol (see the B0e task spec's own account of why
the first attempt at this, at 3 resets under no fixed protocol, was
reopened).

**Code**: `envs/b0e_array_task.py` (one array task per (tier, encoder,
seed) -- reuses `launch_reduced_sweep.index_to_run`'s exact bijection, so
`--array=0-29` per tier means the same thing it already does for
`nav-sweep-*.slurm`). Each task computes, under the interval protocol
(`np.random.default_rng(999)`'s fixed 100-instance subsample, 10 resets,
both eval modes, reusing `fairness_harness._evaluate` verbatim -- no new
eval logic):

- an untrained baseline (`torch.manual_seed(seed)`, fresh encoder weights
  -- the same seed already used to train that cell, not a new enumeration)
- a re-evaluation of that cell's existing final checkpoint
  (`sweep_results/<tier>_<encoder>_seed<seed>_checkpoint.pt` -- must
  already exist from the Stage A/B sweep; this job trains nothing)

Horizon is capped at 15 for EASY/MEDIUM (empirically lossless -- verified
locally before submission and again per-task via each output's
`self_check` field) and left at the frozen 40 for HARD (no cap below 40
was lossless for HARD's untrained baseline). The SAME horizon is used for
the baseline and the checkpoint within a tier, since a mismatched horizon
between the two sides would reintroduce exactly the kind of protocol
seam this job exists to eliminate.

Writes `sweep_results_b0e/<tier>_<encoder>_seed<seed>_b0e.json` -- a NEW
directory; `sweep_results/*.json` is read-only to this job.

### Submit

```bash
sbatch jobs/b0e-easy.slurm
sbatch jobs/b0e-medium.slurm
sbatch jobs/b0e-hard.slurm
```

Same preconditions as every job above: run `acompile` first (this bites
`sbatch` specifically, not just interactive `module load` -- see the top
of this file), edit `WORK_DIR` if needed, confirm the checkpoints these
tasks read already exist on Alpine (copy `sweep_results/*_checkpoint.pt`
and `sweep_results/<tag>.json` over if they don't -- this job cannot
produce them, only read them).

`--time` on all three is a rough estimate extrapolated from local
(non-cluster, single-core) timing, not a cluster-measured number --
EASY/MEDIUM at 1h, HARD at 4h (HARD's untrained-baseline eval is
substantially slower even after the horizon-cap analysis; see
`jobs/b0e-hard.slurm`'s header). Watch the first array's actual wall time
via `sacct` and tighten if it's overprovisioned.

### Getting results back and aggregating

Send back `sweep_results_b0e/*_b0e.json` (all 90). Then, from `envs/`:

```bash
python3 b0e_aggregate.py
```

This prints: the per-seed baseline table, a consistency check against the
6 checkpoints Stage A's A1 already re-evaluated under this exact protocol
(flags any that don't match), the matched comparison table (18 tier x
encoder cells against their own baseline, both modes, points and
baseline-std-units difference, 95%-CI-overlap flag), and the three
specific questions the B0e task asks: whether CNN clears its own chance
rate on HARD, whether chance differs meaningfully by encoder, and whether
the interval-protocol ranking agrees with the full-protocol ranking
already on record. `envs/test_b0e_aggregate.py` covers this script's
math against hand-constructed fixtures; `envs/test_b0e_array_task.py`
smoke-tests the per-task driver end-to-end against a real checkpoint.

## B3 -- convergence calibration (current)

Resumes 6 existing Stage A/B checkpoints (H-Rep, Relational, CNN x HARD,
EASY, all seed0 -- see `jobs/b3-extend.slurm`'s header for why seed0-
everywhere and why EASY is in scope, not just HARD) from iteration 3000 to
a 15,000-iteration ceiling. Rewrites the eval protocol to be seamless: the
full held-out test set at 30 resets at EVERY checkpoint, not the cheap-
sample-then-full-set switch the original sweep and B0e both used -- A1 and
B0e both showed that switch changes a headline ranking (MEDIUM's #1
encoder flips depending which side of it you read), so Stage D can't
inherit it. Also logs B0a's argmax-invariance measure
(`envs/argmax_invariance.py`) every checkpoint, not just as a one-shot
diagnostic.

**Code**: `envs/b3_extend_training.py` (one array task per (tier, encoder,
seed0) cell, `--array-index` into its own `TARGET_RUNS` list -- there is no
30-slot grid here, only 6 explicit runs). Reads
`sweep_results/<tag>_checkpoint.pt` and `sweep_results/<tag>.json`
read-only (never writes there); writes
`sweep_results_b3/<tag>_b3_checkpoint.pt`, `_b3_progress.json`, and (on
reaching 15,000) `_b3.json`. Idempotent and resumable, same contract as
`sweep_cluster.py` -- safe to resubmit.

### Submit

```bash
sbatch jobs/b3-extend.slurm
```

Same preconditions as every job above (`acompile` first, `WORK_DIR`,
confirm the 6 target checkpoints already exist locally on Alpine -- this
job cannot produce them, only extend them).

**Budget is genuinely uncertain and likely tight** -- see
`jobs/b3-extend.slurm`'s header for the full math (local, non-Alpine
timing extrapolates HARD training alone to ~67h before eval, plus
~14-18h of full-protocol eval at `--eval-every 1000`). Expect HARD tasks
especially to need at least one resubmission; that's an accepted outcome
of this design, not a failure -- just resubmit the same script.

### Getting results back

Send back `sweep_results_b3/*_b3.json` and `*_b3_progress.json` (progress
files matter here even for incomplete runs -- the report this stage
produces is a per-encoder curve, not just an endpoint, so a task that's
still mid-run at iteration 9000 when you need to check in is still useful
data, unlike the original sweep's "don't interpret partial results"
convention).

## Step 5b full sweep (later, if the reduced sweep isn't enough)

The 180-run array (6 encoders x 3 tiers x 10 seeds) is built and tested
locally (`envs/launch_sweep.py`, `envs/fairness_harness.py`,
`envs/test_fairness_harness.py`) but not yet turned into `.slurm` files here
-- this is the larger design the reduced sweep above was cut down from, kept
around in case 5 seeds/encoder turns out not to be enough statistical power
for the paper.
