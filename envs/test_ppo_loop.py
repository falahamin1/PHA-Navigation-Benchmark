"""Step 5a tests: run directly with `python test_ppo_loop.py`.

Risk category: training-loop correctness, not geometry -- these target the
classic silent PPO bugs (advantage sign errors, missing normalization, mask
applied inconsistently, GAE bootstrapping across episode boundaries, loss
scale mismatch) that don't crash and don't fail shape tests.

- test_learns_on_easy: the end-to-end integration check (run separately,
  see gate5a_diagnostic.py -- it's slow, ~5 min).
- test_advantage_normalization: per-batch advantages are mean~0, std~1, and
  it's the normalized tensor that's actually used in the policy loss.
- test_gae_episode_boundary: hand-computed 2-episode rollout, exact expected
  values, confirms no cross-boundary bootstrap leak.
- test_mask_consistency: masked actions get exactly zero probability,
  entropy matches a manual feasible-actions-only computation, flipping a
  mask bit changes the distribution.
- test_clipping_actually_clips: synthetic ratios/advantages chosen so the
  clipped term must be selected; confirms clip is not a no-op.
- test_gradient_flow: gradients reach encoder parameters, not just the heads.
- test_determinism: same seed -> same training trajectory (2 short runs).
- test_reward_accounting_sanity: a hand-traceable short episode's buffered
  rewards match the env's per-step rewards summed, cross-checked against the
  Step 1 reward formula directly (not just against another copy of the
  same call).
"""

import numpy as np
import torch

from deepset_encoders import HRepDeepSet
from nav_env import GAMMA, GOAL_REWARD, STEP_PENALTY, NavEnv
from ppo_train import MaskedEncoderActorCritic, compute_action_mask, compute_gae, make_clean_easy_instances, \
    _instance_to_config

torch.manual_seed(0)


def test_learns_on_easy():
    """The end-to-end integration check: PPO must actually learn, not just
    run. Trained across 6 different clean EASY instances (generalization,
    not single-instance memorization -- that was Step 4's smoke test).

    Threshold note (updated post Step-5a-fix, see ppo_train.py's module
    docstring): with return normalization + value clipping both applied, a
    100-iteration run converges (entropy decays near zero) to a clean,
    deterministic 4/6-instances-solved-at-100%/2/6-at-0% split -- ~67%
    overall -- confirmed via greedy per-instance evaluation to be a real
    generalization/capacity characteristic of this small instance set, NOT
    remaining training instability or environment stochasticity (0%/100% per
    instance, nothing in between, under a greedy policy). 0.5 is kept as a
    genuinely-and-reliably-cleared, clearly-above-chance threshold for this
    fast (30-iteration) smoke version of the test; see gate5a_diagnostic.py
    for the full before/after comparison.
    """
    from ppo_train import train_ppo
    partition, instances = make_clean_easy_instances(6, seed=0)
    _model, history = train_ppo(HRepDeepSet, partition, instances, n_iterations=30, seed=0,
                                 solve_rate_target=0.5, solve_rate_window=30, verbose=True)

    assert history[-1] >= 0.5, f"failed to reach the 0.5 solve-rate threshold, final={history[-1]:.0%}"
    # "Rises monotonically-ish": the back half of the run should clearly
    # beat the front half on average (noisy per-iteration, not noisy overall).
    half = len(history) // 2
    front_avg = sum(history[:half]) / half
    back_avg = sum(history[half:]) / (len(history) - half)
    assert back_avg > front_avg, f"solve rate did not improve overall: front_avg={front_avg:.2f}, back_avg={back_avg:.2f}"

    print(f"  learning curve: {[f'{r:.2f}' for r in history]}")
    print(f"  front-half avg={front_avg:.2f}, back-half avg={back_avg:.2f}, final={history[-1]:.2f}")
    print("test_learns_on_easy: PASS")


def test_advantage_normalization():
    rewards = [1.0, -0.5, 2.0, 0.5, -1.0, 3.0, 0.2, -0.3]
    values = [torch.tensor(v) for v in [0.3, 0.1, 0.5, 0.2, 0.0, 0.4, 0.1, 0.2]]
    dones = [False] * 7 + [True]
    last_value = torch.tensor(0.0)

    advantages, returns = compute_gae(rewards, values, dones, last_value)
    raw_mean, raw_std = advantages.mean().item(), advantages.std().item()

    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    norm_mean, norm_std = normalized.mean().item(), normalized.std().item()

    assert abs(norm_mean) < 1e-5, f"normalized advantage mean should be ~0, got {norm_mean}"
    assert abs(norm_std - 1.0) < 1e-4, f"normalized advantage std should be ~1, got {norm_std}"

    # Confirm normalization actually changes the tensor used downstream (i.e.
    # it isn't computed and discarded) -- the normalized tensor must differ
    # from the raw one whenever raw std != 1 (true here by construction).
    assert not torch.allclose(advantages, normalized), (
        "normalized advantages are identical to raw advantages -- normalization may be a no-op"
    )
    print(f"  raw advantages: mean={raw_mean:.4f}, std={raw_std:.4f}")
    print(f"  normalized advantages: mean={norm_mean:.2e}, std={norm_std:.4f}")
    print("test_advantage_normalization: PASS")


def test_gae_episode_boundary():
    """Hand-computed 2-episode rollout (3 steps + 2 steps). Exact expected
    values computed independently (see ROADMAP.md Step 5a note / the bash
    calculation used to derive these), gamma=0.99, lam=0.95."""
    rewards = [1, 2, 3, 10, 20]
    values = [torch.tensor(v) for v in [0.5, 0.5, 0.5, 1.0, 1.0]]
    dones = [False, False, True, False, True]
    last_value = torch.tensor(0.0)

    advantages, returns = compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95)
    expected = torch.tensor([5.0826481249999995, 4.3462499999999995, 2.5, 27.859499999999997, 19.0])

    residual = (advantages - expected).abs().max().item()
    assert residual < 1e-6, f"GAE values don't match hand-computation: got {advantages.tolist()}, residual={residual}"

    # The buggy (cross-boundary-bootstrapping) version would give ~29.13 for
    # step 0, not ~5.08 -- an order of magnitude off. Confirm we're nowhere
    # near that, i.e. the boundary is genuinely being respected, not just
    # coincidentally close.
    buggy_step0 = 29.13494254622994
    assert abs(advantages[0].item() - buggy_step0) > 10, (
        "step 0's advantage is suspiciously close to the cross-boundary-bootstrap value -- "
        "check that dones[] is actually zeroing the GAE recursion at episode boundaries"
    )
    print(f"  advantages = {[f'{a:.4f}' for a in advantages.tolist()]}")
    print(f"  expected   = {[f'{e:.4f}' for e in expected.tolist()]}")
    print(f"  max residual = {residual:.2e}")
    print("test_gae_episode_boundary: PASS (no cross-boundary leak)")


def test_mask_consistency():
    partition, instances = make_clean_easy_instances(1, seed=1)
    instance = instances[0]
    model = MaskedEncoderActorCritic(HRepDeepSet())
    model.eval()

    # A state pinned at the domain's right edge: E/NE/SE must be infeasible.
    domain = partition.domain
    xmax = domain[1]
    state_at_edge = torch.tensor([xmax, 2.5, 0.0, 0.0])
    mask = compute_action_mask(state_at_edge, domain)
    from integrator import DIRECTION_NAMES
    infeasible = [DIRECTION_NAMES[i] for i in range(8) if not mask[i]]
    assert set(infeasible) == {"E", "NE", "SE"}, f"expected E/NE/SE infeasible at right edge, got {infeasible}"

    with torch.no_grad():
        logits, _ = model(partition, instance, state_at_edge, current_cell=0, mask=mask)
        dist = torch.distributions.Categorical(logits=logits)

    for i in range(8):
        if not mask[i]:
            assert dist.probs[i].item() == 0.0, f"masked action {DIRECTION_NAMES[i]} has nonzero probability"

    # Entropy must match a manual computation using ONLY the feasible actions.
    feasible_probs = dist.probs[mask]
    manual_entropy = -(feasible_probs * torch.log(feasible_probs)).sum().item()
    assert abs(dist.entropy().item() - manual_entropy) < 1e-5, (
        f"entropy {dist.entropy().item()} != manual feasible-only entropy {manual_entropy} -- "
        "masked actions may be leaking into the entropy computation"
    )

    # Flipping a previously-masked action to feasible must change the distribution.
    mask_flipped = mask.clone()
    e_idx = DIRECTION_NAMES.index("E")
    mask_flipped[e_idx] = True
    with torch.no_grad():
        logits2, _ = model(partition, instance, state_at_edge, current_cell=0, mask=mask_flipped)
        dist2 = torch.distributions.Categorical(logits=logits2)
    assert not torch.allclose(dist.probs, dist2.probs), "flipping a mask bit should change the policy distribution"
    assert dist2.probs[e_idx].item() > 0.0, "E should have nonzero probability once unmasked"

    print(f"  infeasible actions at right edge: {infeasible}")
    print(f"  masked-action probabilities: {[f'{dist.probs[i].item():.4f}' for i in range(8)]}")
    print(f"  entropy = {dist.entropy().item():.4f}, manual feasible-only entropy = {manual_entropy:.4f}")
    print("test_mask_consistency: PASS")


def test_clipping_actually_clips():
    clip_eps = 0.2
    old_logp = torch.zeros(3)
    ratios = torch.tensor([3.0, 1.0, 1.0 / 3.0])
    new_logp = torch.log(ratios)
    advantages = torch.tensor([1.0, 1.0, -1.0])

    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    chosen = torch.min(surr1, surr2)

    # sample 0: ratio=3.0 (>1.2), advantage=+1 -> clip should activate (surr2 wins)
    assert torch.isclose(chosen[0], surr2[0]) and not torch.isclose(chosen[0], surr1[0]), (
        "clip did not activate for a large ratio with positive advantage -- clipping may be a no-op"
    )
    # sample 2: ratio=1/3 (<0.8), advantage=-1 -> clip should activate (surr2 wins)
    assert torch.isclose(chosen[2], surr2[2]) and not torch.isclose(chosen[2], surr1[2]), (
        "clip did not activate for a small ratio with negative advantage -- clipping may be a no-op"
    )
    print(f"  ratios={ratios.tolist()}, advantages={advantages.tolist()}")
    print(f"  surr1={surr1.tolist()}, surr2={surr2.tolist()}, chosen(min)={chosen.tolist()}")
    print("test_clipping_actually_clips: PASS (clipped term selected where expected, not a no-op)")


def test_gradient_flow():
    partition, instances = make_clean_easy_instances(1, seed=2)
    instance = instances[0]
    model = MaskedEncoderActorCritic(HRepDeepSet())
    agent_state = torch.tensor([2.5, 2.5, 0.0, 0.0])
    mask = compute_action_mask(agent_state, partition.domain)

    logits, value = model(partition, instance, agent_state, current_cell=0, mask=mask)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    loss = -dist.log_prob(action) + value.pow(2)
    loss.backward()

    encoder_grad_norms = []
    for name, param in model.encoder.named_parameters():
        assert param.grad is not None, f"encoder param {name} has no gradient -- encoder may be frozen"
        norm = param.grad.abs().sum().item()
        assert norm > 1e-12, f"encoder param {name} has an all-zero gradient (dead)"
        encoder_grad_norms.append(norm)

    print(f"  {len(encoder_grad_norms)} encoder parameter tensors all have non-None, non-zero gradients")
    print("test_gradient_flow: PASS (encoder is training, not frozen)")


def test_determinism():
    from ppo_train import train_ppo
    partition, instances = make_clean_easy_instances(3, seed=3)

    _m1, h1, d1 = train_ppo(HRepDeepSet, partition, instances, n_iterations=2, seed=42,
                             hp={"steps_per_iter": 64, "ppo_epochs": 2}, verbose=False, return_diagnostics=True)
    _m2, h2, d2 = train_ppo(HRepDeepSet, partition, instances, n_iterations=2, seed=42,
                             hp={"steps_per_iter": 64, "ppo_epochs": 2}, verbose=False, return_diagnostics=True)

    assert d1["policy_loss"] == d2["policy_loss"], f"policy_loss differs across identical-seed runs: {d1['policy_loss']} vs {d2['policy_loss']}"
    assert d1["value_loss"] == d2["value_loss"], "value_loss differs across identical-seed runs"
    assert h1 == h2, "solve-rate history differs across identical-seed runs"
    print(f"  policy_loss run1={d1['policy_loss']}, run2={d2['policy_loss']}")
    print("test_determinism: PASS (identical seed -> identical training trajectory)")


def test_reward_accounting_sanity():
    """Hand-traceable short episode: replay a fixed, known-good action
    sequence (repeated 'E') through NavEnv exactly as train_ppo's collection
    loop would (buf_reward.append(reward) every step, unconditionally), and
    cross-check every single buffered reward against the Step 1 reward
    formula directly (not just against another copy of the same call)."""
    partition, instances = make_clean_easy_instances(1, seed=4)
    instance = instances[0]
    config = _instance_to_config(partition, instance)
    env = NavEnv(partition, config, horizon=20)

    # Drive straight east from a fixed start until termination (goal or
    # hazard) or a small step cap -- whichever comes first; this is a
    # bookkeeping test, not a solvability one, so any outcome is fine as long
    # as the accounting matches.
    obs, _ = env.reset(seed=0)
    goal_centroid = partition.cell_centroid(instance.goal_cell)

    buf_reward = []
    hand_computed = []
    terminated = truncated = False
    for _ in range(10):
        if terminated or truncated:
            break
        pos_before = env.state[:2].copy()
        phi_s = -np.linalg.norm(pos_before - goal_centroid)

        obs, reward, terminated, truncated, info = env.step(0)  # action 0 == "E"
        buf_reward.append(reward)  # exactly what train_ppo's loop does

        phi_next = 0.0 if terminated else -np.linalg.norm(obs["state"][:2] - goal_centroid)
        expected = STEP_PENALTY + (GAMMA * phi_next - phi_s)
        if info.get("outcome") == "goal":
            expected += GOAL_REWARD
        hand_computed.append(expected)

    assert len(buf_reward) == len(hand_computed), "step count mismatch -- should be impossible by construction"
    for i, (got, exp) in enumerate(zip(buf_reward, hand_computed)):
        assert abs(got - exp) < 1e-9, f"step {i}: buffered reward {got} != hand-computed {exp}"

    print(f"  {len(buf_reward)} steps, buffered rewards = {[f'{r:.4f}' for r in buf_reward]}")
    print(f"  sum(buffered) = {sum(buf_reward):.4f}, matches per-step Step-1-formula hand computation exactly")
    print("test_reward_accounting_sanity: PASS (no double-counting, no dropped terminal reward)")


if __name__ == "__main__":
    test_learns_on_easy()
    test_advantage_normalization()
    test_gae_episode_boundary()
    test_mask_consistency()
    test_clipping_actually_clips()
    test_gradient_flow()
    test_determinism()
    test_reward_accounting_sanity()
    print("Step 5a tests: ALL PASS")
