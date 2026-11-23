# Effective-draw soft-label distillation

This is an offline Actor fine-tune, not PPO and not outcome imitation.
It uses the preselected public more-copies decisions from a complete replay.
Winning and losing branch outcomes do not select, weight, or label samples.

## Data

Preparation verifies the frozen actor, native extension, critical sources,
replay artifact hashes, and original episode hashes. It replays each original
episode once, validates every public observation fingerprint and action, and
reconstructs full public V4 tensors at selected roots. No hidden information
or terminal reward enters the Actor input or teaching target.

Targets are legal discards with the same three-route shanten vector and maximum
effective remaining copies on the uniquely closest route. Tied maximum candidates
share target probability uniformly. Available Hu decisions remain pure reference
anchors rather than teaching the Actor to skip Hu.

Each original episode also contributes up to two non-root, non-forced anchors per
decision group: discard, response, Hu, ready, kan. Anchors preserve the original
R5 probability distribution, not just its greedy action. Rows are unique by
episode/decision index; copied baseline suffixes are not duplicated.

All seats and derived branches of one original seed belong to the same split.
The split is approximately 80% train, 10% validation, 10% development holdout.
Aggregate replay results were previously inspected, so this is not a sealed,
untouched final test. New-wall matchups supply the gameplay check.

## Learning

For rule rows: `target = 0.9 * frozen_R5_probability + 0.1 * uniform_best_rule`.
For anchor rows: `target = frozen_R5_probability`.

Cross entropy trains the existing Actor base parameters. The reference checkpoint
is frozen, the kan adapter weights are frozen, and the Critic is neither loaded
nor trained. Shared base parameters can still change kan behavior; anchor metrics
and gameplay evaluation must therefore check it. Dropout stays disabled to match
the deployed deterministic policy; GRUs stay in training mode for cuDNN backward
and gradients still propagate normally.

Defaults: CUDA when available, batch 128, eight epochs, learning rate linearly
decreasing from 3e-5 to 3e-6, no weight decay. Validation selects the lowest target
KL subject to anchor KL <= 0.02 and anchor argmax agreement >= 97%. The original
checkpoint remains the fallback if no epoch passes. Holdout is used only after
checkpoint selection. Neither policy preferences nor validation improvements
prove an increase in Arena net score.

`best_target_actor.pt` is also saved using target KL alone, without the anchor
guard. `target_candidate_summary.json` reports its drift and guard result.
This separately named experimental candidate permits testing stronger rule
weights without silently evaluating the fallback original model. It is never
automatically deployed, even if it passes the validation guard.

