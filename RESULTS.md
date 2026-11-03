# Imitation-only collision recovery: first local round

The task, assets and expert now run from this repository. The active policy is stock LeRobot `DiffusionPolicy`, with standard diffusion MSE only. No safety-loss module is present. All **273,977,798 parameters** were trainable during each round; BatchNorm running statistics stayed fixed consistently.

The supplied checkpoint already had 100,000 task-training updates and used an expert aggregate that may include historical cleanup demonstrations. This pilot measures additional fine-tuning, not safety learning from a pristine task-naive policy.

| Arm | Additional training after import | Held-out full-task safe success | Held-out near-collision safe success |
|---|---|---:|---:|
| A: expert round | 20 full-task expert updates | 1/1 | 1/1 |
| B: more full-task expert | A + 20 matched continuation updates | 0/1 | 1/1 |
| C: targeted collision recovery | A + 20 matched continuation updates | 1/1 | 1/1 |

B's full-task rollout terminated on a prohibited collision, with maximum observed penetration about **0.52 mm**. No other evaluated case introduced a collision or ceiling violation. Reset violations were absent. Each suite uses the same held-out parent seed 9600 and the same physical starting state across arms, with paired diffusion noise seeds. The near-collision state is independently generated on that held-out parent, at approximately 14.1 mm geometry clearance. No runtime filtering or expert assistance occurs during policy evaluation.

These are **one-seed, one-case-per-suite pilot results**, not statistical evidence that recovery training improves safety. A already had zero violations on these cases; this pilot does not demonstrate a reduction relative to A. B's failure must not be generalized to nominal training as a method.

## Data and matched budgets

The initial expert round used two complete successful expert trajectories. One of two new full-task attempts timed out and was excluded; the successful new trajectory was used for B. One of two recovery parents yielded a validated recovery at 11.0 mm initial collision clearance, increasing to 44.9 mm by task completion, with a 131-action expert suffix. The other parent's failed attempts were excluded. Perturbations were physically executed XY commands; no height-only data or teleported training states were used.

B/C both initialized from A's exact checkpoint and used 20 updates, two accumulated examples per update, learning rate 1e-5, the same training seed and all policy modules. At each update the shared nominal trajectory/window was exactly identical. B's additional example was a new full-task window; C's additional example was a recovery window. The audit verifies these conditions. Data-generation cost and trajectory lengths were not matched; successful augmentation pools contain only one trajectory each.

## Saved outputs

Every saved training checkpoint has a raw-policy validation video, metrics and a 2 ms safety trace. Validation seed 9500 is separate from the final held-out seed. All three validation videos succeeded safely.

- [Expert-round checkpoint video](artifacts/round1/expert_round/checkpoints/000020/evaluation/rollout.mp4)
- [More-full-task-expert checkpoint video](artifacts/round1/more_expert/checkpoints/000020/evaluation/rollout.mp4)
- [Collision-recovery checkpoint video](artifacts/round1/recovery_round/checkpoints/000020/evaluation/rollout.mp4)
- [Per-suite comparison table](artifacts/round1/comparison.md) and [raw results](artifacts/round1/comparison.json)
- [Data manifest](artifacts/round1/data_manifest.json) and [budget/video audit](artifacts/round1/audit.json)

Eight local safety/data/metric/port tests passed, including exact reset/short-motion parity against the supplied reference. All checkpoint videos were decoded and checked for black frames. Restarting the comparison reused completed training and rollout artifacts without adding updates.

## External services and next scale

WandB online runs, grouped stage metrics, checkpoint videos and local event replay are implemented. No live WandB run was created here: the sandbox blocks local service sockets, even in offline mode. `tracking_status.json` records that failure and `tracking_events.jsonl` preserves metrics/video references. From the native authenticated tmux terminal, `scripts/run_rounds.sh sync-tracking` uploads them.

The host GitHub login was confirmed by the user as `jpothen8`. The sandbox cannot reach GitHub's API, so this assistant has not verified or completed publication. `scripts/publish_github.sh` creates a new source-only repository from the native authenticated terminal. Model weights, data, credentials and videos are excluded from Git.

The CPU pilot is complete. The larger GPU configuration has not run; GPU devices are unavailable in the tool sandbox. No power/fan settings were changed. Wider training and evaluation, more recovery parents, multiple training seeds and reporting data-generation cost are needed before drawing conclusions.

Earlier height-boundary pipeline checks are archived in [the historical smoke report](docs/archive/RESULTS_smoke.md); they are not this collision-only experiment.
