# Full experiment: imitation-only recovery rounds

The expert preview is [here](artifacts/expert_preview/expert_seed9400.mp4). It shows only the local scripted expert on eight red obstacles, with agent and wrist cameras. It completed safely in 270 control steps / 1,350 physics samples. Playback is explicitly labeled 0.30× simulator time for inspection.

The [port audit](docs/port_audit.json) compares the local task against reference commit `c81ebec7f8113cbf4616ef7c055b22b148ec3135`. Every asset is present; only trailing URDF XML whitespace differs. Environment, scene, expert, randomization helpers and compatibility code match. Config defaults match except for the local asset path. A complete expert episode produced **zero differences** in actions, joint states, camera pixels and outcome measurements. Unrelated π0 and safety-loss training code is intentionally outside this task port.

## Fixed full-run design

[configs/full_experiment.json](configs/full_experiment.json) declares:

| Component | Full-run setting |
|---|---|
| Training seeds | 17, 23, 42 |
| Initial nominal expert data | 150 successful complete trajectories |
| Additional nominal data | 150 successful complete trajectories |
| Recovery data | 50 safe, successful near-collision expert suffixes |
| Validation data | 20 complete successful expert trajectories; 10 validation initial conditions per checkpoint |
| Held-out nominal cases | 100 paired initial conditions per training seed |
| Held-out near-collision cases | Up to 50 paired, expert-validated starting states per training seed; failed setup is recorded as an exclusion |
| Initial expert round | 2,000 updates, all policy modules |
| Nominal/recovery continuation | 1,000 updates each, same parent checkpoint and optimizer settings |
| Effective batch | 32 examples, microbatches of 8 |
| Continuation mixture | 16 identical shared nominal windows plus 16 additional nominal or recovery windows |
| Learning objective | Stock diffusion MSE only |
| Checkpoints | Every 250 updates, with raw-policy video, safety trace and validation outcomes |

The expert parent checkpoint is selected using validation safe-task success (ties: task success, then earlier step). Both continuation arms initialize from that exact parent. Primary B/C results use their matched final training budgets. Test cases never select checkpoints. The original imported policy is task-pretrained and may already contain historical cleanup demonstrations; this is a controlled study of additional fine-tuning.

Parent-seed pools are disjoint between initial nominal, additional nominal, validation and evaluation cohorts. Recovery parents belong to the initial training cohort. Shared recovery data and independently generated test starting states are reused across training seeds. Failed expert runs and recovery attempts do not enter imitation training. Every full-run recovery must contain executed XY perturbation actions and at least a 2 mm reduction in actual collision clearance before the expert continues. The pilot's accepted sample was naturally near-boundary, so it is explicitly classified as targeted replay and excluded from this new recovery cohort. Collection aborts if reserved seed pools cannot meet the requested success counts.

## Performance and reproducibility

Compressed NPZ episodes are expanded once into bounded, memory-mapped caches. Training reads only the needed history/window, rather than decompressing every episode for every sample. The cache limits open file handles and invalidates replaced sources. Actual batched GPU forwards support gradient accumulation, including partial final microbatches.

A real two-example CPU probe updated all 273,977,798 policy parameters with finite gradients; its weights were discarded. CUDA peak memory and a real GPU update are checked by the native full-run probe before long training. Source and model/config identities are recorded; source/spec changes are rejected after training starts. Preparation-only code revisions are logged before the first model update, with task/collection code immutable after data collection.

Validation inference is isolated from CPU/CUDA training RNG. Optimizer, weights and random state are checkpointed. Interrupted checkpoint videos/validation are repaired on resume. Each new checkpoint has a main rollout video plus ten validation cases; raw held-out rollouts are saved too. Policies use the original checkpoint's 56-step horizon and 50-action queue with no runtime safety filtering.

Results retain per-training-seed paired metrics and paired bootstrap intervals/exact discordant-case tests. Replicated initial conditions are not pooled as independent observations. Clean-reset denominators, unsafe resets, setup exclusions, recovery success and nominal retention are recorded separately. A larger sample supports analysis; it does not guarantee a positive result.

## Start from the native server shell

```bash
cd ~/Documents/Safety-Representation
scripts/start_full.sh
```

This creates a detached `cubes-full` tmux session that survives remote disconnection. It checks CUDA, available disk, the audited task, and a real online WandB connection before training. GPU power/fan settings are untouched. Inspect with `tmux attach -t cubes-full`; logs are under `artifacts/full_experiment/logs/`.

The current tool sandbox cannot access GPU devices or service sockets. Its preflight records that failure; **the full GPU training job has not been launched from this sandbox**. Two successful demonstrations from the actual full-run training cohort are already collected and will be reused. All remaining collection and training must be owned by the native tmux process.

For explicit stages:

```bash
scripts/run_full.sh preflight
scripts/run_full.sh run
scripts/run_full.sh summarize
```

The full runner groups WandB stage runs by experiment and records each training seed, validation videos, held-out videos and metrics. Strict online tracking is required; it does not silently fall back to local logging. The existing pilot's local tracking status does not verify the native upload, so preflight checks it again.

To publish these updates to the existing repository, run `scripts/push_updates.sh` from the authenticated native shell. Weights, data and generated videos remain excluded from Git.
