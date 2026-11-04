# Diffusion recovery cubes

A local MuJoCo red-cube/blue-cube task and expert, with **imitation learning only**. Safety behavior is learned through targeted rounds of near-collision expert demonstrations. There is no safety loss, safety classifier, RL, or runtime safety filter.

The task, scene builder, SO-101 robot assets, magnetic grasp and A* expert are ported into `sim/` from the supplied LeRobot-SafetyCubes repository. Task semantics are preserved; see [NOTICE](NOTICE), [port provenance](sim/PORT_PROVENANCE.json), and [operational lessons](LESSONS_REVIEW.md). The task no longer imports simulation or expert code from the neighboring repository. LeRobot is a framework dependency pinned to the supplied fork’s exact source commit in `requirements.txt`; it is not assumed to be a published PyPI release.

## Experiment

1. **Expert round E:** load the supplied task-pretrained diffusion checkpoint, then train on full pick-to-goal expert demonstrations with the standard diffusion MSE. All policy modules are trainable by default; BatchNorm running statistics are held fixed consistently.
2. **A:** retain E's final checkpoint as the reference.
3. **B:** initialize from E and continue with shared nominal replay plus **new full-task expert demonstrations**.
4. **C:** initialize from the identical E checkpoint and continue with shared nominal replay plus **near-collision expert recovery suffixes**.

B/C have equal update counts, learning rate, accumulated sample count, training seed and selected modules. Each update has the same number of shared nominal versus additional examples. Additional examples differ by full-task versus recovery data. Every supervised target is a coherent sequence of actually executed expert actions. Perturbation/scout actions are excluded. Full parent trajectories and initial-condition seeds are split across training, validation and evaluation.

The checkpoint already has 100,000 task-training updates; this is **continued expert training**, not a claim to train from scratch. Its historical training source is an expert aggregate that may already include cleanup demonstrations. Weight import rejects nonzero source safety-loss weights, strips safety-specific config fields and preserves model/normalizer weights. The resulting policy is the stock LeRobot `DiffusionPolicy`, not `SafeDiffusionPolicy`.

## Install and import weights

The current server's reference venv can be used without installing anything. For a new machine, create a Python environment and install [requirements.txt](requirements.txt) plus system `ffmpeg`. The tested framework identifies itself as LeRobot 0.5.2 and is installed from the pinned source commit; Python 3.14 needs the bundled draccus compatibility shim. Dependency installation in a fresh environment has not been validated here.

```bash
cd ~/Documents/Safety-Representation
export SAFETYCUBES_PYTHON="$HOME/Documents/LeRobot-SafetyCubes/.venv/bin/python"
"$SAFETYCUBES_PYTHON" scripts/import_checkpoint.py \
  --source "$HOME/Documents/LeRobot-SafetyCubes/outputs/safe_diffusion_lam0/checkpoints/100000/pretrained_model"
"$SAFETYCUBES_PYTHON" -m unittest discover -s tests -v
"$SAFETYCUBES_PYTHON" scripts/audit_round.py --output artifacts/round1
```

Weights are copied to `checkpoints/base/` and excluded from Git. `provenance.json` records their source and SHA-256. On another machine set `SAFETYCUBES_PYTHON` to that machine's installed interpreter; no neighboring task repository is required after checkpoint import.

## Run inside tmux

Always start the launcher from your **native server tmux shell**, not a remote-display-owned terminal process:

```bash
tmux new -s cube-recovery
cd ~/Documents/Safety-Representation
export SAFETYCUBES_PYTHON="$HOME/Documents/LeRobot-SafetyCubes/.venv/bin/python"
scripts/run_rounds.sh run configs/round1_cpu.json artifacts/round1
```

Individual resumable stages:

```bash
scripts/run_rounds.sh collect-expert
scripts/run_rounds.sh train-expert
scripts/run_rounds.sh collect-recovery
scripts/run_rounds.sh compare
```

The checked-in CPU pilot uses 20 updates for E, then 20 each for B/C, two accumulated examples per update, and one training seed. This is an executable pilot, not a converged or statistically powered experiment. Budgets and trainable modules are configurable. A GPU configuration is provided separately; GPU availability is checked before running.

The launcher forces headless EGL, removes DISPLAY, logs through `tee`, and prevents duplicate launches of the same stage/output. Snapshot/optimizer/RNG checkpoints and per-trajectory files make restarts possible. Do not modify code/configuration or checkpoint files during a running experiment. Current sandbox constraints block the tmux socket, GPU devices, local service sockets and GitHub API access; commands invoked directly from the authenticated native tmux terminal can have different access.

## Near-collision recoveries

The full experiment generates candidates through physically executed, clearance-reducing XY perturbations of grasped expert carry states. The earlier pilot also accepted naturally near-boundary expert states; metadata now distinguishes those from perturbed recovery states. The nearest prohibited pair's **MuJoCo geometry distance** must be between configurable 2 and 15 mm; this is not a height-only scene or a point-at-TCP proxy. Height remains monitored as an environment constraint. No qpos teleport is used to produce perturbations. Simulation snapshots include the magnetic-grip state.

The expert replans from the actual perturbed state and completes the task. A suffix is accepted only if the perturbation and continuation avoid collision/ceiling violations, the task succeeds, actual collision margin improves, and at least a full 56-action target is available. Failed candidates and attempts are retained as metadata and excluded from training. The pilot uses controlled expert-state perturbations; it does not claim successful policy-scout mining. Wider on-policy scout rounds can use the planning-boundary mechanism retained in the legacy collector.

Policy inputs are the original agentview and wrist RGB images and six joint positions. Privileged state guides collection and measurement only. Actions are six absolute joint targets. The checkpoint preserves one observation step, horizon 56, 50 queued executed actions and the original DDPM inference schedule. No per-step queue reset or shortened inference schedule is used to improve results.

## Checkpoint videos and WandB

Every saved checkpoint receives a **raw-policy validation rollout MP4**, metrics and physics-step safety trace under:

```
artifacts/round1/<stage>/checkpoints/<update>/evaluation/
  rollout.mp4
  metrics.json
  safety_trace.json
```

The video seed belongs to validation; final held-out evaluation uses separate initial conditions. Resume regenerates missing checkpoint videos without performing extra training updates. Video inference RNG is isolated from training RNG.

`wandb` defaults to online mode, project `safe-cube`, one grouped run per training stage. It logs imitation loss, gradient norm, nominal/recovery sample counts, checkpoint success/safety metrics and checkpoint videos. Weight uploads are disabled by omission, matching the reference's intent to avoid giant model artifacts. The run ID is stable across restarts. Configure `wandb.entity` for the desired team.

If WandB cannot start, `tracking_status.json` records the error and `tracking_events.jsonl` retains metrics and video references. A separate `paired_comparison` run contains held-out outcome metrics and raw rollout videos, including failures. No dashboard/upload is claimed in this case. Set `wandb.allow_local_fallback=false` for a strict production run. In the current sandbox, even offline WandB cannot open its local sockets. From a native authenticated terminal, replay local metrics/videos with:

```bash
scripts/run_rounds.sh sync-tracking
```

## Evaluation

The final comparison reports task success, safe task success, collision and ceiling episode rates on held-out full-task resets and independently generated held-out near-collision states. Raw checkpoint policies are used without rejection/projection/filtering. Safety is measured every 2 ms simulator substep, including motion within action chunks; reset events are recorded separately. Near-collision safe success is recovery success. Compare held-out nominal success to E to measure nominal retention. Rates use eligible episodes as denominator; failed boundary generation is recorded separately.

The exact environment contract is preserved: red–arm and red–blue contacts are prohibited; intended grasp/table contacts are allowed. The executable reference's height check uses held-cube center world z while attached, with threshold 0.035 m, despite the old gripper/EE naming. Success requires released placement inside the goal square and five resting control steps. The port preserves the current code's geometry defaults rather than substituting conflicting prose values.

Earlier pipeline smoke results are retained in [RESULTS.md](RESULTS.md) and `artifacts/smoke/`; they are distinct from this staged round experiment. Historical details are in [the archived README](docs/legacy_experiment.md); the old height-boundary collector and configs are inactive text archives under `docs/archive/`. Tiny successful rollouts with zero baseline violations cannot establish a safety benefit.

## Publish

The repository target is a **new** `jpothen8/safety-representation` repository. To publish from the host where `gh auth status` succeeds:

```bash
scripts/publish_github.sh
```

The script verifies the authenticated account, refuses an existing repository, commits explicit source paths and pushes without force. Data, model weights, logs, credentials and videos stay outside Git. Git metadata is stored under `.publication/git` because the tool workspace's `.git` directory is managed read-only. Set `GIT_AUTHOR_NAME`/`GIT_AUTHOR_EMAIL` to override the default username/noreply author. Successful publication must be verified from the returned repository URL; preparing this script is not publication.

## Full experiment and verified expert preview

The [full experiment plan](FULL_EXPERIMENT.md) specifies three training seeds, 150 initial expert successes, 150 additional expert successes, 50 recovery demonstrations and paired held-out evaluation. It uses cached temporal windows and true batched forwards for GPU training. Start it from the native server shell with `scripts/start_full.sh`.

A new [expert-only two-camera video](artifacts/expert_preview/expert_seed9400.mp4) is available. The [complete source/asset/rollout audit](docs/port_audit.json) found identical actions, physics states, camera pixels and outcomes across a full 270-step local/reference episode. Reproduce with `scripts/record_expert.py` and `scripts/audit_port.py` using the installed Python environment.
