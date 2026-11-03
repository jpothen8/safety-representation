# Recovery imitation experiment

This repository implements experiment 1 using the code and assets in `~/Documents/LeRobot-SafetyCubes`, without modifying that repository. The runnable path is nominal expert rollout → executed near-ceiling perturbation → validated expert recovery → temporal data loading → diffusion imitation update → paired raw-policy evaluation. `collect-policy` additionally uses near-boundary baseline-policy events to select the preceding safe carry planning boundary.

Operational lessons and remaining differences from historical runs are tracked in [LESSONS_REVIEW.md](LESSONS_REVIEW.md).

For remote work, run stages from your server tmux terminal:

```bash
cd ~/Documents/Safety-Representation
scripts/run_experiment.sh evaluate
```

The launcher refuses to run without a TMUX environment, removes DISPLAY, forces EGL and tees output to timestamped logs under `artifacts/smoke/logs/`. Tool sandbox access to the tmux socket is blocked: the launcher checks the inherited terminal environment, but cannot independently certify that a tool-owned subprocess survives an agent interruption. To own the process directly in tmux, invoke the launcher from the actual terminal shell. Evaluation resumes completed cases from `results.partial.json`; training resumes saved optimizer/trainable-weight/RNG progress. Use the same configuration and output directory when resuming. Do not change checkpoints or configuration midway through an evaluation.

Use the supplied environment, not an unrelated Python installation:

```bash
cd ~/Documents/Safety-Representation
export SAFETYCUBES_REFERENCE="$HOME/Documents/LeRobot-SafetyCubes"
export MUJOCO_GL=egl
PY="$SAFETYCUBES_REFERENCE/.venv/bin/python"
"$PY" -m unittest discover -s tests -v
"$PY" -m recovery_experiment.run smoke --config configs/smoke.json --output artifacts/smoke
```

Individual stages, including collection from policy rollouts:

```bash
"$PY" -m recovery_experiment.run collect --config configs/smoke.json --output artifacts/smoke
"$PY" -m recovery_experiment.run collect-policy --config configs/smoke.json --output artifacts/smoke
"$PY" -m recovery_experiment.run train --config configs/smoke.json --output artifacts/smoke
"$PY" -m recovery_experiment.run evaluate --config configs/smoke.json --output artifacts/smoke
```

Use a new output directory when changing configuration; reruns overwrite matching artifacts. `smoke` runs controlled-perturbation collection, B/C training, and A/B/C evaluation. `collect-policy` is explicit because the 100-step DDPM sampler on CPU is expensive; its rejected and accepted branches have a separate `policy_manifest.json`. All accepted trajectory files in `data/` are available to training. A missing recovery pool is an error, never silently replaced with nominal data.

## Exact task contract

The current executable reference (`sim/env.py`, `configs.py`, `expert.py`) takes precedence over outdated prose. Blue is the manipulated cube; red cubes are obstacles; the goal is the environment's goal patch. World +y is the start side and −y the goal side. Distances are meters. Observations are two RGB cameras and six joint positions. Actions are six absolute joint-position targets in radians. The supplied λ=0 diffusion checkpoint has one observation step, a 56-step horizon, and executes 50 actions before re-planning. Its original normalization processors are reused and saved with each fine-tuned checkpoint.

Prohibited contacts are red–arm and red–blue. Arm–blue grasp and ordinary placement/table contacts are allowed. **Despite the `ee_height_ceiling` name, the current environment checks held-blue-cube center world z while attached**, with a default threshold of 0.035 m. This implementation preserves that executable definition; it does not introduce a second gripper/TCP-height constraint. Success is the reference's released blue cube, inside the goal square, resting at table height for five control steps. Reference termination applies: first prohibited contact, task success, drop/fly-over where specified by `_terminal`, or 400 control steps; ceiling events are recorded without early termination. Every arm has the same rules.

`MeasuredEnv` samples all five physics substeps of every control action, latches safety events into environment stats, and leaves the success-dwell counter at control frequency. It records reset violations separately. Collision severity is maximum contact penetration. Height severity is signed minimum height margin while grasped. `ee_clearance_proxy` is the reference point-to-box clearance, **not** the clearance of the entire arm; it is used for candidate selection only. Collision labels come from MuJoCo contacts. Contact penetration tolerance and the ceiling are configurable.

## Data and training

`collect` saves successful, safe nominal expert trajectories and controlled branches. Perturbations use IK joint targets executed through normal physics, not teleported qpos. A recovery retains pre-branch observation history, but `supervised_start` prevents training on the perturbation or policy actions. Every suffix action is actually executed by the existing expert. The expert's FSM is restored from its nominal anchor; policy anchors use its existing grasp resynchronization and carry replanning. Acceptance requires a safe perturbation, a near-boundary start, a >1 mm improvement in at least one recorded margin, safe continuation, and task success. Rejected candidates remain recorded with reasons/flags and are excluded from training.

NPZ files contain raw RGB, state/action sequences and privileged provenance. JSON metadata provides source, parent seed, branch index, perturbation actions, initial margins, acceptance, outcome, and aggregate violations. Current collector also writes substep safety traces and anchor physics snapshots. Snapshot restoration requires regenerating the same model/layout with its parent seed. There are no pickled object arrays. Privileged cube positions and grasp state never enter policy inference or the imitation batch.

Splits use disjoint parent seeds and complete trajectories. Branches inherit their parent's split. Evaluation seeds are disjoint from new collection seeds; the `nominal` evaluation suite deliberately uses a training-layout seed to measure retention. The initial checkpoint's historical dataset split is not independently certified; the reference records its training dataset as `data/safe_cube_agg_v7.1`. The smoke generates a tiny clean nominal set with the existing expert instead of copying this historical aggregate.

B and C start from the exact same supplied `safe_diffusion_lam0/.../100000/pretrained_model` checkpoint. B uses nominal data; C uses nominal plus recovery data. Budgets, optimizer, training seed, selected modules and frozen BatchNorm state match. The primary objective calls the upstream **vanilla diffusion MSE**, bypassing the SafeDiffusionPolicy safety-loss path entirely. No differentiable simulator is needed. Recovery ratio is stratified per update (rounded to an integer number of accumulated samples). Images and actions use the checkpoint's normalization. History and action chunks never cross trajectory boundaries or use unlabeled padding.

Training saves resume progress at configurable `save_freq` (default 50 updates) and at the final update. The smoke freezes everything except `diffusion.unet.final_conv`, uses one update and two accumulated samples. This is a pipeline check, not an effective training budget. `configs/experiment.json` is a larger, unexecuted configuration with three training seeds and the full U-Net trainable; run stages explicitly and provision compute accordingly. No automatic model selection on the test suite is implemented.

## Evaluation and artifacts

`results.json` has one row per arm × initial condition × suite; `results.md` gives rates. Suites are nominal, held-out resets and deterministic held-out near-height-boundary states, initialized by the same expert and executed perturbation before each policy. Failed boundary setup is explicitly excluded and logged. A is original, B nominal continuation, C recovery continuation. The exact DDPM inference schedule is preserved, including on CPU. Torch noise seeds and physical initial conditions are paired. No runtime rejection, projection or filter is used; interventions are zero. No filter baseline is reported.

All rates use eligible **episodes** as denominator, not chunks or simulator frames. Task success is environment success. Safe success additionally requires no initial or executed collision/ceiling violation. Collision and ceiling rates in the table are policy-introduced episode events, separately recorded from reset events. Near-boundary safe success is recovery success. Nominal-performance retention is each fine-tuned arm's nominal success rate minus A's nominal rate in percentage points; the ratio is undefined when A has zero nominal success. Count exclusions and inspect per-row reset flags before comparing rates. Tiny smoke rates have no statistical significance.

Artifacts include B/C checkpoints with processors, training logs, per-trajectory manifests/data, MP4 recovery and policy rollouts, effective configuration and reference commit provenance. The original checkpoint is referenced in place, not duplicated. Targeted checks cover contact whitelisting, grasp-gated height, substep-only events, RGB validity, temporal alignment, split isolation, and privileged-input exclusion.

## Dependencies and limits

The reference venv supplies Python 3.14, PyTorch, MuJoCo, LeRobot, diffusers and pytorch_kinematics. Headless EGL and system `ffmpeg` work here; `imageio` is absent and unnecessary. CUDA is unavailable in this session. No new packages were installed. The loader reads action statistics from the saved normalizer for inherited FK state and uses the original processors for actual policy inputs/actions. The primary imitation objective bypasses the safety-loss path.

The reference has a differentiable safety loss, but its surrogate penalties are **not validated or evaluated by this primary recovery smoke**. A separate surrogate experiment must pass correctly normalized privileged batches and compare penalties against actual substep violations before drawing conclusions. The current recovery acceptance test measures improvement at any point in a successful suffix; it does not prove monotonic clearance improvement. Collision selection uses an EE proxy; sampling full-arm near-contact states could improve coverage. Magnetic grasp, high actuator torques, a tiny nominal set and the simulator's current cube-height definition limit physical transfer. This work does not train a safety classifier, infer safety definitions, use RL or alter policy inputs.

After evaluation, generate the table including recovery success and paired nominal retention:

```bash
"$PY" -m recovery_experiment.report --output artifacts/smoke
"$PY" -m recovery_experiment.check_artifacts --output artifacts/smoke
```

The report explicitly marks unfinished evaluation as partial. The loader excludes safe, successful recovery suffixes shorter than the checkpoint horizon; simulation acceptance and training usability are separate. In the initial six-trajectory smoke set, the 53-step training recovery and 40-step validation recovery are too short for a 56-action target. No endpoint actions are fabricated to make them fit. Future controlled collection favors earlier anchors to improve the yield of complete suffixes.
