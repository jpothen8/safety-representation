# First experiment: completed pipeline smoke

The supplied nominal diffusion checkpoint and both one-update continuations completed the same three paired initial conditions with their original raw-policy sampling settings. One training seed (17) and one case per suite were used; these are execution checks, not evidence of a safety improvement.

| Arm | Task success | Safe success | Collision episodes | Ceiling episodes | Recovery success | Nominal retention |
|---|---:|---:|---:|---:|---:|---:|
| A: supplied baseline | 3/3 | 3/3 | 0/3 | 0/3 | 1/1 | reference |
| B: nominal continuation | 3/3 | 3/3 | 0/3 | 0/3 | 1/1 | 0 pp change |
| C: nominal + recovery | 3/3 | 3/3 | 0/3 | 0/3 | 1/1 | 0 pp change |

Suites: nominal layout 8100, held-out reset 8300, held-out near-height-boundary state from 8300. The initial simulator measurements match exactly across arms within each suite. Reset violations were absent; no runtime safety filter or intervention was used. Collision and height events were measured every 2 ms physics substep, including within action chunks. Counts are episodes; recovery uses the single near-boundary case. The full case table and latency/margin measurements are in [summary.md](artifacts/smoke/summary.md) and [results.json](artifacts/smoke/results.json).

Regenerated, reviewed data contains two training nominal trajectories, two training recoveries, one validation nominal trajectory and one validation recovery. All six are safe successes, both camera streams are nonblack, and every expert suffix fits a complete 56-action window. Recovery suffix lengths are 102, 82 and 85 actions. Snapshots include the magnetic-grip state. Substep traces match their reported sample counts and 2 ms timestamps. See [data verification](artifacts/reviewed/data_verification.json), [manifest](artifacts/reviewed/manifest.json), and [checkpoint verification](artifacts/reviewed/verification.json).

B and C each performed one standard diffusion-imitation update with two accumulated samples, learning rate 1e-5, training only the final output block. B used two nominal samples; C used one recovery and one nominal sample. Gradients were finite; selected weights changed while a frozen tensor remained identical. Optimizer/weight/RNG resume was checked. Repeating training on the reviewed data produced model weights, policy configs and normalization processors byte-identical to the evaluated policies: [equivalence certificate](artifacts/reviewed/checkpoint_equivalence.json).

Eighteen targeted tests passed: five local safety/data/metric checks and thirteen upstream snapshot/diffusion tests. All fifteen completed representative MP4 videos were decoded and checked for black frames: [video audit](artifacts/smoke/video_verification.json). [Reproduction commands](README.md) and [the full lesson review](LESSONS_REVIEW.md) describe exact semantics and operational constraints.

Limitations: the baseline had no violations in these three cases, so this sample cannot demonstrate a reduction. The single policy-scout pilot yielded no qualifying planning anchor; validated recovery data comes from physically executed perturbations of expert carry states. The optional differentiable safety-loss arm was not run or revalidated here. This is a tiny new nominal set, not continued training over the full historical BC dataset. CUDA/driver access is unavailable in this session, and the larger three-seed configuration remains unexecuted. Local logs are saved; no WandB dashboard was created. The tmux environment is present but its socket is blocked from the tool sandbox; launch directly from the actual tmux shell for terminal ownership. No GPU power/fan settings were changed.
