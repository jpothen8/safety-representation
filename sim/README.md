# Local cube task and expert

`env.py`, `scene.py`, `configs.py`, `expert.py`, `randomize.py` and the robot assets are ported from the supplied reference. `PORT_PROVENANCE.json` records its commit, original file hashes and local changes. `configs.py` resolves the URDF from this directory so the environment does not need the neighboring working tree. LeRobot supplies the standard diffusion policy as a pinned dependency; the task and expert are local.

Blue is the manipulated cube; red cubes are prohibited obstacles. The expert picks the blue cube, carries it low through the red field and releases it inside the goal square. World +y is the initial side and −y the goal side. Observations are agentview/wrist uint8 RGB images plus six measured joint positions. Actions are six absolute joint-position targets in radians.

The original magnetic grasp and A* carry replanning are retained. Snapshot restoration includes the attachment offset and gripper bookkeeping, in addition to MuJoCo physics state. A restored state requires the same model and layout; saved experiment snapshots include a parent seed so the identical model/layout can be reconstructed.

Prohibited contacts are red–arm and red–blue; intended arm–blue grasp contacts and table contacts are allowed. Despite the legacy `ee_height_ceiling` field name, the executable environment checks held-blue-cube center world z while grasped, default 0.035 m. Released placement inside the goal, resting at table height for five control steps, triggers success. No safety-learning objective is implemented here.

The scene uses 2 ms physics steps and five physics steps per control action. The experiment's safety monitor labels every physics substep; the success dwell remains at control-step frequency. Near-collision candidates use MuJoCo geometry distance over prohibited pairs, separate from the original EE point-clearance diagnostic. Expert/perturbation state and labels never become extra diffusion-policy observations.
