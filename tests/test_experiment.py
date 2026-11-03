import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from recovery_experiment.runtime import bootstrap
bootstrap()
from sim.configs import EnvConfig
from recovery_experiment.safety import MeasuredEnv
from recovery_experiment.data import Trajectories, save_trajectory


class Checks(unittest.TestCase):
    def test_allowed_contacts_and_ceiling_gate(self):
        env = MeasuredEnv(EnvConfig())
        try:
            env.reset(seed=8400)
            # Ordinary table/red resting contacts exist at reset and are allowed.
            self.assertFalse(env.measure()['collision'])
            self.assertIsNone(env.measure()['height_margin'])
            self.assertGreater(env._observe()['image'].std(), 1.)
            # Classification against explicit synthetic contact pairs, using the
            # actual model's contact sets. Blue-red and arm-red are prohibited;
            # arm-blue and red-table are allowed.
            original_data = env.data
            red = next(iter(env._red_geom_ids))
            arm = next(iter(env._arm_geom_ids))
            blue = next(iter(env._blue_geom_ids))
            table = next(g for g in range(env.model.ngeom) if g not in env._red_geom_ids | env._arm_geom_ids | env._blue_geom_ids)
            original_grasp = env._blue_grasped
            original_clear = env.current_clearance
            env._blue_grasped = lambda: False
            env.current_clearance = lambda: .03
            for pair, expected in [((red, blue), True), ((arm, red), True), ((arm, blue), False), ((red, table), False)]:
                env.data = SimpleNamespace(contact=[SimpleNamespace(geom1=pair[0], geom2=pair[1], dist=-.001)], time=0.)
                self.assertEqual(env.measure()['collision'], expected)
            env.data = original_data
            env._blue_grasped = original_grasp
            env.current_clearance = original_clear
        finally:
            env.close()

    def test_substep_event_survives_safe_endpoint(self):
        env = MeasuredEnv(EnvConfig())
        try:
            obs, _ = env.reset(seed=8401)
            env.begin_measurement()
            original = env.measure
            count = [0]
            def transient():
                sample = original()
                sample['height'] = count[0] == 0
                count[0] += 1
                return sample
            env.measure = transient
            env.step(obs['state'])
            self.assertEqual(len(env.samples), env.cfg.sim_substeps)
            self.assertFalse(env.samples[-1]['height'])
            self.assertTrue(env.summary()['introduced_height'])
            self.assertTrue(env._stats.ceiling_violation)
        finally:
            env.close()

    def test_headless_environment_and_planning_queue(self):
        import os
        from collections import deque
        from recovery_experiment.policy import Policy
        self.assertNotIn('DISPLAY', os.environ)
        self.assertEqual(os.environ['MUJOCO_GL'], 'egl')
        policy = Policy.__new__(Policy)
        policy.model = SimpleNamespace(_queues={'action': deque()})
        self.assertTrue(policy.planning_boundary())
        policy.model._queues['action'].append(np.zeros(6))
        self.assertFalse(policy.planning_boundary())

    def test_metric_denominators_and_paired_retention(self):
        from recovery_experiment.report import summarize
        def row(arm, seed, success):
            return dict(arm=arm,seed=seed,suite='nominal',success=success,safe_success=success,
                safety=dict(introduced_collision=not success,introduced_height=False,
                            reset_collision=False,reset_height=False,min_height_margin=.004),
                stats=dict(min_clearance=.02))
        rows = [row('A',1,True),row('B',1,False),row('B',2,True),
                dict(arm='B',seed=3,suite='nominal',excluded=True)]
        result = next(r for r in summarize(rows) if r['arm']=='B')
        self.assertEqual(result['n'],2)
        self.assertEqual(result['excluded'],1)
        self.assertEqual(result['success'],.5)
        # Only the paired initial condition contributes to retention.
        self.assertEqual(result['nominal_retention_pp'],-100.)

    def test_temporal_chunks_and_privileged_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            frames = [dict(state=np.full(6, t, np.float32), action=np.full(6,t,np.float32),
                           image=np.zeros((8,8,3),np.uint8), wrist_image=np.zeros((8,8,3),np.uint8),
                           grasped=np.ones(1,np.float32)) for t in range(12)]
            for split, seed in [('train',1),('validation',2)]:
                save_trajectory(d, split, frames, dict(kind='recovery', split=split, seed=seed,
                    accepted=True, supervised_start=4))
            save_trajectory(d, 'short', frames[:5], dict(kind='recovery',split='train',seed=3,accepted=True,supervised_start=0))
            cfg = SimpleNamespace(n_obs_steps=3, horizon=6, input_features={'observation.images.agentview':None})
            ds = Trajectories(d, 'train', cfg)
            self.assertEqual(len(ds.groups['recovery']),1)
            self.assertEqual(len(ds.excluded),1)
            batch, _ = ds.sample(np.random.default_rng(2), 1.)
            self.assertEqual(tuple(batch['action'].shape),(1,6,6))
            np.testing.assert_array_equal(batch['observation.state'][0,:,0],batch['action'][0,:3,0])
            self.assertGreaterEqual(batch['observation.state'][0,-1,0],4)
            self.assertGreaterEqual(batch['action'][0,0,0],4)
            self.assertFalse(any('privileged' in k or 'grasped' in k for k in batch))


if __name__ == '__main__':
    unittest.main()
