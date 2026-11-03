import importlib.util
import unittest
from pathlib import Path
import numpy as np
from recovery_experiment.runtime import bootstrap
ROOT=bootstrap()
import mujoco
from sim.configs import EnvConfig
from recovery_experiment.safety import MeasuredEnv


class LocalPortChecks(unittest.TestCase):
    def test_task_import_is_local_and_no_safety_loss_module(self):
        import sim
        self.assertEqual(Path(sim.__file__).resolve().parent,ROOT/'sim')
        self.assertIsNone(importlib.util.find_spec('sim.safe_diffusion_policy'))
        self.assertIsNone(importlib.util.find_spec('sim.safety_geometry'))

    def test_prohibited_geometry_distance_agrees_with_contact(self):
        env=MeasuredEnv(EnvConfig())
        try:
            env.reset(seed=9700)
            before,_=env.collision_clearance()
            red=env._red_body_ids[0]
            joint=int(env.model.body_jntadr[red]);address=int(env.model.jnt_qposadr[joint])
            env.data.qpos[address:address+3]=env.blue_cube_position()
            mujoco.mj_forward(env.model,env.data)
            after,pair=env.collision_clearance()
            self.assertLess(after,before)
            self.assertLessEqual(after,0.)
            self.assertIsNotNone(pair)
            self.assertTrue(env.measure()['collision'])
        finally:env.close()

    def test_port_matches_reference_reset_when_reference_is_available(self):
        reference=Path.home()/'Documents/LeRobot-SafetyCubes/sim'
        if not (reference/'__init__.py').exists():self.skipTest('Optional supplied reference is not installed')
        import sys
        spec=importlib.util.spec_from_file_location('reference_sim',reference/'__init__.py',submodule_search_locations=[str(reference)])
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        reference_config=module.EnvConfig()
        reference_config.scene.mjcf_path=str(reference/'assets/so101/so101_new_calib.urdf')
        local=MeasuredEnv(EnvConfig());other=module.SafeCubeEnv(reference_config)
        try:
            local_obs,local_info=local.reset(seed=9701)
            ref_obs,ref_info=other.reset(seed=9701)
            np.testing.assert_array_equal(local.data.qpos,other.data.qpos)
            np.testing.assert_array_equal(local_info['cube_positions'],ref_info['cube_positions'])
            np.testing.assert_array_equal(local_obs['state'],ref_obs['state'])
            # Exact same geometry/physics after executing the same joint command.
            for _ in range(3):
                local.step(local_obs['state']);other.step(ref_obs['state'])
            np.testing.assert_allclose(local.data.qpos,other.data.qpos,rtol=0,atol=1e-12)
        finally:local.close();other.close()


if __name__=='__main__':unittest.main()
