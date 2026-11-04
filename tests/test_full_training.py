import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from recovery_experiment.data import save_trajectory,Trajectories
from recovery_experiment.cache import TrajectoryCache
from recovery_experiment.optimization import imitation_update


class FullTrainingChecks(unittest.TestCase):
    def test_memory_mapped_samples_match_compressed_history_and_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);rng=np.random.default_rng(2)
            frames=[dict(state=np.full(6,t,np.float32),action=np.full(6,t,np.float32),image=rng.integers(0,255,(8,8,3),np.uint8),
                         wrist_image=rng.integers(0,255,(8,8,3),np.uint8)) for t in range(20)]
            save_trajectory(root,'trajectory',frames,dict(kind='recovery',group='recovery',split='train',seed=1,accepted=True,supervised_start=6))
            cfg=SimpleNamespace(horizon=8,n_obs_steps=3,input_features={'observation.images.agentview':None,'observation.images.wrist':None})
            compressed=Trajectories(root,'train',cfg);mapped=Trajectories(root,'train',cfg,root/'cache')
            a,_=compressed.sample(np.random.default_rng(17),1.);b,_=mapped.sample(np.random.default_rng(17),1.)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            self.assertGreaterEqual(float(b['action'][0,0,0]),6)
            self.assertIsInstance(next(iter(mapped.cache.loaded.values()))['image'],np.memmap)

    def test_cache_invalidates_a_replaced_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'sample.npz';cache=TrajectoryCache(root/'cache')
            np.savez_compressed(path,state=np.zeros((3,6),np.float32))
            first=cache.open(path)['state'];self.assertEqual(float(first.sum()),0)
            np.savez_compressed(path,state=np.ones((4,6),np.float32))
            second=cache.open(path)['state'];self.assertEqual(float(second.sum()),24)
            self.assertEqual(first.shape,(3,6));self.assertEqual(second.shape,(4,6))

    def test_mapped_file_handles_are_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cache=TrajectoryCache(root/'cache',max_open=1)
            for index in range(3):np.savez_compressed(root/f'{index}.npz',state=np.full((2,6),index,np.float32))
            with cache.trajectory(root/'0.npz') as arrays:previous=arrays['state']
            with cache.trajectory(root/'1.npz') as arrays:self.assertEqual(float(arrays['state'].sum()),12.)
            self.assertTrue(previous._mmap.closed)
            self.assertEqual(len(cache.loaded),1)
            with cache.trajectory(root/'0.npz') as arrays:self.assertEqual(float(arrays['state'].sum()),0.)

    def test_full_plan_shares_data_and_holds_out_evaluation(self):
        from recovery_experiment.full import prepare
        spec=Path(__file__).resolve().parents[1]/'configs/full_experiment.json'
        with tempfile.TemporaryDirectory() as folder:
            settings,out,c,configs,shared=prepare(spec,folder)
            self.assertEqual(set(configs),{17,23,42})
            variants=[json.loads(path.read_text()) for path in configs.values()]
            self.assertEqual(len({v['data_root'] for v in variants}),1)
            for v in variants:
                train=set(v['base_expert_seeds'])|set(v['more_expert_seeds'])
                self.assertFalse(train&set(v['eval_seeds']))
                self.assertFalse(train&set(v['validation_seeds']))
                self.assertEqual(len(v['checkpoint_eval_seeds']),10)
                self.assertEqual(v['microbatch_size'],8)
                self.assertFalse(v['wandb']['allow_local_fallback'])

    def test_full_plan_rejects_source_change_after_training_started(self):
        from recovery_experiment.full import prepare
        spec=Path(__file__).resolve().parents[1]/'configs/full_experiment.json'
        with tempfile.TemporaryDirectory() as folder:
            prepare(spec,folder)
            path=Path(folder)/'manifest.json';manifest=json.loads(path.read_text())
            manifest['signature']='prior-source-signature';path.write_text(json.dumps(manifest))
            (Path(folder)/'training_started.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'after training started'):prepare(spec,folder)

    def test_strict_tracking_failure_is_persisted(self):
        from unittest.mock import patch
        from recovery_experiment.tracking import Tracker
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/('long-output-name-'*6)
            with patch('recovery_experiment.tracking.require_local_socket',side_effect=PermissionError('socket blocked')):
                with self.assertRaises(PermissionError):
                    Tracker(directory,'test',{'wandb':{'mode':'online','allow_local_fallback':False}})
            status=json.loads((directory/'tracking_status.json').read_text())
            self.assertFalse(status['active'])
            self.assertIn('socket blocked',status['error'])

    def test_interrupted_trajectory_write_cannot_look_complete(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            frames=[dict(state=np.ones(6,np.float32),action=np.ones(6,np.float32)) for _ in range(2)]
            def interrupted(path,**arrays):
                Path(path).write_bytes(b'incomplete archive')
                raise OSError('simulated interrupted write')
            with patch('recovery_experiment.data.np.savez_compressed',side_effect=interrupted):
                with self.assertRaises(OSError):save_trajectory(root,'episode',frames,dict(accepted=True))
            self.assertFalse((root/'episode.json').exists())
            self.assertFalse((root/'episode.npz').exists())
            saved=save_trajectory(root,'episode',frames,dict(accepted=True))
            with np.load(root/saved['file'],allow_pickle=False) as arrays:
                self.assertEqual(arrays['state'].shape,(2,6))
            self.assertEqual(json.loads((root/'episode.json').read_text())['frames'],2)

    def test_uneven_microbatches_preserve_the_mean_gradient(self):
        class Policy:
            def __init__(self):self.model=torch.nn.Linear(2,1,bias=False)
            def imitation_loss(self,batch):return ((self.model(batch['x'])-batch['y'])**2).mean()
        torch.manual_seed(1);a=Policy();b=Policy();b.model.load_state_dict(a.model.state_dict())
        examples=[{'x':torch.tensor([[float(i),1.]]),'y':torch.tensor([[float(i)*.1]])} for i in range(3)]
        oa=torch.optim.SGD(a.model.parameters(),lr=.01);ob=torch.optim.SGD(b.model.parameters(),lr=.01)
        la,_=imitation_update(a,oa,examples,1);lb,_=imitation_update(b,ob,examples,2)
        self.assertAlmostEqual(la,lb,places=6)
        torch.testing.assert_close(a.model.weight,b.model.weight,rtol=1e-6,atol=1e-7)


if __name__=='__main__':unittest.main()
