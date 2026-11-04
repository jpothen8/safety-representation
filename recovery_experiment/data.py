"""Whole-trajectory storage and checkpoint-compatible temporal sampling."""
import json
from pathlib import Path
import numpy as np
import torch


def save_trajectory(root, name, frames, meta):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.stack([f[k] for f in frames]) for k in frames[0]}
    path = root / f'{name}.npz'
    temporary=path.with_name('.'+name+'.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    meta = dict(meta, file=path.name, frames=len(frames))
    metadata_path=root/f'{name}.json'
    metadata_tmp=metadata_path.with_suffix('.json.tmp')
    metadata_tmp.write_text(json.dumps(meta, indent=2, allow_nan=False))
    metadata_tmp.replace(metadata_path)
    return meta


def frame(obs, action, info):
    return dict(state=obs['state'].copy(), image=obs['image'].copy(), wrist_image=obs['wrist_image'].copy(),
                action=np.asarray(action, np.float32), cube_positions=info['cube_positions'].copy(),
                grasped=np.array([info['grasped']], np.float32))


class Trajectories:
    def __init__(self, root, split, cfg, cache_root=None):
        self.cfg = cfg
        from .cache import TrajectoryCache
        self.cache = TrajectoryCache(cache_root) if cache_root else None
        self.groups = {'nominal': [], 'recovery': []}
        self.excluded = []
        for p in sorted(Path(root).glob('*.json')):
            m = json.loads(p.read_text())
            if m.get('split') == split and m.get('accepted'):
                if m['frames'] - m.get('supervised_start', 0) < cfg.horizon:
                    self.excluded.append(dict(file=m['file'], reason='expert_suffix_shorter_than_horizon'))
                    continue
                self.groups[m['kind']].append((m, p.parent / m['file']))

    def sample(self, rng, recovery_ratio):
        kind = 'recovery' if rng.random() < recovery_ratio else 'nominal'
        if not self.groups[kind]:
            raise ValueError(f'No validated {kind} trajectories; collect more seeds before training')
        m, path = self.groups[kind][rng.integers(len(self.groups[kind]))]
        source = self.cache.trajectory(path) if self.cache else np.load(path, allow_pickle=False)
        with source as d:
            # Only complete chunks are used: no unlabelled endpoint padding.
            lo = m.get('supervised_start', 0) + self.cfg.n_obs_steps - 1
            hi = len(d['action']) - (self.cfg.horizon - self.cfg.n_obs_steps + 1)
            if hi < lo:
                raise ValueError(f'Trajectory too short for horizon: {path}')
            t = int(rng.integers(lo, hi + 1))
            hist = slice(t-self.cfg.n_obs_steps+1, t+1)
            actions = slice(t-self.cfg.n_obs_steps+1, t-self.cfg.n_obs_steps+1+self.cfg.horizon)
            batch = {'observation.state': torch.tensor(d['state'][hist]).unsqueeze(0),
                     'action': torch.tensor(d['action'][actions]).unsqueeze(0)}
            for key, source in [('observation.images.agentview', 'image'), ('observation.images.wrist', 'wrist_image')]:
                if key in self.cfg.input_features:
                    batch[key] = torch.tensor(d[source][hist]).permute(0, 3, 1, 2).unsqueeze(0).float()/255
            batch['action_is_pad'] = torch.zeros((1, self.cfg.horizon), dtype=torch.bool)
            assert all(k.startswith('observation.') or k in ('action', 'action_is_pad') for k in batch)
            self.last_sample = dict(file=m['file'], group=m.get('group'), current_step=t, kind=kind)
            return batch, kind
