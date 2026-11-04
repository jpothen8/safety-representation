"""WandB runs plus durable local event logs; never claim a failed upload worked."""
from __future__ import annotations
import hashlib
import json
import os
import socket
import tempfile
from pathlib import Path


def require_local_socket():
    # AF_UNIX paths are limited to roughly 108 bytes. Experiment output paths
    # can exceed that, so keep the probe in a short temporary directory.
    with tempfile.TemporaryDirectory(prefix='wb-probe-') as directory:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as probe:
            probe.bind(str(Path(directory)/'s'))


class Tracker:
    def __init__(self, directory, name, config):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root/'tracking_events.jsonl'
        self.run = None
        self.wandb = None
        settings = config.get('wandb', {})
        run_id = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        status = dict(requested_mode=settings.get('mode','online'), run_id=run_id, active=False)
        try:
            if status['requested_mode'] == 'disabled':
                raise RuntimeError('WandB disabled explicitly by configuration')
            # wandb-core needs a local socket even in offline mode. Diagnose this
            # before launching a service that cannot start in this sandbox.
            require_local_socket()
            for key in ['WANDB_DIR','WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR']:
                location = self.root/'wandb_runtime'/key.lower()
                location.mkdir(parents=True,exist_ok=True)
                os.environ[key] = str(location)
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=settings.get('project','safe-cube'), entity=settings.get('entity'),
                group=config.get('experiment_name','collision-recovery-only'), name=name, job_type=name,
                tags=['imitation-only','no-safety-loss'], id=run_id,
                resume='allow', mode=status['requested_mode'], config=config,
                settings=wandb.Settings(init_timeout=30))
            status.update(active=True, url=self.run.url, mode=self.run.settings.mode)
        except Exception as error:
            status['error'] = f'{type(error).__name__}: {error}'
            if not settings.get('allow_local_fallback',True):
                (self.root/'tracking_status.json').write_text(json.dumps(status,indent=2)+'\n')
                raise
        (self.root/'tracking_status.json').write_text(json.dumps(status,indent=2)+'\n')
        print('tracking', name, status, flush=True)

    def log(self, values, step):
        event = dict(step=step, values=values)
        with self.events.open('a') as f:
            f.write(json.dumps(event,allow_nan=False)+'\n')
        if self.run:
            self.run.log(values,step=step)

    def video(self, path, step, key='rollouts/checkpoint'):
        path = Path(path).resolve()
        with self.events.open('a') as f:
            f.write(json.dumps(dict(step=step, video=str(path), key=key))+'\n')
        if self.run:
            self.run.log({key:self.wandb.Video(str(path),format='mp4')},step=step)

    def finish(self):
        if self.run:
            self.run.finish()
