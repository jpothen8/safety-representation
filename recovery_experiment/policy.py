from pathlib import Path
import numpy as np
import json
import time
import torch
import sim._draccus_compat  # Python 3.14 compatibility, no safety objective
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


class Policy:
    def __init__(self, checkpoint, device='cpu'):
        checkpoint = str(Path(checkpoint).resolve())
        from lerobot.configs import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = device
        cfg.pretrained_backbone_weights = None  # checkpoint contains the backbone
        self.model = DiffusionPolicy.from_pretrained(checkpoint, config=cfg)
        self.model.to(device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=checkpoint,
            preprocessor_overrides={'device_processor': {'device': device}})
        self.device = device
        self.cfg = cfg

    def reset(self):
        self.model.reset()
        self.latencies = []

    def planning_boundary(self):
        return len(self.model._queues['action']) == 0

    def act(self, obs):
        start = time.perf_counter()
        batch = {'observation.state': torch.as_tensor(obs['state']).unsqueeze(0)}
        for key, source in [('observation.images.agentview', 'image'), ('observation.images.wrist', 'wrist_image')]:
            if key in self.cfg.input_features:
                batch[key] = torch.from_numpy(obs[source].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        with torch.inference_mode():
            action = self.post(self.model.select_action(self.pre(batch))).squeeze(0).cpu().numpy()
        self.latencies.append(time.perf_counter()-start)
        return action

    def imitation_loss(self, batch):
        # Stock diffusion MSE only. No safety-loss class is imported or instantiated.
        return DiffusionPolicy.forward(self.model, self.pre(batch))[0]
