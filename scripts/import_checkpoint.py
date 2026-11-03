"""Import λ=0 task-pretrained weights as a plain DiffusionPolicy checkpoint."""
import argparse
import hashlib
import json
import shutil
from dataclasses import fields
from pathlib import Path
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--destination',type=Path,default=Path('checkpoints/base'))
a=p.parse_args()
config=json.loads((a.source/'config.json').read_text())
if config.get('safety_weight',0)!=0:
    raise ValueError('Select a checkpoint trained with safety_weight=0 for this imitation-only experiment')
a.destination.mkdir(parents=True,exist_ok=True)
allowed={f.name for f in fields(DiffusionConfig)}
plain={k:v for k,v in config.items() if k in allowed}
plain.update(type='diffusion',pretrained_backbone_weights=None)
for f in a.source.iterdir():
    if f.name=='model.safetensors' or f.name.startswith('policy_'):
        shutil.copyfile(f,a.destination/f.name)
(a.destination/'config.json').write_text(json.dumps(plain,indent=2)+'\n')
h=hashlib.sha256()
with (a.source/'model.safetensors').open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
(a.destination/'provenance.json').write_text(json.dumps(dict(source=str(a.source.resolve()),model_sha256=h.hexdigest(),
    source_type=config['type'],source_safety_weight=config.get('safety_weight',0),
    note='Task-pretrained, not generic diffusion pretraining. Historical expert aggregate may include cleanup demonstrations.'),indent=2)+'\n')
print(a.destination.resolve())
