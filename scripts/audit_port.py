"""Audit local task files/assets/defaults and a complete expert rollout."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recovery_experiment.runtime import bootstrap
root=bootstrap()
import numpy as np
from sim.configs import EnvConfig,ExpertConfig
from sim.env import SafeCubeEnv
from sim.expert import ScriptedExpert

p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=Path.home()/'Documents/LeRobot-SafetyCubes')
p.add_argument('--seed',type=int,default=9400);p.add_argument('--max-steps',type=int,default=400)
p.add_argument('--output',type=Path,default=Path('artifacts/expert_preview/port_audit.json'))
a=p.parse_args();ref=a.reference.expanduser().resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(reference=str(ref),reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ref,text=True).strip(),modules={},assets={})
for name in ['env.py','scene.py','expert.py','randomize.py','_draccus_compat.py']:
    same=(root/'sim'/name).read_bytes()==(ref/'sim'/name).read_bytes()
    report['modules'][name]=dict(identical=same,local_sha256=sha(root/'sim'/name),reference_sha256=sha(ref/'sim'/name))
    assert same,f'Task module differs: {name}'
local_assets={str(x.relative_to(root/'sim/assets')) for x in (root/'sim/assets').rglob('*') if x.is_file()}
ref_assets={str(x.relative_to(ref/'sim/assets')) for x in (ref/'sim/assets').rglob('*') if x.is_file()}
assert local_assets==ref_assets,dict(missing=sorted(ref_assets-local_assets),extra=sorted(local_assets-ref_assets))
for name in sorted(ref_assets):
    left=root/'sim/assets'/name;right=ref/'sim/assets'/name
    same=left.read_bytes()==right.read_bytes()
    semantic=same or (left.suffix=='.urdf' and ET.canonicalize(left.read_text(),strip_text=True)==ET.canonicalize(right.read_text(),strip_text=True))
    report['assets'][name]=dict(byte_identical=same,semantic_identical=semantic,local_sha256=sha(left),reference_sha256=sha(right))
    assert semantic,f'Robot asset differs: {name}'
alias='_port_audit_reference'
spec=importlib.util.spec_from_file_location(alias,ref/'sim/__init__.py',submodule_search_locations=[str(ref/'sim')])
module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module)
ref_expert=importlib.import_module(alias+'.expert')
local_cfg=EnvConfig(max_episode_steps=a.max_steps);ref_cfg=module.EnvConfig(max_episode_steps=a.max_steps)
left=asdict(local_cfg);right=asdict(ref_cfg);left['scene'].pop('mjcf_path');right['scene'].pop('mjcf_path')
assert left==right,'Task defaults differ'
assert asdict(ExpertConfig())==asdict(module.ExpertConfig()),'Expert defaults differ'
ref_cfg.scene.mjcf_path=str(ref/'sim/assets/so101/so101_new_calib.urdf')
report['defaults_identical_except_asset_path']=True
report['configuration_sha256']=dict(local=sha(root/'sim/configs.py'),reference=sha(ref/'sim/configs.py'))
local=SafeCubeEnv(local_cfg);original=module.SafeCubeEnv(ref_cfg)
try:
    obs,info=local.reset(seed=a.seed);ref_obs,ref_info=original.reset(seed=a.seed)
    expert=ScriptedExpert(local,ExpertConfig());other=ref_expert.ScriptedExpert(original,module.ExpertConfig())
    qdiff=adiff=0.;pixel_sum=0;pixel_count=0
    for step in range(a.max_steps):
        action=expert.act(info);reference_action=other.act(ref_info)
        adiff=max(adiff,float(np.max(np.abs(action-reference_action))))
        qdiff=max(qdiff,float(np.max(np.abs(local.data.qpos-original.data.qpos))))
        for key in ['image','wrist_image']:
            pixel_sum+=int(np.abs(obs[key].astype(np.int16)-ref_obs[key].astype(np.int16)).sum())
            pixel_count+=obs[key].size
        np.testing.assert_allclose(action,reference_action,rtol=0,atol=1e-10)
        np.testing.assert_allclose(local.data.qpos,original.data.qpos,rtol=0,atol=1e-10)
        obs,_,done,timeout,info=local.step(action)
        ref_obs,_,ref_done,ref_timeout,ref_info=original.step(reference_action)
        assert (done,timeout)==(ref_done,ref_timeout),'Termination differs'
        if done or timeout:break
    assert info['stats']==ref_info['stats'],'Outcome/measurements differ'
    assert info['stats']['success'],'Expert did not complete the parity case'
    report['expert_rollout']=dict(seed=a.seed,control_steps=step+1,max_qpos_difference=qdiff,max_action_difference=adiff,
        camera_pixel_mean_absolute_difference=pixel_sum/pixel_count,local_stats=info['stats'],reference_stats=ref_info['stats'])
finally:local.close();original.close()
report['passed']=True
report['scope']='Complete task runtime, expert, configuration and every asset; unrelated policy/safety-loss training scaffolding is not part of the task port.'
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['assets','modules']},indent=2))
