"""Audit complete windows, substep labels, checkpoint videos and matched budgets."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/round1'));a=p.parse_args();out=a.output.resolve()
manifest=[]
for path in sorted((out/'data').glob('*.json')):
    m=json.loads(path.read_text())
    if m['accepted']:
        assert m['stats']['success'] and not m['safety']['collision'] and not m['safety']['height']
        assert m['frames']-m['supervised_start']>=56
        with np.load(path.parent/m['file'],allow_pickle=False) as d:
            assert np.isfinite(d['action']).all() and np.isfinite(d['state']).all()
            for key in ['image','wrist_image']:assert (d[key].std(axis=(1,2,3))>1).all()
        if m['kind']=='recovery':
            assert .002<=m['collision_margin']<=.015
            assert m['source']=='expert_carry_executed_xy_perturbation'
            assert m['supervised_start']>0 and m['margin_improved']
    manifest.append(m)
(out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
training={}
videos=[]
for name in ['expert_round','more_expert','recovery_round']:
    path=out/name/'training.json'
    if not path.exists():continue
    log=json.loads(path.read_text());training[name]=log
    assert log['loss']=='standard_diffusion_MSE'
    assert log['trainable_parameters']==273977798
    for checkpoint in sorted((out/name/'checkpoints').glob('*')):
        config=json.loads((checkpoint/'config.json').read_text())
        assert config['type']=='diffusion' and not any(k.startswith('safety_') for k in config)
        video=checkpoint/'evaluation/rollout.mp4'
        assert video.exists(),f'Checkpoint is missing a video: {checkpoint}'
        proc=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf','scale=1:1','-f','rawvideo','-pix_fmt','rgb24','-'],capture_output=True,check=True)
        means=np.frombuffer(proc.stdout,dtype=np.uint8).reshape(-1,3).mean(axis=1)
        assert len(means)>0 and (means>10).all()
        videos.append(dict(checkpoint=str(checkpoint),video=str(video),frames=len(means),minimum_rgb_mean=float(means.min())))
matched=None
if 'more_expert' in training and 'recovery_round' in training:
    b=training['more_expert'];c=training['recovery_round']
    assert b['initial']==c['initial']
    assert len(b['updates'])==len(c['updates'])
    for n,r in zip(b['updates'],c['updates'],strict=True):
        assert n['samples'][1]==r['samples'][1],(n['step'],'Shared replay differs')
        assert n['samples'][0]['group']=='more' and r['samples'][0]['group']=='recovery'
    matched=dict(updates=len(b['updates']),shared_windows_identical=True,initial_checkpoint_identical=True,
                 additional_samples_per_update=1,shared_nominal_samples_per_update=1)
paired=None
if (out/'comparison.json').exists():
    results=json.loads((out/'comparison.json').read_text())
    for suite in ['held_out','near_collision']:
        eligible=[r for r in results if r['suite']==suite and not r.get('excluded')]
        for seed in {r['seed'] for r in eligible}:
            rows=[r for r in eligible if r['seed']==seed]
            assert {r['arm'] for r in rows}=={'A','B','C'}
            assert all(r['initial']==rows[0]['initial'] for r in rows)
    assert all(not r.get('runtime_filter',False) for r in results)
    paired=dict(eligible_cases=sum(not r.get('excluded',False) for r in results),initial_measurements_identical=True,runtime_filters=False)
result=dict(paired_evaluation=paired,accepted_trajectories=sum(m['accepted'] for m in manifest),rejected_trajectories=sum(not m['accepted'] for m in manifest),
            checkpoint_videos=videos,matched_comparison=matched)
(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
