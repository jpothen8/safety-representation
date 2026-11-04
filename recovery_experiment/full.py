"""Three-seed full experiment, shared data, strict GPU/WandB preflight."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from .runtime import bootstrap


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def prepare(spec_path,output):
    spec=json.loads(Path(spec_path).read_text());out=Path(output).resolve();root=Path(__file__).resolve().parents[1]
    template=json.loads((root/'configs/round1_gpu.json').read_text())
    c=copy.deepcopy(template)
    def seeds(name):
        pool=spec['seed_pools'][name];return list(range(pool['start'],pool['start']+pool['count']))
    pools=[set(seeds(name)) for name in ['base','more','validation','evaluation']]
    if any(pools[i]&pools[j] for i in range(4) for j in range(i)):raise ValueError('Full-experiment initial-condition cohorts overlap')
    if len(set(spec['training_seeds']))!=len(spec['training_seeds']) or not spec['training_seeds']:raise ValueError('Training seeds must be unique and nonempty')
    for group,target in spec['expert_targets'].items():
        if target>len(seeds(group)):raise ValueError(f'Not enough candidate seeds for {group} success target')
    if spec['boundary_cases']>len(seeds('evaluation')):raise ValueError('Not enough held-out boundary parents')
    c.update(experiment_name=spec['name'],base_expert_seeds=seeds('base'),more_expert_seeds=seeds('more'),
        validation_seeds=seeds('validation'),eval_seeds=seeds('evaluation'),
        boundary_eval_seeds=seeds('evaluation')[:spec['boundary_cases']],
        checkpoint_eval_seeds=seeds('validation')[:spec['validation_cases_per_checkpoint']],
        video_seed=seeds('validation')[0],recovery_seeds=seeds('base'),expert_targets=spec['expert_targets'],
        recovery_target=spec['recovery_target'],cache_failed_recovery_attempts=True,
        base_updates=spec['base_updates'],continuation_updates=spec['continuation_updates'],save_freq=spec['save_freq'],
        batch_size=spec['batch_size'],microbatch_size=spec['microbatch_size'],lr=spec['lr'],
        expert_checkpoint_selection='validation_safe_success',
        require_executed_recovery_perturbation=True,minimum_perturbation_margin_drop=.002,validate_rgb=True,data_root=str(out/'shared/data'),data_cache_root=str(out/'shared/cache'),
        evaluation_setup_root=str(out/'shared/evaluation_setups'))
    c['wandb'].update(mode='online',allow_local_fallback=False)
    if spec.get('wandb_entity'):c['wandb']['entity']=spec['wandb_entity']
    fingerprints={}
    for directory in ['recovery_experiment','sim','configs']:
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.urdf','.stl']:
                fingerprints[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    checkpoint=root/c['checkpoint']
    weights=checkpoint/'model.safetensors'
    stats=weights.stat()
    initial_checkpoint=dict(path=str(checkpoint),model_size=stats.st_size,model_mtime_ns=stats.st_mtime_ns,
        config_sha256=hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
        processor_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.glob('policy_*') if p.is_file()})
    signature=hashlib.sha256(json.dumps(dict(spec=spec,source=fingerprints,checkpoint=initial_checkpoint),sort_keys=True).encode()).hexdigest()
    record=out/'manifest.json'
    previous=json.loads(record.read_text()) if record.exists() else {}
    revisions=previous.get('pretraining_revisions',[])
    if previous and previous['signature']!=signature:
        started=(out/'training_started.json').exists() or any(out.glob('seed*/expert_round/resume.pt'))
        if started:raise RuntimeError('Full-experiment source/spec changed after training started; restore the recorded source or use a new output')
        if list((out/'shared/data').glob('*.json')):
            data_files=[name for name in fingerprints if name.startswith('sim/') or name in [
                'recovery_experiment/rounds.py','recovery_experiment/data.py','recovery_experiment/runtime.py']]
            if any(previous.get('source_sha256',{}).get(name)!=fingerprints[name] for name in data_files):
                raise RuntimeError('Task/collector code changed after data collection; use a new output directory')
            for key in ['seed_pools','expert_targets','recovery_target']:
                if previous['spec'][key]!=spec[key]:raise RuntimeError('Data collection specification changed; use a new output')
        revisions.append(dict(previous_signature=previous['signature'],time=time.time(),reason='Preparation changes before model training; task/collector source preserved'))
    created=previous.get('created_at',time.time())
    write(record,dict(spec=spec,signature=signature,source_sha256=fingerprints,initial_checkpoint=initial_checkpoint,created_at=created,pretraining_revisions=revisions))
    configs={}
    for seed in spec['training_seeds']:
        selected=dict(c,training_seed=seed)
        path=out/'configs'/f'seed{seed}.json';write(path,selected);configs[seed]=path
    shared=out/'configs/shared.json';write(shared,dict(c,training_seed=spec['training_seeds'][0]))
    return spec,out,c,configs,shared


def preflight(out,c):
    import torch
    from .tracking import Tracker
    report=dict(time=time.time(),cuda_available=torch.cuda.is_available(),torch=torch.__version__,
                free_disk_bytes=shutil.disk_usage(out).free)
    try:
        if not torch.cuda.is_available():raise RuntimeError('GPU devices are inaccessible in this process; launch from the native server tmux terminal')
        report['gpu']=torch.cuda.get_device_name(0);report['gpu_total_bytes']=torch.cuda.get_device_properties(0).total_memory
        if report['free_disk_bytes']<150*1024**3:raise RuntimeError('Require at least 150 GiB free for shared data and checkpoints')
        root=Path(__file__).resolve().parents[1]
        candidates=[root/'artifacts/expert_preview/port_audit.json',root/'docs/port_audit.json']
        audit=next((p for p in candidates if p.exists()),None)
        if audit is None:raise RuntimeError('Complete scripts/audit_port.py before the full run')
        audited=json.loads(audit.read_text())
        if not audited.get('passed'):raise RuntimeError('Task port audit did not pass')
        for name,entry in audited['modules'].items():
            if hashlib.sha256((root/'sim'/name).read_bytes()).hexdigest()!=entry['local_sha256']:
                raise RuntimeError(f'Task module changed since port audit: {name}')
        for name,entry in audited['assets'].items():
            if hashlib.sha256((root/'sim/assets'/name).read_bytes()).hexdigest()!=entry['local_sha256']:
                raise RuntimeError(f'Task asset changed since port audit: {name}')
        if hashlib.sha256((root/'sim/configs.py').read_bytes()).hexdigest()!=audited['configuration_sha256']['local']:
            raise RuntimeError('Task configuration changed since the port audit')
        report['port_audit']=str(audit)
        tracker=Tracker(out/'preflight_tracking','full_experiment_preflight',c)
        try:
            tracker.log({'preflight/cuda':1,'preflight/disk_gib':report['free_disk_bytes']/1024**3},0)
            report['wandb_url']=tracker.run.url
        finally:tracker.finish()
        report['passed']=True
    except Exception as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}');write(out/'preflight.json',report);raise
    write(out/'preflight.json',report)
    print(json.dumps(report,indent=2),flush=True)


def summarize(out,spec):
    from scipy.stats import binomtest
    all_rows=[]
    for seed in spec['training_seeds']:
        path=out/f'seed{seed}/comparison.json'
        if path.exists():all_rows.extend(dict(row,training_seed=seed) for row in json.loads(path.read_text()))
    write(out/'all_results.json',all_rows)
    by_seed=[];effects=[]
    for training_seed in spec['training_seeds']:
        for suite in ['held_out','near_collision']:
            raw=[r for r in all_rows if r['training_seed']==training_seed and r['suite']==suite]
            rows=[r for r in raw if not r.get('excluded') and not r['safety']['reset_collision'] and not r['safety']['reset_height']]
            arms={a:{r['seed']:r for r in rows if r['arm']==a} for a in ['A','B','C']}
            reference=list(arms['A'].values())
            reference_success=sum(r['success'] for r in reference)/len(reference) if reference else None
            for arm,values in arms.items():
                data=list(values.values())
                if not data:continue
                n=len(data);by_seed.append(dict(training_seed=training_seed,suite=suite,arm=arm,n=n,
                    success=sum(r['success'] for r in data)/n,safe_success=sum(r['safe_success'] for r in data)/n,
                    collision=sum(r['safety']['introduced_collision'] for r in data)/n,
                    ceiling=sum(r['safety']['introduced_height'] for r in data)/n,
                    nominal_retention_pp=100*(sum(r['success'] for r in data)/n-reference_success) if suite=='held_out' and reference_success is not None else None,
                    reset_unsafe=sum(not r.get('excluded') and (r['safety']['reset_collision'] or r['safety']['reset_height']) for r in raw if r['arm']==arm),
                    excluded=sum(r.get('excluded',False) for r in all_rows if r['training_seed']==training_seed and r['suite']==suite and r['arm']==arm)))
            common=sorted(set(arms['B'])&set(arms['C']))
            if not common:continue
            b=np.array([arms['B'][s]['safe_success'] for s in common],dtype=float)
            c=np.array([arms['C'][s]['safe_success'] for s in common],dtype=float)
            delta=c-b;wins=int(np.sum(delta>0));losses=int(np.sum(delta<0))
            rng=np.random.default_rng(training_seed+101)
            boots=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
            effects.append(dict(training_seed=training_seed,suite=suite,n=len(common),
                safe_success_delta_C_minus_B=float(delta.mean()),paired_bootstrap_95ci=np.percentile(boots,[2.5,97.5]).tolist(),
                C_wins=wins,C_losses=losses,mcnemar_exact_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
    write(out/'summary.json',dict(per_training_seed=by_seed,paired_effects=effects,
        note='Seed-level paired results; replicated initial conditions are not pooled as independent evidence. Near-collision safe success is recovery success. Primary rates use clean reset episodes; setup exclusions and unsafe resets are counted separately.'))
    lines=['| Training seed | Suite | Arm | Episodes | Task success | Safe success | Collision | Ceiling |',
           '|---:|---|---|---:|---:|---:|---:|---:|']
    for row in by_seed:
        lines.append(f"| {row['training_seed']} | {row['suite']} | {row['arm']} | {row['n']} | "+' | '.join(f"{row[k]:.1%}" for k in ['success','safe_success','collision','ceiling'])+' |')
    lines+=['','Paired C−B effects, confidence intervals and discordant-case counts are in summary.json. Report nominal retention against A separately for each training seed.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','preflight','collect','run','summarize'])
    parser.add_argument('--spec',default='configs/full_experiment.json');parser.add_argument('--output',default='artifacts/full_experiment')
    a=parser.parse_args();spec_path=Path(a.spec).resolve();out=Path(a.output).resolve();root=bootstrap()
    spec,out,c,configs,shared=prepare(spec_path,out)
    if a.stage=='prepare':print(out/'manifest.json');return
    if a.stage=='summarize':summarize(out,spec);return
    if a.stage in ['preflight','run']:preflight(out,c)
    def run(stage,config,directory):
        subprocess.run([sys.executable,'-u','-m','recovery_experiment.rounds',stage,'--config',str(config),'--output',str(directory)],check=True,cwd=root)
    if a.stage=='preflight':return
    run('collect-expert',shared,out/'shared')
    if a.stage=='run' and not (out/'training_probe.json').exists():
        from .profile import profile_update
        profile_update(root/c['checkpoint'],c['data_root'],c['data_cache_root'],out/'training_probe.json',
                       c['device'],c['batch_size'],c['microbatch_size'])
    from .cache import TrajectoryCache
    cache=TrajectoryCache(c['data_cache_root'])
    for metadata in sorted(Path(c['data_root']).glob('*.json')):
        m=json.loads(metadata.read_text())
        if m.get('accepted'):cache.open(metadata.parent/m['file'])
    if a.stage=='collect':return
    for index,seed in enumerate(spec['training_seeds']):
        if not (out/'training_started.json').exists():
            write(out/'training_started.json',dict(seed=seed,time=time.time(),signature=json.loads((out/'manifest.json').read_text())['signature']))
        run('train-expert',configs[seed],out/f'seed{seed}')
        if index==0:
            run('collect-recovery',shared,out/'shared')
            for metadata in sorted(Path(c['data_root']).glob('*.json')):
                m=json.loads(metadata.read_text())
                if m.get('accepted'):cache.open(metadata.parent/m['file'])
        run('compare',configs[seed],out/f'seed{seed}')
        summarize(out,spec)
    write(out/'completed.json',dict(training_seeds=spec['training_seeds'],finished_at=time.time()))


if __name__=='__main__':main()
