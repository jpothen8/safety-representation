"""Local task, pure diffusion imitation, full-task BC then targeted collision recovery.

Stage E trains every selected policy module on complete expert demonstrations.
N and R then initialize from exactly E's last checkpoint with equal budgets.
N receives NEW complete expert trajectories; R mixes nominal with validated
near-collision expert suffixes. No safety-loss implementation is imported.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import subprocess
import numpy as np
import torch
from .runtime import bootstrap


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['collect-expert','train-expert','collect-recovery','compare','run','sync-tracking'])
    parser.add_argument('--config',default='configs/round1_cpu.json')
    parser.add_argument('--output',default='artifacts/round1')
    args=parser.parse_args()
    c=json.loads(Path(args.config).resolve().read_text())
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    root=bootstrap()
    from sim.configs import EnvConfig,ExpertConfig
    from sim.expert import ScriptedExpert
    from .safety import MeasuredEnv
    from .policy import Policy
    from .data import frame,save_trajectory,Trajectories
    from .tracking import Tracker
    torch.set_num_threads(c['threads'])
    if c['device'].startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use the explicitly budgeted CPU pilot or run from the native GPU terminal')
    pools=[set(c[k]) for k in ['base_expert_seeds','more_expert_seeds','validation_seeds','eval_seeds']]
    assert all(not pools[i]&pools[j] for i in range(len(pools)) for j in range(i)), 'Initial-condition split leakage'
    assert set(c['recovery_seeds']) <= pools[0], 'Recovery parent must belong to the training split'
    assert set(c.get('boundary_eval_seeds',c['eval_seeds'])) <= pools[3], 'Boundary test parents must be held out'
    n_additional=round(c['recovery_ratio']*c['batch_size'])
    if not 0<n_additional<c['batch_size']:
        raise ValueError('Choose a recovery ratio and batch size with both nominal and additional examples per update')
    write(out/'config.json',c)
    write(out/'environment.json',dict(env=asdict(EnvConfig()),expert=asdict(ExpertConfig()),port=json.loads((root/'sim/PORT_PROVENANCE.json').read_text())))
    base=root/c['checkpoint']

    def new_env():
        cfg=EnvConfig(max_episode_steps=c['max_episode_steps'],terminate_on_red_contact=True)
        cfg.scene.ee_height_ceiling=c['ceiling']
        return MeasuredEnv(cfg,c['contact_tolerance'])

    def save_video(path,frames):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if not frames:raise ValueError('Cannot export an empty rollout')
        h,w=frames[0].shape[:2]
        tmp=path.with_suffix('.tmp.mp4')
        proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
            '-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(tmp)],stdin=subprocess.PIPE)
        for image in frames:proc.stdin.write(image.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError('ffmpeg failed')
        tmp.replace(path)

    def execute(env,obs,info,actor,prefix=None):
        frames=list(prefix or []);video=[]
        for t in range(c['max_episode_steps']):
            action=actor(obs,info)
            frames.append(frame(obs,action,info))
            if t%3==0:video.append(obs['image'].copy())
            obs,_,done,timeout,info=env.step(action)
            if done or timeout:break
        return frames,video,info['stats']

    def valid(stats,safety):
        return bool(stats['success'] and not stats['fly_over'] and not safety['collision'] and not safety['height']
                    and not safety['reset_collision'] and not safety['reset_height'])

    def expert_collect():
        for group,seeds,split in [('base',c['base_expert_seeds'],'train'),('more',c['more_expert_seeds'],'train'),
                                 ('validation',c['validation_seeds'],'validation')]:
            for seed in seeds:
                name=f'{group}_{seed}'
                if (out/'data'/f'{name}.json').exists():continue
                env=new_env()
                try:
                    obs,info=env.reset(seed=seed);env.begin_measurement()
                    expert=ScriptedExpert(env,ExpertConfig())
                    frames,video,stats=execute(env,obs,info,lambda o,i:expert.act(i))
                    safety=env.summary()
                    meta=save_trajectory(out/'data',name,frames,dict(kind='nominal',group=group,split=split,seed=seed,
                        source='full_task_expert',supervised_start=0,accepted=valid(stats,safety),stats=stats,safety=safety))
                    write(out/'traces'/f'{name}.json',env.samples)
                    save_video(out/'expert_videos'/f'{name}.mp4',video)
                    print('expert trajectory',name,meta['accepted'],len(frames),flush=True)
                finally:env.close()

    def dataset(groups):
        data=Trajectories(out/'data','train',type('Temporal',(),dict(horizon=56,n_obs_steps=1,input_features={}))())
        # Replaced with the loaded checkpoint config by train(); filter lineage.
        for kind in data.groups:
            data.groups[kind]=[(m,p) for m,p in data.groups[kind] if m.get('group') in groups]
        return data

    def evaluate_checkpoint(checkpoint,directory,seed,initial=None):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        result_path=directory/'metrics.json'
        if result_path.exists() and (directory/'rollout.mp4').exists():return json.loads(result_path.read_text())
        env=new_env()
        try:
            obs,info=env.reset(seed=seed)
            if initial is not None:
                env.monitoring=False
                obs,info=env.restore(initial)
            env.begin_measurement();start=env.measure()
            policy=Policy(checkpoint,c['device']);policy.reset();torch.manual_seed(seed)
            _,video,stats=execute(env,obs,info,lambda o,i:policy.act(o))
            safety=env.summary()
            row=dict(seed=seed,initial=start,success=bool(stats['success']),safe_success=valid(stats,safety),stats=stats,
                     safety=safety,runtime_filter=False,interventions=0,checkpoint=str(checkpoint),
                     latency_median_ms=float(np.median(policy.latencies)*1000),
                     latency_p95_ms=float(np.percentile(policy.latencies,95)*1000))
            save_video(directory/'rollout.mp4',video)
            write(directory/'safety_trace.json',env.samples)
            write(result_path,row)
            print('checkpoint rollout',str(checkpoint),row['success'],safety,flush=True)
            return row
        finally:env.close()

    def train(name,initial,budget,groups,ratio):
        stage=out/name;stage.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(c['training_seed']);rng=np.random.default_rng(c['training_seed'])
        policy=Policy(initial,c['device']);ds=dataset(groups);ds.cfg=policy.cfg
        common_ds=dataset(['base']);common_ds.cfg=policy.cfg
        more_ds=dataset(['more']);more_ds.cfg=policy.cfg
        assert policy.cfg.n_obs_steps==1 and policy.cfg.horizon==56,'Use checkpoint-compatible temporal metadata'
        for key,p in policy.model.named_parameters():p.requires_grad_(any(key.startswith(prefix) for prefix in c['trainable_modules']))
        parameters=[p for p in policy.model.parameters() if p.requires_grad]
        if not parameters:raise ValueError('No trainable modules')
        if not ds.groups['nominal'] or (ratio and not ds.groups['recovery']):raise ValueError(f'Missing required trajectory group for {name}')
        optimizer=torch.optim.AdamW(parameters,lr=c['lr'],betas=(.95,.999),weight_decay=1e-6)
        policy.model.eval() # Freeze BatchNorm statistics consistently across arms; gradients remain enabled.
        tracker=Tracker(stage,name,c)
        signature=hashlib.sha256(json.dumps(dict(config=c,name=name,initial=str(initial),groups=groups),sort_keys=True).encode()).hexdigest()
        resume=stage/'resume.pt';start=0;log=[]
        if resume.exists():
            state=torch.load(resume,map_location=c['device'],weights_only=False)
            if state['signature']!=signature:raise ValueError('Resume config mismatch')
            policy.model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['torch_rng'].cpu());rng.bit_generator.state=state['numpy_rng']
            if torch.cuda.is_available() and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
            start=state['update'];log=state['log']
        try:
            for update in range(start,budget):
                optimizer.zero_grad(set_to_none=True);losses=[];sources=[];samples=[]
                for sample in range(c['batch_size']):
                    fraction=float(sample<round(ratio*c['batch_size']))
                    selected=ds
                    if name=='more_expert':
                        selected=more_ds if sample<round(c['recovery_ratio']*c['batch_size']) else common_ds
                    sample_rng=rng
                    if name in ['more_expert','recovery_round'] and sample>=round(c['recovery_ratio']*c['batch_size']):
                        # Stateless common replay stream: identical base windows
                        # in B/C despite different augmentation pool lengths.
                        sample_rng=np.random.default_rng(c['training_seed']*100000+update*c['batch_size']+sample)
                    batch,kind=selected.sample(sample_rng,fraction)
                    samples.append(selected.last_sample)
                    loss=policy.imitation_loss(batch)
                    (loss/c['batch_size']).backward();losses.append(float(loss.detach()));sources.append(kind)
                grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradients')
                optimizer.step();step=update+1
                entry=dict(step=step,loss=float(np.mean(losses)),grad_norm=float(grad),sources=sources,samples=samples)
                log.append(entry);tracker.log({'train/loss':entry['loss'],'train/grad_norm':entry['grad_norm'],
                    'train/recovery_samples':sources.count('recovery'),'train/nominal_samples':sources.count('nominal'),
                    'train/additional_samples':sum(s['group'] in ['more','recovery'] for s in samples)},step)
                print(name,entry,flush=True)
                if step%c['save_freq']==0 or step==budget:
                    ckpt=stage/'checkpoints'/f'{step:06d}';ckpt.mkdir(parents=True,exist_ok=True)
                    policy.model.save_pretrained(ckpt);policy.pre.save_pretrained(ckpt);policy.post.save_pretrained(ckpt)
                    tmp=resume.with_suffix('.tmp')
                    torch.save(dict(signature=signature,model=policy.model.state_dict(),optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,update=step,log=log),tmp)
                    tmp.replace(resume)
                    write(stage/'training.json',dict(initial=str(initial),loss='standard_diffusion_MSE',trainable_modules=c['trainable_modules'],
                        trainable_parameters=sum(p.numel() for p in parameters),updates=log,groups=groups,recovery_ratio=ratio))
                    # Evaluate the SAVED checkpoint, not the live optimizer model.
                    # Isolate video inference RNG so it cannot perturb training.
                    rng_state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    row=evaluate_checkpoint(ckpt,ckpt/'evaluation',c['video_seed'])
                    torch.set_rng_state(rng_state)
                    if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
                    tracker.log({'eval/success':int(row['success']),'eval/safe_success':int(row['safe_success']),
                        'eval/collision':int(row['safety']['introduced_collision']),'eval/ceiling':int(row['safety']['introduced_height'])},step)
                    tracker.video(ckpt/'evaluation/rollout.mp4',step)
            # A restart after weights were saved but before video export must
            # regenerate the missing checkpoint video, even with no updates left.
            for ckpt in sorted((stage/'checkpoints').glob('*')):
                if not (ckpt/'evaluation/rollout.mp4').exists():
                    evaluate_checkpoint(ckpt,ckpt/'evaluation',c['video_seed'])
                    tracker.video(ckpt/'evaluation/rollout.mp4',int(ckpt.name))
        finally:tracker.finish()
        return stage/'checkpoints'/f'{budget:06d}'

    def recoveries():
        """Physically approach red-cube collision boundaries, then expert escape.

        No height perturbation or height-only candidate gate. The ported expert
        supplies a coherent completion; all perturbation actions are excluded.
        """
        for seed in c['recovery_seeds']:
            if (out/'data'/f'recovery_{seed}.json').exists():continue
            env=new_env()
            try:
                obs,info=env.reset(seed=seed);env.begin_measurement()
                expert=ScriptedExpert(env,ExpertConfig());prefix=[];anchors=[]
                for t in range(c['max_episode_steps']):
                    action=expert.act(info)
                    if info['grasped'] and env.blue_cube_position()[2]>.023 and t%10==0 and np.linalg.norm(info['blue_cube_pos'][:2]-info['goal_pos'][:2])>.12:
                        anchors.append((env.snapshot(),list(prefix),copy.deepcopy({k:v for k,v in expert.__dict__.items() if k!='env'})))
                    prefix.append(frame(obs,action,info))
                    obs,_,done,timeout,info=env.step(action)
                    if done or timeout:break
                attempts=[];saved=False
                for attempt,(snap,history,expert_state) in enumerate(anchors[:5]):
                    env.monitoring=False;obs,info=env.restore(snap);env.begin_measurement()
                    history=list(history);perturb=[];collision=False;near=False
                    for k in range(c['max_perturb_steps']):
                        clearance,pair=env.collision_clearance()
                        if c['minimum_collision_margin']<=clearance<=c['near_collision_margin']:
                            near=True;break
                        if clearance<c['minimum_collision_margin']:break
                        red_id=env._red_body_ids[int(np.argmin(np.linalg.norm(info['cube_positions'][:,:2]-info['blue_cube_pos'][:2],axis=1)))]
                        direction=env.data.xpos[red_id,:2]-env.blue_cube_position()[:2]
                        direction/=max(float(np.linalg.norm(direction)),1e-8)
                        target=env.grasp_position().copy();target[:2]+=direction*c['perturb_xy_step']
                        action=np.r_[env.ik_solve(target,damping=.05,pos_tol=.0005,target_grasp_site=True),env.gripper_qpos()]
                        for _ in range(c['perturb_settle_steps']):
                            history.append(frame(obs,action,info));perturb.append(action.tolist())
                            obs,_,done,timeout,info=env.step(action)
                            if done or timeout:break
                        if env.summary()['collision'] or env.summary()['height'] or done or timeout:
                            collision=True;break
                    start=env.measure();clearance,pair=env.collision_clearance()
                    if not near or collision:
                        attempts.append(dict(attempt=attempt,accepted=False,reason='could_not_reach_safe_collision_band',
                                             clearance=clearance,safety=env.summary(),perturb_actions=perturb));continue
                    recovery_snap=env.snapshot();env.begin_measurement()
                    expert=ScriptedExpert(env,ExpertConfig())
                    for key,value in expert_state.items():setattr(expert,key,value)
                    expert.replan_carry_from_current(info['cube_positions'])
                    suffix_start=len(history)
                    frames,video,stats=execute(env,obs,info,lambda o,i:expert.act(i),history)
                    safety=env.summary()
                    # Require a larger actual geometry margin later in the
                    # expert continuation, not a point/TCP clearance proxy.
                    finish_margin,_=env.collision_clearance()
                    good=valid(stats,safety) and len(frames)-suffix_start>=c['minimum_recovery_steps'] and finish_margin>clearance+.001
                    attempts.append(dict(attempt=attempt,accepted=good,initial_clearance=clearance,final_clearance=finish_margin,
                                         suffix_steps=len(frames)-suffix_start,safety=safety,stats=stats))
                    if not good:continue
                    meta=save_trajectory(out/'data',f'recovery_{seed}',frames,dict(kind='recovery',group='recovery',split='train',seed=seed,
                        source='expert_carry_executed_xy_perturbation',accepted=True,supervised_start=suffix_start,stats=stats,safety=safety,
                        near_boundary=True,collision_margin=clearance,closest_geom_pair=pair,margin_improved=True,
                        perturb_actions=perturb,start=start,anchor_attempt=attempt))
                    np.savez_compressed(out/'snapshots'/f'recovery_{seed}.npz',**recovery_snap)
                    write(out/'traces'/f'recovery_{seed}.json',env.samples)
                    save_video(out/'recovery_videos'/f'{seed}.mp4',video)
                    saved=True;print('near-collision recovery',seed,clearance,meta['frames']-suffix_start,flush=True);break
                write(out/'recovery_attempts'/f'{seed}.json',dict(accepted=saved,attempts=attempts))
                if not saved:print('no validated collision recovery for seed',seed,flush=True)
            finally:env.close()

    def compare():
        expert_ckpt=out/'expert_round/checkpoints'/f"{c['base_updates']:06d}"
        if not expert_ckpt.exists():raise FileNotFoundError('Run train-expert before the comparison')
        nominal=train('more_expert',expert_ckpt,c['continuation_updates'],['base','more'],0.)
        recovery=train('recovery_round',expert_ckpt,c['continuation_updates'],['base','recovery'],c['recovery_ratio'])
        results=[]
        for arm,path in [('A',expert_ckpt),('B',nominal),('C',recovery)]:
            for seed in c['eval_seeds']:
                # This seed is already rendered at the checkpoint: reuse the
                # exact raw-policy outcome instead of generating a second one.
                directory=path/'evaluation' if seed==c['video_seed'] else out/'evaluation'/f'{arm}_nominal_{seed}'
                row=evaluate_checkpoint(path,directory,seed)
                results.append(dict(row,arm=arm,suite='held_out'))
            # Separate evaluation parents from every training/recovery seed.
            for seed in c.get('boundary_eval_seeds',c['eval_seeds']):
                initial=make_evaluation_boundary(seed)
                if initial is not None:
                    row=evaluate_checkpoint(path,out/'evaluation'/f'{arm}_near_collision_{seed}',seed,initial)
                    results.append(dict(row,arm=arm,suite='near_collision'))
                else:
                    results.append(dict(arm=arm,seed=seed,suite='near_collision',excluded=True,reason='boundary_setup_failed'))
            write(out/'comparison.partial.json',results)
        write(out/'comparison.json',results)
        lines=['| Arm | Suite | N | Success | Safe success | Collision | Ceiling |','|---|---|---:|---:|---:|---:|---:|']
        for arm in ['A','B','C']:
            for suite in ['held_out','near_collision']:
                rows=[r for r in results if r['arm']==arm and r['suite']==suite and not r.get('excluded')]
                if not rows:continue
                n=len(rows);values=[sum(r[k] for r in rows)/n for k in ['success','safe_success']]
                values += [sum(r['safety'][k] for r in rows)/n for k in ['introduced_collision','introduced_height']]
                lines.append(f'| {arm} | {suite} | {n} | '+' | '.join(f'{v:.1%}' for v in values)+' |')
        (out/'comparison.md').write_text('\n'.join(lines)+'\n')
        log_comparison(results)

    def log_comparison(results):
        location=out/'comparison_tracking'
        digest=hashlib.sha256(json.dumps(results,sort_keys=True).encode()).hexdigest()
        marker=location/'recorded.json'
        if marker.exists() and json.loads(marker.read_text())['digest']==digest:return
        tracker=Tracker(location,'paired_comparison',c)
        try:
            for index,row in enumerate(results):
                prefix=f"test/{row['arm']}/{row['suite']}"
                if row.get('excluded'):
                    tracker.log({prefix+'/excluded':1},index);continue
                tracker.log({prefix+'/task_success':int(row['success']),prefix+'/safe_success':int(row['safe_success']),
                    prefix+'/collision':int(row['safety']['introduced_collision']),prefix+'/ceiling':int(row['safety']['introduced_height']),
                    prefix+'/episodes':1},index)
                if row['suite']=='held_out':
                    video=Path(row['checkpoint'])/'evaluation/rollout.mp4' if row['seed']==c['video_seed'] else out/'evaluation'/f"{row['arm']}_nominal_{row['seed']}"/'rollout.mp4'
                else:video=out/'evaluation'/f"{row['arm']}_near_collision_{row['seed']}"/'rollout.mp4'
                tracker.video(video,index,key=prefix+'/rollout')
            write(marker,dict(digest=digest,rows=len(results)))
        finally:tracker.finish()

    def make_evaluation_boundary(seed):
        path=out/'evaluation_initials'/f'{seed}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}
        if (out/'evaluation_initials'/f'{seed}_excluded.json').exists():return None
        # Use the same collision-boundary generator on a held-out parent,
        # never import a training anchor into the evaluation split.
        eval_out=out/'boundary_generation'
        # recoveries closes over out. A separate invocation provides isolation
        # and writes its own provenance; its data is never used by training.
        config=dict(c,recovery_seeds=[seed],base_expert_seeds=[seed],more_expert_seeds=[],validation_seeds=[],eval_seeds=[])
        config_path=out/'boundary_config.json';write(config_path,config)
        subprocess.run([__import__('sys').executable,'-m','recovery_experiment.rounds','collect-recovery',
                        '--config',str(config_path),'--output',str(eval_out)],check=True,cwd=root)
        source=eval_out/'snapshots'/f'recovery_{seed}.npz'
        if not source.exists():
            write(out/'evaluation_initials'/f'{seed}_excluded.json',dict(reason='no_validated_boundary_setup'))
            return None
        import shutil
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
        with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}

    if args.stage=='sync-tracking':
        for stage in ['expert_round','more_expert','recovery_round','comparison_tracking']:
            location=out/stage;events=location/'tracking_events.jsonl'
            if not events.exists():continue
            tracker=Tracker(location,stage,dict(c,wandb={**c.get('wandb',{}),'allow_local_fallback':False}))
            try:
                for line in events.read_text().splitlines():
                    e=json.loads(line)
                    if 'values' in e:tracker.run.log(e['values'],step=e['step'])
                    elif 'video' in e:tracker.run.log({e['key']:tracker.wandb.Video(e['video'],format='mp4')},step=e['step'])
            finally:tracker.finish()
        return
    (out/'snapshots').mkdir(exist_ok=True)
    if args.stage in ['collect-expert','run']:expert_collect()
    if args.stage in ['train-expert','run']:train('expert_round',base,c['base_updates'],['base'],0.)
    if args.stage in ['collect-recovery','run']:recoveries()
    if args.stage in ['compare','run']:compare()


if __name__=='__main__':main()
