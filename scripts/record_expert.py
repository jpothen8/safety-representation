"""Record only the local scripted expert, with main and wrist cameras."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recovery_experiment.runtime import bootstrap
bootstrap()
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from sim.configs import EnvConfig,ExpertConfig
from sim.expert import ScriptedExpert
from recovery_experiment.safety import MeasuredEnv

p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,default=9400)
p.add_argument('--output',type=Path,default=Path('artifacts/expert_preview/expert_seed9400.mp4'))
p.add_argument('--size',type=int,default=480)
p.add_argument('--fps',type=int,default=30)
p.add_argument('--max-steps',type=int,default=400)
a=p.parse_args();a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
cfg=EnvConfig(max_episode_steps=a.max_steps);cfg.scene.image_size=(a.size,a.size)
env=MeasuredEnv(cfg);proc=None
try:
    obs,info=env.reset(seed=a.seed);env.begin_measurement()
    expert=ScriptedExpert(env,ExpertConfig())
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,18) if Path(font_path).exists() else ImageFont.load_default()
    width=2*a.size;height=a.size+92
    tmp=a.output.with_suffix('.tmp.mp4')
    proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(a.fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(tmp)],stdin=subprocess.PIPE)
    count=0;minimum_image_std=float('inf');last=None
    for step in range(a.max_steps):
        action=expert.act(info)
        minimum_image_std=min(minimum_image_std,float(obs['image'].std()),float(obs['wrist_image'].std()))
        if minimum_image_std<=1:raise RuntimeError('Black/constant camera frame detected')
        canvas=Image.new('RGB',(width,height),(22,25,30));canvas.paste(Image.fromarray(obs['image']),(0,92));canvas.paste(Image.fromarray(obs['wrist_image']),(a.size,92))
        draw=ImageDraw.Draw(canvas)
        measure=env.measure();margin=measure['height_margin']
        draw.text((12,7),f'EXPERT ONLY | seed {a.seed} | {cfg.scene.n_red_cubes} red obstacles | {expert._phase.name}',font=font,fill='white')
        draw.text((12,35),'Agent view',font=font,fill='white');draw.text((a.size+12,35),'Wrist view',font=font,fill='white')
        draw.text((12,63),f'Step {step} | collision: {env._stats.red_contact} | ceiling: {env._stats.ceiling_violation}',font=font,fill='white')
        draw.text((a.size+12,63),f'Playback {cfg.sim_substeps*env.model.opt.timestep*a.fps:.2f}x simulator time',font=font,fill='white')
        last=np.asarray(canvas);proc.stdin.write(last.tobytes());count+=1
        obs,_,done,truncated,info=env.step(action)
        if done or truncated:break
    # Display the actual terminal state. The final one second is a labeled pause.
    canvas=Image.new('RGB',(width,height),(22,25,30));canvas.paste(Image.fromarray(obs['image']),(0,92));canvas.paste(Image.fromarray(obs['wrist_image']),(a.size,92))
    draw=ImageDraw.Draw(canvas);safety=env.summary()
    result='SUCCESS' if info['stats']['success'] else 'FAILED / TIMEOUT'
    draw.text((12,10),f'{result} | final state (paused)',font=font,fill='white')
    draw.text((12,40),f"Collision: {safety['collision']} | ceiling: {safety['height']} | policy: scripted expert only",font=font,fill='white')
    last=np.asarray(canvas)
    for _ in range(a.fps):proc.stdin.write(last.tobytes());count+=1
    proc.stdin.close()
    if proc.wait():raise RuntimeError('ffmpeg failed')
    tmp.replace(a.output)
    report=dict(video=str(a.output),seed=a.seed,stats=info['stats'],safety=safety,frames=count,fps=a.fps,
        cameras=['agentview','wrist'],render_size=[a.size,a.size],minimum_raw_image_std=minimum_image_std,
        simulator_timestep=float(env.model.opt.timestep),control_substeps=cfg.sim_substeps,
        playback_speed=float(cfg.sim_substeps*env.model.opt.timestep*a.fps),policy='local ScriptedExpert',
        environment_module=__import__('sim.env',fromlist=['__file__']).__file__,expert_module=__import__('sim.expert',fromlist=['__file__']).__file__)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not info['stats']['success'] or safety['collision'] or safety['height']:raise RuntimeError('Expert demonstration did not pass task and safety checks')
finally:
    if proc is not None and proc.poll() is None:
        if proc.stdin and not proc.stdin.closed:proc.stdin.close()
        proc.wait()
    env.close()
