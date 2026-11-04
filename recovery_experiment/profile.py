"""One real imitation update for the data/batching path; weights are discarded."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
import numpy as np
from .runtime import bootstrap


def profile_update(checkpoint,data,cache,output,device='cpu',batch_size=2,microbatch_size=2):
    from .policy import Policy
    from .data import Trajectories
    from .optimization import imitation_update
    torch.set_num_threads(4);torch.manual_seed(17)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for the real training probe')
    policy=Policy(checkpoint,device);policy.model.eval()
    for parameter in policy.model.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(policy.model.parameters(),lr=1e-5,betas=(.95,.999),weight_decay=1e-6)
    ds=Trajectories(data,'train',policy.cfg,cache_root=cache)
    ds.groups['nominal']=[(m,p) for m,p in ds.groups['nominal'] if m.get('group')=='base']
    rng=np.random.default_rng(17)
    examples=[ds.sample(rng,0.)[0] for _ in range(batch_size)]
    assert all(all(k.startswith('observation.') or k in ['action','action_is_pad'] for k in example) for example in examples)
    selected=list(policy.model.parameters())[-1];before=selected.detach().clone()
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter();loss,grad=imitation_update(policy,optimizer,examples,microbatch_size)
    if device.startswith('cuda'):torch.cuda.synchronize()
    change=float((selected.detach()-before).abs().max())
    assert change>0 and np.isfinite(loss) and np.isfinite(grad)
    report=dict(passed=True,device=device,examples=batch_size,microbatch_size=microbatch_size,loss=loss,grad_norm=grad,
        seconds=time.perf_counter()-start,selected_parameter_change=change,
        trainable_parameters=sum(p.numel() for p in policy.model.parameters()),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
        checkpoint_saved=False,note='Probe weights discarded; all actual training starts from the supplied checkpoint.')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2)+'\n')
    del optimizer,policy,ds,examples,before,selected;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/base');p.add_argument('--data',default='artifacts/round1/data')
    p.add_argument('--cache',default='artifacts/expert_preview/training_probe/cache');p.add_argument('--output',default='artifacts/expert_preview/training_probe/report.json')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=2);p.add_argument('--microbatch-size',type=int,default=2)
    a=p.parse_args();bootstrap();print(json.dumps(profile_update(a.checkpoint,a.data,a.cache,a.output,a.device,a.batch_size,a.microbatch_size),indent=2))


if __name__=='__main__':main()
