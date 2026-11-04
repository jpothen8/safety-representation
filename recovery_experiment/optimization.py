"""Standard diffusion imitation updates, with actual batched GPU forwards."""
import torch


def imitation_update(policy,optimizer,batches,microbatch_size=1):
    if not batches or microbatch_size<1:raise ValueError('Require examples and a positive microbatch size')
    optimizer.zero_grad(set_to_none=True)
    total=len(batches);mean_loss=0.
    for offset in range(0,total,microbatch_size):
        examples=batches[offset:offset+microbatch_size]
        keys=set(examples[0])
        if any(set(x)!=keys for x in examples):raise ValueError('Batch schemas differ')
        batch={key:torch.cat([example[key] for example in examples],dim=0) for key in keys}
        loss=policy.imitation_loss(batch)
        weight=len(examples)/total
        (loss*weight).backward()
        mean_loss+=float(loss.detach())*weight
    parameters=[p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
    if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient')
    optimizer.step()
    return mean_loss,float(grad)
