import torch
import torch.optim

from .utils import warmup, exp_avg_sq_, beta_debias, update_param_


class ForeachAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup_steps=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, k=0, warmup_steps=warmup_steps, train_mode=True, weight_sum=0.0,
                        lr_max=-1.0, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            decay = group['weight_decay']
            k = group['k']

            if not group['train_mode']:
                raise Exception("Not in train mode!")

            active_p = [p for p in group['params'] if p.grad is not None]

            for p in active_p:
                if 'exp_avg' not in self.state[p]:
                    self.state[p]['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float32)
                    self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float32)

            y, grad, exp_avg_sq, exp_avg = zip(
                *[(p.data, p.grad.float(), self.state[p]['exp_avg_sq'], self.state[p]['exp_avg']) for p in active_p])

            # Decay the first and second moment running average coefficient
            exp_avg.lerp_(grad, 1 - beta_debias(group['betas'][0], k + 1))
            denom = exp_avg_sq_(exp_avg_sq, grad, beta_debias(group['betas'][1], k + 1), eps)

            # Normalize grad in-place for memory efficiency
            lr = -warmup(group['lr'], k + 1, group['warmup_steps'])
            update_param_(y, exp_avg, lr, decay, lambda p, e, l: torch._foreach_addcdiv_(p, e, denom, l))
            group['k'] = k + 1
        return loss
