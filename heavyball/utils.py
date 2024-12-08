"""


Originally from Evan Walters and Omead Pooladzandi, 2024
Modified under Creative Commons Attribution 4.0 International
Source available at https://github.com/evanatyourservice/kron_torch/blob/97a2b5ee8a1a4c29e4780bbf6c521e545189eff9/kron_torch/kron.py
"""

import functools
import gc
import math
import random
import string
from typing import List, Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch import Tensor
from torch._dynamo.exc import TorchDynamoException
from torch.backends import cudnn, opt_einsum
from torch.utils._pytree import tree_map

compile_mode = "max-autotune-no-cudagraphs"
dynamic = False
compile_mode_recommended_to_none = None
zeroth_power_mode = 'qr'  # 'qr' is baseline, 'newtonschulz' converges better and faster, 'eigh' is perfect but slow
tiny_bf16 = torch.finfo(torch.bfloat16).tiny


def decorator(func):
    compiled = None

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        disable = compile_mode_recommended_to_none is None
        if is_compiling() or compile_mode_recommended_to_none is None:
            return func(*args, **kwargs)
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(fullgraph=True, dynamic=dynamic, mode=compile_mode_recommended_to_none)(func)
        return compiled(*args, **kwargs)

    return _fn


def decorator_knowngood(func: Callable):
    compiled = None

    @functools.wraps(func)
    def _fn(*args, **kwargs):
        if is_compiling() or compile_mode is None:
            return func(*args, **kwargs)
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(fullgraph=True, dynamic=dynamic, mode=compile_mode)(func)
        return compiled(*args, **kwargs)

    return _fn


einsum_base = string.ascii_lowercase + string.ascii_uppercase


def warmup(lr: float, step: int, warmup_steps: int):
    if step >= warmup_steps:  # if instead of min to guard against 0 div
        return lr
    return lr * step / warmup_steps


@decorator_knowngood
def _compilable_schedule_free_(p: List[Tensor], z: List[Tensor], ckp1: Tensor, grad: List[Tensor], lr: Tensor,
                               beta1: Tensor, decay: float):
    grad = [u_.view_as(p_) for u_, p_ in zip(grad, p)]
    p32, z32, g32 = [list(map(promote, x)) for x in (p, z, grad)]
    for p_, z_, g_ in zip(p32, z32, g32):
        if decay != 0:
            g_.add_(p_, alpha=decay)
        p_.lerp_(z_, ckp1)
        p_.add_(g_, alpha=lr - lr * (beta1 * (1 - ckp1)))
        z_.add_(g_, alpha=lr)
    copy_stochastic_list_(p, p32)
    copy_stochastic_list_(z, z32)


def schedule_free_(lr: float, weight_lr_power: float, weight_sum: float, beta1: float, parameters: List[Tensor],
                   z: List[Tensor], grad: List[Tensor], r: float = 0.0, step: int = 0, decay: float = 0.0):
    weight = abs(lr) ** weight_lr_power * max(step, 1) ** r
    weight_sum = weight_sum + weight

    try:
        ckp1 = weight / weight_sum
    except ZeroDivisionError:
        ckp1 = 0
    ckp1 = 0

    # These operations update y in-place,
    # without computing x explicitly.
    lr, ckp1, beta1 = scalar_guard(lr, parameters[0]), scalar_guard(ckp1, parameters[0]), scalar_guard(beta1, parameters[0])
    _compilable_schedule_free_(parameters, z, ckp1, grad, lr, beta1, decay)
    return weight_sum


def append_or_extend(base, new):
    if isinstance(new, list):
        base.extend(new)
    else:
        base.append(new)


def dim_merger(grad, max_precond_dim, split: bool = False):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done
    """
    shape = grad.shape
    new_shape = []

    curr_shape = 1

    for sh in shape[1:][::-1]:
        temp_shape = curr_shape * sh
        if temp_shape > max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape
    new_shape = [*shape[:1], *new_shape[::-1]]

    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)

    new_grad = grad.reshape(new_shape)  # needs to be .reshape() due to channels_last
    if not split:
        return new_grad

    grads = [new_grad]
    for i, sh in reversed(list(enumerate(new_shape[:]))):
        if sh == 1:
            grads = [g.squeeze(dim=i) for g in grads]
            continue
        if sh <= max_precond_dim:
            continue
        grads = [a for g in grads for a in g.split(max_precond_dim, dim=i)]
    if len(grads) == 1:
        return new_grad
    new_grads = []
    for g in grads:
        append_or_extend(new_grads, dim_merger(g, max_precond_dim, split))
    return new_grads


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta ** step)


@decorator_knowngood
def _compilable_exp_avg_sq_(state: List[Tensor], grad: List[Tensor], beta2: Tensor, eps: Tensor,
                            out: List[Optional[Tensor]]):
    s32, g32 = [list(map(promote, x)) for x in (state, grad)]
    torch._foreach_mul_(s32, beta2)
    [s.addcmul_(g, g, value=1 - beta2) for s, g in zip(s32, g32)]
    denom = torch._foreach_sqrt(s32)
    [d.clamp_(min=eps) for d in denom]
    copy_stochastic_list_(state, s32)

    if out[0] is None:
        return denom

    copy_stochastic_list_(out, denom)
    return out


def exp_avg_sq_(state, grad, beta2, eps, out=None):
    state, grad, out = list_guard(state), list_guard(grad), list_guard(out)
    beta2, eps = scalar_guard(beta2, state[0]), scalar_guard(eps, state[0])
    return _compilable_exp_avg_sq_(state, grad, beta2, eps, out)


@decorator_knowngood
def _compilable_scale_by_exp_avg_sq_(state: List[Tensor], grad: List[Tensor], beta2: Tensor, eps: Tensor,
                                     out: List[Optional[Tensor]]):
    s32, g32 = [list(map(promote, x)) for x in (state, grad)]
    torch._foreach_mul_(s32, beta2)
    [s.addcmul_(g, g, value=1 - beta2) for s, g in zip(s32, g32)]
    denom = torch._foreach_sqrt(s32)
    [d.clamp_(min=eps) for d in denom]
    out = torch._foreach_div(g32, denom)
    copy_stochastic_list_(state, s32)
    return stochastic_round_list_(grad, out)


def scale_by_exp_avg_sq_(grad, exp_avg_sq, beta2, eps):
    grad, exp_avg_sq = list_guard(grad), list_guard(exp_avg_sq)
    beta2, eps = scalar_guard(beta2, grad[0]), scalar_guard(eps, grad[0])
    return _compilable_scale_by_exp_avg_sq_(grad, exp_avg_sq, beta2, eps, grad)


@decorator_knowngood
def _compilable_agc_(parameters: List[Tensor], gradients: List[Tensor], clip_val: float, minimum: float, eps: float):
    p_norm = torch._foreach_norm(parameters)
    g_norm = torch._foreach_norm(gradients)
    torch._foreach_maximum_(p_norm, minimum)
    torch._foreach_maximum_(g_norm, eps)
    torch._foreach_div_(p_norm, g_norm)
    torch._foreach_mul_(p_norm, clip_val)
    torch._foreach_minimum_(p_norm, 1)
    return torch._foreach_mul(gradients, p_norm)


def adaptive_gradient_clipping_(parameters: List[Tensor], gradients: List[Tensor], clip_val: float,
                                minimum: float = 1e-3, eps: float = 1e-8):
    if clip_val <= 0:
        return gradients
    parameters, gradients = list_guard(parameters), list_guard(gradients)
    clip_val = scalar_guard(clip_val, parameters[0])
    return _compilable_agc_(parameters, gradients, clip_val, minimum, eps)


def is_compiling():
    try:
        return torch.compiler.is_compiling()
    except TorchDynamoException:
        return True


def set_(dst: Tensor, src: Tensor):
    if not is_compiling() and src.data_ptr() == dst.data_ptr():
        return
    if src.shape != dst.shape:
        src = src.reshape_as(dst)
    dst.copy_(src)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


def set_torch():
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")  # highest: FP32, high: TF32, medium: bf16
    opt_einsum.enabled = True
    opt_einsum.strategy = "auto-hq"


@decorator
def zeropower_via_newtonschulz5(G, init, steps=2, eps=1e-7):
    """
    Modified from "modded-nanogpt" under the MIT license:
    Original: https://github.com/KellerJordan/modded-nanogpt/blob/a0dcbfdd9a0617d091d5123cfc354745428e40d3/train_gpt2.py

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    init = init / (init.norm() + eps)  # ensure top singular value <= 1
    X = X / (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T  # preconditioner
        B = A @ init
        init = X = a * init + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


def ortho(x):
    if zeroth_power_mode == 'qr':
        return torch.linalg.qr(x).Q
    if zeroth_power_mode == 'svd':
        u, s, v = torch.linalg.svd(x)
        return u @ v.T
    raise NotImplementedError(f"Unknown zeroth_power_mode: {zeroth_power_mode}")


def get_orthogonal_matrix_QR(GG, Q, exp_avg_sq):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    for m, o in zip(GG, Q):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))
        else:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))

    indices = []

    for ind, (m, o, q) in enumerate(zip(matrix, orth_matrix, Q)):
        if len(m) == 0:
            indices.append(None)
            continue

        tmp = m @ o
        est_eig = torch.einsum('ij,ij->j', o, tmp)
        sort_idx = torch.argsort(est_eig, descending=True)
        indices.append(sort_idx)
        if zeroth_power_mode == 'eigh':
            set_(q, torch.linalg.eigh(m)[1])
        elif zeroth_power_mode.startswith('newtonschulz'):
            iterations = zeroth_power_mode[len('newtonschulz'):]
            if iterations == '':
                iterations = 10
            else:
                iterations = int(iterations)
            set_(q, zeropower_via_newtonschulz5(m, o[:, sort_idx], iterations))
        else:
            set_(q, ortho(tmp[:, sort_idx]))

    indices = tuple(slice(None) if ind is None else ind.view(*(1,) * i, -1, *(1,) * (exp_avg_sq.dim() - i - 1))  #
                    for i, ind in enumerate(indices))
    set_(exp_avg_sq, exp_avg_sq[indices])


def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(promote(m.data))
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue

        device, dtype = m.device, m.dtype
        for modifier in (None, torch.double, 'cpu'):
            if modifier is not None:
                m = m.to(modifier)
            try:
                Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))[1].to(device=device,
                                                                                                dtype=dtype)
                break
            except torch.OutOfMemoryError:
                pass
            except RuntimeError:  # failed to compute eigenvalues
                continue
            clean()
        else:
            raise RuntimeError("Failed to compute eigenvalues.")

        Q = torch.flip(Q, [1])

        final.append(Q)

    return final


@decorator_knowngood
def _compilable_stochastic_lerp_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        x32.lerp_(y32, a)
        copy_stochastic_(x_, x32)


def get_beta1(group):
    beta = None
    if 'beta' in group:
        beta = group['beta']
    if beta is None and 'betas' in group:
        beta = group['betas'][0]
    if beta is None:
        raise ValueError("Beta not found in group.")
    return beta


def get_beta2(group):
    beta = None
    if 'beta2_scale' in group:
        step = max(group.get("step", 1), 1)
        return 1 - step ** -group['beta2_scale']
    if 'betas' in group:
        return group['betas'][1]
    raise ValueError("Beta2 not found in group.")


def stochastic_lerp_(x: List[Tensor], y: List[Tensor], a: Union[float, int, Tensor]):
    x, y = list_guard(x), list_guard(y)
    a = scalar_guard(a, x[0])
    _compilable_stochastic_lerp_(x, y, a)


def list_guard(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]


def scalar_guard(x, ref):
    if isinstance(x, float):
        return torch.empty((), dtype=torch.float32, device=ref.device).fill_(x)
    if isinstance(x, int):
        return torch.empty((), dtype=torch.int64, device=ref.device).fill_(x)
    return x


@decorator_knowngood
def _compilable_stochastic_add_(x: List[Tensor], y: List[Tensor], alpha: Union[float, int, Tensor]):
    for x_, y_ in zip(x, y):
        x32 = promote(x_)
        y32 = promote(y_)
        x32.add_(y32, alpha=alpha)
        copy_stochastic_(x_, x32)


def stochastic_add_(x: List[Tensor], y: List[Tensor], alpha: Union[float, int, Tensor]):
    x, y = list_guard(x), list_guard(y)
    alpha = scalar_guard(alpha, x[0])
    _compilable_stochastic_add_(x, y, alpha)


@decorator
def compute_ggt(grad, GG, max_precond_dim, precondition_1d, beta):
    if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return

    for idx, sh in enumerate(grad.shape):
        if sh > max_precond_dim:
            continue
        b = einsum_base[idx]
        g0 = einsum_base[:grad.dim()]
        g1 = g0.replace(b, b.upper())
        outer_product = torch.einsum(f'{g0},{g1}->{b + b.upper()}', grad, grad)
        GG[idx].lerp_(promote(outer_product), 1 - beta)


def promote(x):
    if isinstance(x, torch.dtype) and x in (torch.bfloat16, torch.float16):
        return torch.float32
    if isinstance(x, Tensor) and x.dtype in (torch.bfloat16, torch.float16):
        return x.float()
    return x


def min_dtype(xs: List[Tensor]):
    dtypes = [x.dtype for x in xs]
    for d in (torch.float32, torch.bfloat16, torch.float16):
        if all(x in (d, torch.float32, torch.float64) for x in dtypes):
            return d
    return torch.float32


def update_preconditioner(grad, Q, GG, exp_avg_sq, max_precond_dim, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    compute_ggt(grad, GG, max_precond_dim, precondition_1d, beta)
    if update_precond:
        get_orthogonal_matrix_QR(GG, Q, exp_avg_sq)


def init_preconditioner(grad, state, beta, max_precond_dim=10000, precondition_1d=False):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state['GG'] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if precondition_1d or grad.shape[0] > max_precond_dim:
            state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype))
        else:
            state['GG'].append([])

    else:
        for sh in grad.shape:
            if sh > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype))

    compute_ggt(grad, state['GG'], max_precond_dim, precondition_1d, beta)
    state['Q'] = get_orthogonal_matrix(state['GG'])


@decorator
def project(grad, Q, back: bool):
    """

    :param grad:
    :param Q:
    :param merge_dims:
    :param max_precond_dim:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    param = einsum_base[:grad.dim()]
    preconditioners = ",".join([(g + g.upper())[::-1 if back else 1] for m, g in zip(Q, param) if len(m) > 0])
    if preconditioners:
        out = ''.join([c.upper() if c.upper() in preconditioners else c for c in param])
        grad = torch.einsum(f'{param},{preconditioners}->{out}', grad, *[q for q in Q if len(q) > 0])
    return grad


class StatefulOptimizer(torch.optim.Optimizer):
    ema_decay: float = 0.001

    def __init__(self, params, defaults, foreach: bool = True, use_ema: bool = False):
        super().__init__(params, {**defaults, 'foreach': foreach})
        self.fake_groups = {}
        self.use_ema = use_ema
        self.mapping = {}

    def get_groups(self, group):
        if group['foreach']:
            return [group]

        for p in group['params']:
            if p not in self.fake_groups:
                self.fake_groups[p] = {**group, 'params': [p]}

        return [self.fake_groups[p] for p in group['params']]

    def state_(self, arg: Tensor):
        return self.state[self.mapping.get(arg, arg)]

    def mars_correct_list(self, group, p_list, g_list, mars_gamma, beta):
        for p, g in zip(p_list, g_list):
            state = self.state_(p)
            if 'mars_old_grad' not in state:
                state['mars_old_grad'] = torch.zeros_like(g)
        old_gs = [self.state_(p)['mars_old_grad'] for p in p_list]
        mars_correction(g_list, old_gs, mars_gamma, beta)

    def split_p_and_g_in_group(self, group: dict, skip_none: bool = True, should_promote: bool = True,
                               beta1: float = -1.0):
        for p in group["params"]:
            if skip_none and p.grad is None:
                continue

            if p.grad is None:
                grad = None
            else:
                if should_promote:
                    grad = promote(p.grad)
                else:
                    grad = p.grad
                if beta1 >= 0 and group.get('mars', False):
                    self.mars_correct_list(group, [p], [grad], group['mars_gamma'], beta1)

                p.grad = None

            p_views = merge_group(group, p)
            if grad is not None:
                grad = merge_group(group, grad)
            for i, pv in enumerate(p_views):
                self.mapping[pv] = (p, i)
            if isinstance(p_views, Tensor):
                yield p_views, grad
                continue
            if grad is None:
                yield from zip(p_views, [None] * len(p_views))
                continue
            yield from zip(p_views, grad)

    def state_size(self) -> int:
        total_bytes = 0

        def _add(x):
            nonlocal total_bytes
            if isinstance(x, Tensor):
                total_bytes += x.numel() * x.element_size()

        for group in self.param_groups:
            for p, _ in self.split_p_and_g_in_group(group, skip_none=False):
                tree_map(_add, self.state_(p))
        return total_bytes

    def _step(self, group):
        raise NotImplementedError

    def ema_update(self):
        with torch.no_grad():
            for top_group in self.param_groups:
                for group in self.get_groups(top_group):
                    active_p = [p for p in group['params']]

                    if not active_p:
                        return

                    k = group['ema_step'] = group.get('ema_step', -1) + 1

                    for p in active_p:
                        if 'param_ema' not in self.state_(p):
                            self.state_(p)['param_ema'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                    y, param_ema = zip(*[(p.data, self.state_(p)['param_ema']) for p in active_p])
                    torch._foreach_lerp_(param_ema, y, weight=beta_debias(1 - self.ema_decay, k + 1))

    def copy_emas_to_params(self):
        with torch.no_grad():
            for top_group in self.param_groups:
                for group in self.get_groups(top_group):
                    active_p = [p for p in group['params']]

                    if not active_p:
                        return

                    for p in active_p:
                        if 'param_ema' in self.state_(p):
                            p_clone = p.data.clone()
                            set_(p.data, self.state_(p)['param_ema'])
                            set_(self.state_(p)['param_ema'], p_clone)

    def copy_params_to_emas(self):
        with torch.no_grad():
            for top_group in self.param_groups:
                for group in self.get_groups(top_group):
                    active_p = [p for p in group['params']]

                    if not active_p:
                        return

                    for p in active_p:
                        if 'param_ema' in self.state_(p):
                            ema_clone = self.state_(p)['param_ema'].data.clone()
                            set_(self.state_(p)['param_ema'], p.data)
                            set_(p.data, ema_clone)

    def step(self, closure: Optional[Callable] = None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        # we assume that parameters are constant and that there are no excessive recompiles
        with torch.no_grad(), torch._dynamo.utils.disable_cache_limit():
            for top_group in self.param_groups:
                for group in self.get_groups(top_group):
                    self._step(group)
                    self.mapping.clear()
                    if self.use_ema:
                        self.ema_update(group)

        return loss


def copy_stochastic_list_(target: List[Tensor], source: List[Tensor]):
    for t, s in zip(target, source):
        copy_stochastic_(t, s)


@decorator_knowngood
def _compilable_adam_(exp_avg: List[Tensor], exp_avg_sq: List[Tensor], grad: List[Tensor], beta1: Tensor, beta2: Tensor,
                      step: Tensor):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    g32, exp_avg_sq32, exp_avg32 = [list(map(promote, x)) for x in [grad, exp_avg_sq, exp_avg]]

    [ea32.lerp_(g, 1 - beta1) for ea32, g in zip(exp_avg32, g32)]
    denom = exp_avg_sq_(exp_avg_sq32, g32, beta2, 1e-8)
    u32 = torch._foreach_div(exp_avg32, denom)

    copy_stochastic_list_(exp_avg, exp_avg32)
    copy_stochastic_list_(exp_avg_sq, exp_avg_sq32)
    return stochastic_round_list_(exp_avg, u32)


def adam_(exp_avg: List[Tensor], exp_avg_sq: List[Tensor], grad: List[Tensor], beta1: float, beta2: float, step: int):
    exp_avg, exp_avg_sq, grad = map(list_guard, (exp_avg, exp_avg_sq, grad))
    beta1, beta2, step = scalar_guard(beta1, exp_avg[0]), scalar_guard(beta2, exp_avg[0]), scalar_guard(step,
                                                                                                        exp_avg[0])
    return _compilable_adam_(exp_avg, exp_avg_sq, grad, beta1, beta2, step)


@decorator_knowngood
def _fused_compilable_adam_(y: List[Tensor], exp_avg: List[Tensor], exp_avg_sq: List[Tensor], grad: List[Tensor],
                            beta1: Tensor, beta2: Tensor, step: Tensor, decay: Tensor, lr: Tensor, eps: Tensor,
                            caution: bool):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    g32, exp_avg_sq32, exp_avg32 = [list(map(promote, x)) for x in [grad, exp_avg_sq, exp_avg]]

    [ea32.lerp_(g, 1 - beta1) for ea32, g in zip(exp_avg32, g32)]
    denom = exp_avg_sq_(exp_avg_sq32, g32, beta2, 1e-8)
    u32 = torch._foreach_div(exp_avg32, denom)

    copy_stochastic_list_(exp_avg, exp_avg32)
    copy_stochastic_list_(exp_avg_sq, exp_avg_sq32)
    _compilable_update_(y, u32, decay, lambda a, b, c: a.add_(b, alpha=c), lr, caution, g32)


def fused_adam_(y: List[Tensor], exp_avg: List[Tensor], exp_avg_sq: List[Tensor], grad: List[Tensor], beta1: float,
                beta2: float, step: int, lr: float, eps: float, decay: float, caution: bool):
    y, exp_avg, exp_avg_sq, grad = map(list_guard, (y, exp_avg, exp_avg_sq, grad))
    beta1, beta2, step, lr = [scalar_guard (x, y[0]) for x in (beta1, beta2, step, lr)]
    return _fused_compilable_adam_(y, exp_avg, exp_avg_sq, grad, beta1, beta2, step, decay, lr, eps, caution)


@decorator_knowngood
def _compilable_laprop_(exp_avg: List[Tensor], exp_avg_sq: List[Tensor], grad_projected: List[Tensor], beta1: Tensor,
                        beta2: Tensor, step: Tensor):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    gp32, exp_avg_sq32 = [list(map(promote, x)) for x in [grad_projected, exp_avg_sq]]

    denom = exp_avg_sq_(exp_avg_sq32, gp32, beta2, 1e-8)
    gp32 = torch._foreach_div(gp32, denom)
    stochastic_lerp_(exp_avg, gp32, 1 - beta1)

    copy_stochastic_list_(exp_avg_sq, exp_avg_sq32)


def laprop_(exp_avg: List[Tensor], exp_avg_sq: List[Tensor], grad_projected: List[Tensor], beta1: float, beta2: float,
            step: int):
    exp_avg, exp_avg_sq, grad_projected = list_guard(exp_avg), list_guard(exp_avg_sq), list_guard(grad_projected)
    beta1, beta, step = scalar_guard(beta1, exp_avg[0]), scalar_guard(beta2, exp_avg[0]), scalar_guard(step, exp_avg[0])
    _compilable_laprop_(exp_avg, exp_avg_sq, grad_projected, beta1, beta2, step)
    return exp_avg


@decorator_knowngood
def _fused_compilable_laprop_(y: List[Tensor], exp_avg: List[Tensor], exp_avg_sq: List[Tensor],
                              grad_projected: List[Tensor], beta1: Tensor, beta2: Tensor, step: Tensor, lr: Tensor,
                              decay: Tensor, caution: bool):
    beta1 = beta_debias(beta1, step)
    beta2 = beta_debias(beta2, step)

    gp32, exp_avg_sq32 = [list(map(promote, x)) for x in [grad_projected, exp_avg_sq]]

    denom = exp_avg_sq_(exp_avg_sq32, gp32, beta2, 1e-8)
    gp32 = torch._foreach_div(gp32, denom)
    stochastic_lerp_(exp_avg, gp32, 1 - beta1)
    update_param_(y, gp32, lr, decay, caution=caution, grad=gp32)

    copy_stochastic_list_(exp_avg_sq, exp_avg_sq32)


def fused_laprop_(y: List[Tensor], exp_avg: List[Tensor], exp_avg_sq: List[Tensor], grad_projected: List[Tensor],
                  beta1: float, beta2: float, step: int, lr: float, decay: float, caution: bool):
    y, exp_avg, exp_avg_sq, grad_projected = map(list_guard, (y, exp_avg, exp_avg_sq, grad_projected))
    beta1, beta2, step, lr = [scalar_guard (x, y[0]) for x in (beta1, beta2, step, lr)]
    _fused_compilable_laprop_(y, exp_avg, exp_avg_sq, grad_projected, beta1, beta2, step, lr, decay, caution)


@decorator_knowngood
def _fused_compilable_adopt_(y, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution):
    g32, exp_avg32, exp_avg_sq32 = [list(map(promote, x)) for x in [grad, exp_avg, exp_avg_sq]]
    update_param_(y, exp_avg, lr, decay, caution=caution, grad=g32)

    beta1 = beta_debias(beta1, step)
    denom = torch._foreach_sqrt(exp_avg_sq32)
    [denom.clamp_(min=eps) for denom in denom]
    torch._foreach_mul_(exp_avg32, beta1)
    [ea32.addcdiv_(g, d, value=1 - beta1) for ea32, g, d in zip(exp_avg32, g32, denom)]

    beta2 = beta_debias(beta2, step + 1)
    torch._foreach_mul_(exp_avg_sq32, beta2)
    [eas32.addcmul_(g, g, value=1 - beta2) for eas32, g in zip(exp_avg_sq32, g32)]

    copy_stochastic_list_(exp_avg, exp_avg32)
    copy_stochastic_list_(exp_avg_sq, exp_avg_sq32)


def fused_adopt_(y, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution):
    y, grad, exp_avg_sq, exp_avg = list_guard(y), list_guard(grad), list_guard(exp_avg_sq), list_guard(exp_avg)
    beta1, beta2, step, lr = [scalar_guard (x, y[0]) for x in (beta1, beta2, step, lr)]
    _fused_compilable_adopt_(y, grad, exp_avg_sq, exp_avg, beta1, beta2, step, lr, eps, decay, caution)


@decorator_knowngood
def _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step):
    g32, exp_avg32, exp_avg_sq32 = [list(map(promote, x)) for x in [grad, exp_avg, exp_avg_sq]]
    update = [e.clone() for e in exp_avg]

    beta1 = beta_debias(beta1, step)
    denom = torch._foreach_sqrt(exp_avg_sq32)
    [denom.clamp_(min=1e-8) for denom in denom]
    torch._foreach_mul_(exp_avg32, beta1)
    [ea32.addcdiv_(g, d, value=1 - beta1) for ea32, g, d in zip(exp_avg32, g32, denom)]

    beta2 = beta_debias(beta2, step + 1)
    torch._foreach_mul_(exp_avg_sq32, beta2)
    [eas32.addcmul_(g, g, value=1 - beta2) for eas32, g in zip(exp_avg_sq32, g32)]

    copy_stochastic_list_(exp_avg, exp_avg32)
    copy_stochastic_list_(exp_avg_sq, exp_avg_sq32)

    return update


def adopt(grad, exp_avg_sq, exp_avg, beta1, beta2, step):
    grad, exp_avg_sq, exp_avg = list_guard(grad), list_guard(exp_avg_sq), list_guard(exp_avg)
    beta1, beta2, step = scalar_guard(beta1, grad[0]), scalar_guard(beta2, grad[0]), scalar_guard(step, grad[0])
    return _compilable_adopt_(grad, exp_avg_sq, exp_avg, beta1, beta2, step)


def stochastic_round_list_(ref: List[Tensor], source: List[Tensor]):
    return [stochastic_round_(r, s) for r, s in zip(ref, source)]


@decorator_knowngood
def stochastic_round_(ref: Tensor, source: Tensor):
    if source.dtype == torch.bfloat16 or ref.dtype == source.dtype:
        return source
    if ref.dtype != torch.bfloat16:
        return source.to(ref.dtype)
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32
    return result.view(dtype=torch.float32).bfloat16()


@decorator_knowngood
def _compilable_copy_stochastic_(target: Tensor, source: Tensor):
    target.copy_(stochastic_round_(target, source))


def copy_stochastic_(target: Tensor, source: Tensor):
    if not is_compiling() and target.data_ptr() == source.data_ptr():
        return
    if target.dtype == torch.bfloat16 and source.dtype in (torch.float16, torch.float32, torch.float64):
        _compilable_copy_stochastic_(target, source.float())
    set_(target, source)


@decorator_knowngood
def _compilable_update_(p: List[Tensor], u: List[Tensor], decay: Tensor, add_fn: callable, lr: Tensor, caution: bool,
                        g: List[Optional[Tensor]]):
    u = [u_.view_as(p_) for u_, p_ in zip(u, p)]
    p32, u32 = [list(map(promote, x)) for x in [p, u]]

    if decay > 0:
        torch._foreach_mul_(p32, 1 - decay * lr)

    for p32_, u32_, g_ in zip(p32, u32, g):  # lr is data-dependent -> can't compile a foreach
        if caution:
            _compilable_cautioning_(promote(g_), u32_)
        add_fn(p32_, u32_, lr)

    copy_stochastic_list_(p, p32)


def update_param_(param: List[Tensor], update: List[Tensor], lr: float, decay: float, add_fn: callable = None,
                  caution: bool = False, grad: List[Tensor] = None):
    param, update, grad = list_guard(param), list_guard(update), list_guard(grad)
    lr = scalar_guard(lr, param[0])
    if not caution:
        grad = [None] * len(param)
    if add_fn is None:
        add_fn = stochastic_add_
    _compilable_update_(param, update, decay, add_fn, lr, caution, grad)


def precond_schedule(step, precond_scheduler, rng):
    precond_prob = max(step, 1) ** precond_scheduler[0]
    precond_prob = math.log10(precond_prob)
    precond_prob = precond_prob ** precond_scheduler[1] + 1
    precond_prob = 1 / precond_prob
    update_precond = rng.random() < precond_prob
    return update_precond


def init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape

    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
        return [Q, (exprA, tuple(exprGs), exprP)]

    # Tensor
    if len(shape) > 13:
        raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

    scale = scale ** (1 / len(shape))

    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == "one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [False for _ in shape]
        dim_diag[rev_sorted_dims[0]] = True
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                         "[None, 'one_diag', 'all_diag']")

    Q = []
    piece1A, piece2A, piece3A = ([], "", "")
    exprGs = []
    piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
    for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
        if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
            # use diagonal matrix as preconditioner for this dim
            Q.append(scale * torch.ones(size, dtype=promote(dtype), device=t.device))

            piece1A.append(letters[i])
            piece2A = piece2A + letters[i]
            piece3A = piece3A + letters[i]

            piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
            exprGs.append(subscripts)

            piece1P.append(letters[i + 13])
            piece2P.append(letters[i + 13])
            piece3P = piece3P + letters[i + 13]
            piece4P = piece4P + letters[i + 13]
        else:
            # use triangular matrix as preconditioner for this dim
            Q.append(scale * torch.eye(size, dtype=dtype, device=t.device))

            piece1A.append(letters[i] + letters[i + 13])
            piece2A = piece2A + letters[i + 13]
            piece3A = piece3A + letters[i]

            piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
            piece2 = "".join([(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))])
            subscripts = (piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26])
            exprGs.append(subscripts)

            a, b, c = (letters[i], letters[i + 13], letters[i + 26])
            piece1P.append(a + b)
            piece2P.append(a + c)
            piece3P = piece3P + c
            piece4P = piece4P + b

    exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
    exprP = (",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P)
    return [Q, (exprA, tuple(exprGs), exprP)]


@decorator
def psgd_balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    torch._foreach_mul_(Q_in, list(norms))


def psgd_calc_A_and_conjB(exprA, G, Q):
    V = torch.randn(G.shape, dtype=G.dtype, device=G.device)
    eps = torch.tensor(torch.sqrt(torch.finfo(torch.float32).eps), dtype=G.dtype, device=G.device)
    G += eps * torch.mean(torch.abs(G)) * V
    md = min_dtype(Q + [G])
    A = torch.einsum(exprA, *[q.to(md) for q in Q], G.to(md)).to(G.dtype)
    order = G.dim()
    p = list(range(order))
    conjB = torch.permute(V, p[1:] + p[:1]).to(promote(G.dtype))
    Q = [promote(q) for q in Q]
    for i, q in enumerate(Q):
        if q.dim() <= 1:
            conjB /= q
        else:
            unsqueeze = conjB.dim() <= 1
            if unsqueeze:
                conjB = conjB.unsqueeze(0)
            conjB = torch.linalg.solve_triangular(q, conjB, upper=True, left=False)
            if unsqueeze:
                conjB = conjB.squeeze(0)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    return A, conjB


def psgd_lb(A, max_abs):
    A /= max_abs
    a0 = torch.einsum('ij,ij->j', A, A)
    i = torch.argmax(a0)

    x = torch.index_select(A, 1, i).flatten().contiguous()

    x = torch.einsum('i,ij->j', x, A)
    x /= x.norm()
    x = torch.einsum('j,kj->k', x, A)
    x = x.norm()
    x *= max_abs
    return x


@decorator
def psgd_update_precond(Q, exprs, G, precond_lr, oq, store_triu_as_line):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs

    A, conjB = psgd_calc_A_and_conjB(exprA, G, Q)

    for q, exprG, o in zip(Q, exprGs, oq):
        term1 = promote(torch.einsum(exprG, A, A))
        term2 = promote(torch.einsum(exprG, conjB, conjB))

        term2 += term1  # a + b
        term1 *= 2  # 2a
        if term1.dtype == term2.dtype:
            term1 -= term2  # 2a - (a + b) == a - b
        else:
            term1 = term1 - term2

        term1 *= precond_lr
        norm = term2.norm(float('inf'))
        if q.dim() < 2:
            term1 *= q.to(term1.dtype)
            term1 /= norm.clamp_(min=tiny_bf16)
        else:
            torch.triu(term1, out=term1)
            term1 /= psgd_lb(term2, norm).clamp_(tiny_bf16)
            torch.matmul(term1, q, out=term1)
        if store_triu_as_line:
            term1 = triu_to_line([term1])[0][1]
            o = o[1]
        stochastic_add_([o], [term1], -1)


@decorator_knowngood
def _compilable_l2_clip_(x):
    ref = x
    x = list(map(promote, x))
    norm = torch._foreach_norm(x)
    torch._foreach_maximum_(norm, 1e-8)
    out = torch._foreach_div(x, norm)
    return stochastic_round_list_(ref, out)


def l2_clip_(x):
    x = list_guard(x)
    return _compilable_l2_clip_(x)


@decorator_knowngood
def _compilable_rmsnorm_clip_(x):
    x = list(map(promote, x))
    norm = torch._foreach_norm(x)
    norm = [n.div_(x_.numel() ** 0.5) for n, x_ in zip(norm, x)]
    torch._foreach_maximum_(norm, 1e-6)
    return torch._foreach_div(x, norm)


def rmsnorm_clip_(x):
    x = list_guard(x)
    return _compilable_rmsnorm_clip_(x)


def mu_law_compress(x, mu=127.0):
    """
    Foreach version of https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py

    μ-law compression
    Args:
        x: Input tensor
        mu: Compression parameter (default 127.0 for behavior similar to trust_region=1.5)
    """
    xa = torch._foreach_abs_(x)
    torch._foreach_mul_(xa, mu)
    torch._foreach_log1p_(xa)
    torch._foreach_div_(xa, math.log1p(mu))
    return [xa_.copysign_(x_) for x_, xa_ in zip(x, xa)]


def a_law_compress(x, A=87.6):
    """
    Foreach version of https://github.com/opooladz/modded-nanogpt-psgd/blob/dc7c78082ac15fbf326f1bacd9e0ead0a2b45908/kron_mu.py

    A-law compression
    Args:
        x: Input tensor
        A: Compression parameter (default 87.6 - European PCM standard)
    """
    xa = torch._foreach_abs(x)
    torch._foreach_mul_(xa, A)
    [torch.where(x_ < 1, x_, 1 + torch.log_(x_), out=x_) for x_ in xa]
    [xa_.copysign(x_) for x_, xa_ in zip(x, xa)]
    torch._foreach_mul_(xa, 1 / (1 + math.log(A)))
    return xa


def identity(x):
    return x


@decorator_knowngood
def _compilable_trust_region_clip_(grad, lerp: float = 0.9, scale: float = 1.5):
    g32 = list(map(promote, grad))
    [g.mul_(1 / scale) for g in g32]
    tanh = torch._foreach_tanh(g32)
    torch._foreach_abs_(g32)
    torch._foreach_log1p_(g32)
    [g.copysign_(t).lerp_(t, lerp).mul_(scale) for t, g in zip(tanh, g32)]

    torch._foreach_maximum_(g32, -2)
    torch._foreach_minimum_(g32, 2)
    return [stochastic_round_(grad, g32) for grad, g32 in zip(grad, g32)]


def trust_region_clip_(grad, lerp=0.9, scale=1.5):
    grad = list_guard(grad)
    lerp, scale = scalar_guard(lerp, grad[0]), scalar_guard(scale, grad[0])
    return _compilable_trust_region_clip_(grad, lerp, scale)


@decorator
def triu_to_line(Q_list: List[Tensor]):
    out = []
    for q in Q_list:
        if q.dim() < 2:
            out.append((None, q))
        else:
            out.append((q.shape, q[tuple(torch.triu_indices(*q.shape))]))
    return out


def _triu_shape(numel):
    n = int((2 * numel) ** 0.5)
    assert n * (n + 1) == 2 * numel
    return n, n


@decorator
def line_to_triu(Q_list: List[Tuple[Optional[List[int]], Tensor]]):
    new = []
    for shape, q in Q_list:
        if shape is not None:
            shape = _triu_shape(q.numel())
            x = torch.zeros(shape, device=q.device, dtype=q.dtype)
            x[tuple(torch.triu_indices(*shape, device=q.device))] = q
            q = x
        new.append(q)
    return new


def update_triu_(q_state, materialised):
    for (shape0, q), (shape1, m) in zip(q_state, triu_to_line(materialised)):
        assert shape0 == shape1
        copy_stochastic_(q, m)


def psgd_should_update(group, prob: Union[float, callable], rng: Optional[random.Random] = None,
                       name: str = 'cumulative_prob'):
    group[f'{name}_prob_step'] = group.get(f'{name}_prob_step', 0) + 1
    if not isinstance(prob, float):
        prob = prob(group[f'{name}_prob_step'])
    if group['stochastic_schedule']:
        return rng.random() < prob
    cumulative_prob = state.get(name, 0)
    group[name] = cumulative_prob + prob
    return int(group[name]) > int(cumulative_prob)


@decorator_knowngood
def precond_grad_cached_(expr: str, ea: Tensor, *cached_q: Tensor, cast: bool = True):
    md = min_dtype(list(cached_q) + [ea])
    args = [q.to(md) for q in cached_q]
    args = args + [ea.to(md)]
    new = torch.einsum(expr, *args)
    if cast:
        return new.to(ea.dtype)
    return new


@decorator_knowngood
def _compilable_fused_precond_grad_cached_(expr: str, ea: Tensor, param, lr, grad, decay, caution, *cached_q: Tensor):
    precond = precond_grad_cached_(expr, ea, *cached_q, cast=False)
    update_param_(param, precond, lr, decay, caution=caution, grad=grad)


def fused_precond_grad_cached_(expr: str, ea: Tensor, param, lr, grad, decay, caution, *cached_q: Tensor):
    lr = scalar_guard(lr, param[0])
    _compilable_fused_precond_grad_cached_(expr, ea, param, lr, grad, decay, caution, *cached_q)


@decorator_knowngood
def psgd_precond_grad(expr: str, ea: Tensor, *preconds: Tensor):
    md = min_dtype(list(preconds) + [ea])
    args = [q.to(md) for q in preconds]
    args = args + args + [ea.to(md)]
    new = torch.einsum(expr, *args)
    return new.to(ea.dtype)


def _compilable_fused_psgd_precond_grad(expr: str, ea: Tensor, param, lr, grad, decay, caution, *preconds: Tensor):
    precond = psgd_precond_grad(expr, grad, *preconds)
    update_param_(param, precond, lr, decay, caution=caution, grad=grad)


def fused_psgd_precond_grad(expr: str, ea: Tensor, param, lr, grad, decay, caution, *preconds: Tensor):
    lr = scalar_guard(lr, param[0])
    _compilable_fused_psgd_precond_grad(expr, ea, param, lr, grad, decay, caution, *preconds)


@decorator_knowngood
def _compilable_mars_correction_(g: Tensor, old_g: Tensor, a: Tensor):
    g_copy = [g_.clone() for g_ in g]
    _compilable_stochastic_lerp_(g, old_g, a)
    copy_stochastic_list_(old_g, g_copy)


def mars_correction(g, old_g, beta1, gamma):
    a = -gamma * beta1 / (1 - beta1)
    g, old_g = list_guard(g), list_guard(old_g)
    a = scalar_guard(a, g[0])
    _compilable_mars_correction_(g, old_g, a)


@decorator_knowngood
def _compilable_cautioning_(g: Tensor, update: Tensor):
    mask = (g * update) > 0
    update.masked_fill_(~mask, 0)
    scale = mask.numel() / mask.sum().clamp(min=1)
    update.mul_(scale)


def caution(g, update):
    _compilable_cautioning_(g, update)


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    def _schedule(n):
        if n < flat_start:  # higher numerical stability
            return max_prob

        n -= flat_start
        prob = max_prob * math.exp(-decay * (n - flat_start))
        return max(min_prob, min(max_prob, prob))

    return _schedule


def merge_group(group, *tensors):
    if not group.get('merge_dims', False):
        return tensors
    if isinstance(tensors[0], list):
        return [merge_group(group, *t) for t in tensors]

    out = []
    for t in tensors:
        append_or_extend(out, dim_merger(t, group['max_size_triangular'] if 'max_size_triangular' in group else group[
            'max_precond_dim'], group.get('split', False)))
    return out


def hook_optimizer_into_model(model, optimizer, *args, **kwargs):
    def _step(p: Tensor, o: torch.optim.Optimizer):
        o.step()
        o.zero_grad()

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(functools.partial(_step, o=optimizer([p], *args, **kwargs)))
