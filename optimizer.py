from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                #
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                # extract hyperparameters from group object
                beta_1, beta_2 = group["betas"][0], group["betas"][1]
                weight_decay = group["weight_decay"]
                eps = group["eps"]

                # initialize or access state variables
                m = torch.zeros_like(p.data, memory_format=torch.preserve_format) if 'm' not in state else state['m']
                v = torch.zeros_like(p.data, memory_format=torch.preserve_format) if 'v' not in state else state['v']
                t = 0 if 't' not in state else state['t']

                # increment t
                t += 1

                # update first and second moments of gradient
                m = beta_1 * m + (1-beta_1) * grad
                v = beta_2 * v + (1-beta_2) * grad**2

                # update parameters (p.data)
                if group["correct_bias"]:   # apply bias correction and update
                    alpha_t = alpha * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)  # optimized version
                    denominator = torch.sqrt(v).add(eps)
                    p.data.addcdiv_(m, denominator, value=-alpha_t)

                else:  # update without bias correction
                    denominator = torch.sqrt(v).add(eps)
                    p.data.addcdiv_(m, denominator, value=-alpha)

                # apply weight decay to gradient
                p.data -= weight_decay * alpha * p.data

                # update state
                state['t'], state['m'], state['v'] = t, m, v
                p.grad = grad

        return loss
