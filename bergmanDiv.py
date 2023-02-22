import torch
import torch.nn.functional as F

from bert import BertModel


class MBPP(object):
    """Momentum Bregman Proximal Point Optimization (or 'Mean Teacher')
    Source: https://arxiv.org/pdf/1703.01780.pdf"""

    def __init__(self,
                 model: BertModel,
                 beta: float = 0.8,
                 mu: float = 1
    ):
        self.model = model
        self.beta = beta
        self.mu = mu
        self.theta_state = {}
        for name, param in self.model.named_parameters():
            self.theta_state[name] = param.data

    def apply_momentum(self, named_parameters):
        for name, param in named_parameters:
            self.theta_state[name] = (1-self.beta) * param.data.clone() + self.beta * self.theta_state[name]

    def bregman_divergence(self, batch, logits):
        theta_prob = F.softmax(logits, dim=-1)

        param_bak = {}
        for name, param in self.model.named_parameters():
            param_bak[name] = param.data.clone()
            param.data = self.theta_state[name]

        with torch.no_grad():
            theta_til_prob = F.softmax(self.model(*batch), dim=-1)

        for name, param in self.model.named_parameters():
            param.data = param_bak[name]

        bregman_divergence = F.kl_div(theta_prob.log(), theta_til_prob, reduction='batchmean') + \
            F.kl_div(theta_til_prob.log(), theta_prob, reduction='batchmean')

        return self.mu * bregman_divergence
